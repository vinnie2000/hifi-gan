from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from env import AttrDict
from meldataset import MAX_WAV_VALUE, load_wav, mel_spectrogram
from models import Generator
from stream_inference import StreamingHiFiGAN, normalize_mel


def load_config(config_file):
    with open(config_file) as f:
        return AttrDict(json.loads(f.read()))


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def get_mel(wav, h):
    return mel_spectrogram(wav, h.n_fft, h.num_mels, h.sampling_rate,
                           h.hop_size, h.win_size, h.fmin, h.fmax)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def conv1d_flops(module, inputs, output):
    x = inputs[0]
    batch = output.shape[0]
    out_channels = output.shape[1]
    out_length = output.shape[2]
    kernel_ops = module.kernel_size[0] * (module.in_channels // module.groups)
    return batch * out_channels * out_length * kernel_ops * 2


def estimate_flops(model, mel_shape, device):
    flops = {'total': 0}
    hooks = []

    def hook(module, inputs, output):
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            flops['total'] += conv1d_flops(module, inputs, output)

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            hooks.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    x = torch.zeros(mel_shape, device=device)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    return flops['total']


def estimate_streaming_flops(model, h, mel_frames, chunk_frames, context_frames, device):
    total_flops = 0
    buffer_start_frame = 0
    buffer_end_frame = 0
    emitted_until_frame = 0

    for start in range(0, mel_frames, chunk_frames):
        end = min(mel_frames, start + chunk_frames)
        buffer_end_frame += end - start
        emit_end_frame = max(emitted_until_frame, buffer_end_frame - context_frames)
        if emit_end_frame > emitted_until_frame:
            gen_start_frame = max(buffer_start_frame, emitted_until_frame - context_frames)
            gen_end_frame = min(buffer_end_frame, emit_end_frame + context_frames)
            total_flops += estimate_flops(model, (1, h.num_mels, gen_end_frame - gen_start_frame), device)
            emitted_until_frame = emit_end_frame

        keep_from_frame = max(buffer_start_frame, emitted_until_frame - context_frames)
        buffer_start_frame = keep_from_frame

    if buffer_end_frame > emitted_until_frame:
        gen_start_frame = max(buffer_start_frame, emitted_until_frame - context_frames)
        gen_end_frame = buffer_end_frame
        total_flops += estimate_flops(model, (1, h.num_mels, gen_end_frame - gen_start_frame), device)
    return total_flops


def peak_allocated_mb(device):
    if device.type != 'cuda':
        return None
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024


def reset_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def full_inference(generator, mel):
    with torch.no_grad():
        y = generator(mel)
    audio = y.squeeze()
    audio = audio * MAX_WAV_VALUE
    return audio.cpu().numpy().astype('int16')


def streaming_inference(generator, mel_np, h, device, chunk_frames, context_frames):
    mel_np = normalize_mel(mel_np)
    streamer = StreamingHiFiGAN(generator, h, device, context_frames)
    chunks = []
    first_chunk_latency = None

    for start in range(0, mel_np.shape[-1], chunk_frames):
        end = min(mel_np.shape[-1], start + chunk_frames)
        sync(device)
        t0 = time.perf_counter()
        audio_chunk = streamer.push(mel_np[:, :, start:end])
        sync(device)
        if first_chunk_latency is None and len(audio_chunk) > 0:
            first_chunk_latency = time.perf_counter() - t0
        chunks.append(audio_chunk)

    sync(device)
    t0 = time.perf_counter()
    tail = streamer.flush()
    sync(device)
    if first_chunk_latency is None and len(tail) > 0:
        first_chunk_latency = time.perf_counter() - t0
    chunks.append(tail)

    if chunks:
        out = np.concatenate(chunks)
    else:
        out = np.zeros(0, dtype=np.float32)
    audio = (np.clip(out, -1.0, 1.0) * MAX_WAV_VALUE).astype('int16')
    return audio, first_chunk_latency


def load_generator(h, checkpoint_file, device):
    generator = Generator(h).to(device)
    if checkpoint_file:
        state_dict_g = load_checkpoint(checkpoint_file, device)
        generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator


def measure_full(generator, wav_tensor, h, device, warmup, runs):
    for _ in range(warmup):
        mel = get_mel(wav_tensor, h)
        sync(device)
        with torch.no_grad():
            generator(mel)
        sync(device)

    total_times = []
    vocoder_times = []
    peak_memory = []
    audio = None

    for _ in range(runs):
        reset_peak_memory(device)
        sync(device)
        total_start = time.perf_counter()
        mel = get_mel(wav_tensor, h)
        sync(device)
        vocoder_start = time.perf_counter()
        audio = full_inference(generator, mel)
        sync(device)
        vocoder_time = time.perf_counter() - vocoder_start
        total_time = time.perf_counter() - total_start
        total_times.append(total_time)
        vocoder_times.append(vocoder_time)
        peak_memory.append(peak_allocated_mb(device))

    return {
        'total_time_sec': float(np.mean(total_times)),
        'vocoder_time_sec': float(np.mean(vocoder_times)),
        'latency_sec': float(np.mean(vocoder_times)),
        'peak_allocated_mb': None if peak_memory[0] is None else float(np.max(peak_memory)),
        'output_duration_sec': len(audio) / h.sampling_rate,
    }, audio


def measure_streaming(generator, wav_tensor, h, device, chunk_frames, context_frames, warmup, runs):
    for _ in range(warmup):
        mel = get_mel(wav_tensor, h)
        mel_np = mel.cpu().numpy()
        streaming_inference(generator, mel_np, h, device, chunk_frames, context_frames)

    total_times = []
    vocoder_times = []
    latencies = []
    peak_memory = []
    audio = None

    for _ in range(runs):
        reset_peak_memory(device)
        sync(device)
        total_start = time.perf_counter()
        mel = get_mel(wav_tensor, h)
        mel_np = mel.cpu().numpy()
        sync(device)
        vocoder_start = time.perf_counter()
        audio, first_chunk_latency = streaming_inference(generator, mel_np, h, device,
                                                         chunk_frames, context_frames)
        sync(device)
        vocoder_time = time.perf_counter() - vocoder_start
        total_time = time.perf_counter() - total_start
        total_times.append(total_time)
        vocoder_times.append(vocoder_time)
        latencies.append(first_chunk_latency)
        peak_memory.append(peak_allocated_mb(device))

    return {
        'total_time_sec': float(np.mean(total_times)),
        'vocoder_time_sec': float(np.mean(vocoder_times)),
        'latency_sec': float(np.mean(latencies)),
        'peak_allocated_mb': None if peak_memory[0] is None else float(np.max(peak_memory)),
        'output_duration_sec': len(audio) / h.sampling_rate,
    }, audio


def build_model_specs(args):
    return [
        ('v1', args.v1_config, args.v1_checkpoint),
        ('v2', args.v2_config, args.v2_checkpoint),
        ('v3', args.v3_config, args.v3_checkpoint),
    ]


def format_value(value):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return '{:.6f}'.format(value)
    return str(value)


def print_rows(rows):
    columns = [
        'model', 'method', 'chunk_frames', 'context_frames', 'params',
        'flops', 'total_time_sec', 'vocoder_time_sec', 'latency_sec',
        'rtf', 'peak_allocated_mb', 'output_duration_sec'
    ]
    widths = {c: len(c) for c in columns}
    for row in rows:
        for c in columns:
            widths[c] = max(widths[c], len(format_value(row.get(c))))

    print('  '.join(c.ljust(widths[c]) for c in columns))
    print('  '.join('-' * widths[c] for c in columns))
    for row in rows:
        print('  '.join(format_value(row.get(c)).ljust(widths[c]) for c in columns))


def write_csv(rows, csv_file):
    if not csv_file:
        return
    columns = [
        'model', 'method', 'chunk_frames', 'context_frames', 'params',
        'flops', 'total_time_sec', 'vocoder_time_sec', 'latency_sec',
        'rtf', 'peak_allocated_mb', 'output_duration_sec'
    ]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_audio(output_dir, model_name, method, sampling_rate, audio):
    if not output_dir or audio is None:
        return
    from scipy.io.wavfile import write
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, '{}_{}.wav'.format(model_name, method))
    write(output_file, sampling_rate, audio)
    print(output_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', required=True)
    parser.add_argument('--chunk_frames', default=128, type=int)
    parser.add_argument('--context_frames', default=32, type=int)
    parser.add_argument('--warmup', default=2, type=int)
    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--csv_file', default=None)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--v1_config', default='config_v1.json')
    parser.add_argument('--v2_config', default='config_v2.json')
    parser.add_argument('--v3_config', default='config_v3.json')
    parser.add_argument('--v1_checkpoint', default=None)
    parser.add_argument('--v2_checkpoint', default=None)
    parser.add_argument('--v3_checkpoint', default=None)
    args = parser.parse_args()
    context_frames = args.context_frames

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wav, sr = load_wav(args.input_wav)
    wav = wav / MAX_WAV_VALUE
    rows = []

    for model_name, config_file, checkpoint_file in build_model_specs(args):
        h = load_config(config_file)
        torch.manual_seed(h.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(h.seed)
        wav_tensor = torch.FloatTensor(wav).to(device).unsqueeze(0)

        generator = load_generator(h, checkpoint_file, device)
        params = parameter_count(generator)
        mel_frames = get_mel(wav_tensor, h).shape[-1]
        full_flops = estimate_flops(generator, (1, h.num_mels, mel_frames), device)
        streaming_flops = estimate_streaming_flops(generator, h, mel_frames, args.chunk_frames,
                                                   context_frames, device)

        full_row = {
            'model': model_name,
            'method': 'full',
            'chunk_frames': None,
            'context_frames': None,
            'params': params,
            'flops': full_flops,
            'total_time_sec': None,
            'vocoder_time_sec': None,
            'latency_sec': None,
            'rtf': None,
            'peak_allocated_mb': None,
            'output_duration_sec': None,
        }
        streaming_row = {
            'model': model_name,
            'method': 'streaming',
            'chunk_frames': args.chunk_frames,
            'context_frames': context_frames,
            'params': params,
            'flops': streaming_flops,
            'total_time_sec': None,
            'vocoder_time_sec': None,
            'latency_sec': None,
            'rtf': None,
            'peak_allocated_mb': None,
            'output_duration_sec': None,
        }

        if checkpoint_file:
            full, full_audio = measure_full(generator, wav_tensor, h, device, args.warmup, args.runs)
            full['rtf'] = full['total_time_sec'] / full['output_duration_sec']
            full_row.update(full)
            write_audio(args.output_dir, model_name, 'full', h.sampling_rate, full_audio)

            streaming, streaming_audio = measure_streaming(generator, wav_tensor, h, device, args.chunk_frames,
                                                           context_frames, args.warmup, args.runs)
            streaming['rtf'] = streaming['total_time_sec'] / streaming['output_duration_sec']
            streaming_row.update(streaming)
            write_audio(args.output_dir, model_name, 'streaming', h.sampling_rate, streaming_audio)

        rows.append(full_row)
        rows.append(streaming_row)

    print_rows(rows)
    write_csv(rows, args.csv_file)


if __name__ == '__main__':
    main()
