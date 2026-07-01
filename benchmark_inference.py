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


def estimate_chunked_flops(model, h, mel_frames, chunk_frames, overlap_frames, device):
    total_flops = 0
    step = max(1, chunk_frames - overlap_frames)
    for start in range(0, mel_frames, step):
        s = max(0, start - overlap_frames)
        e = min(mel_frames, start + chunk_frames + overlap_frames)
        total_flops += estimate_flops(model, (1, h.num_mels, e - s), device)
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


def chunked_inference(generator, mel_np, h, device, chunk_frames, overlap_frames):
    if mel_np.ndim == 2:
        mel_np = mel_np[np.newaxis, ...]
    if mel_np.ndim == 3 and mel_np.shape[0] != 1:
        mel_np = mel_np[0:1]

    x = torch.FloatTensor(mel_np).to(device)
    _, _, T = x.shape

    hop = h.hop_size
    total_samples = int(T * hop)
    out = np.zeros(total_samples, dtype=np.float32)
    weight = np.zeros(total_samples, dtype=np.float32)
    step = max(1, chunk_frames - overlap_frames)
    first_chunk_latency = None

    for start in range(0, T, step):
        s = max(0, start - overlap_frames)
        e = min(T, start + chunk_frames + overlap_frames)
        chunk = x[:, :, s:e]

        sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            y = generator(chunk)
        sync(device)
        if first_chunk_latency is None:
            first_chunk_latency = time.perf_counter() - t0

        y = y.squeeze(0).squeeze(0).cpu().numpy()
        start_sample = int(s * hop)
        end_sample = start_sample + y.shape[0]

        L = y.shape[0]
        win = np.ones(L, dtype=np.float32)
        ov_samp = int(overlap_frames * hop)
        if s > 0 and ov_samp > 0:
            win[:ov_samp] = np.linspace(0.0, 1.0, ov_samp, endpoint=False, dtype=np.float32)
        if e < T and ov_samp > 0:
            win[-ov_samp:] = np.linspace(1.0, 0.0, ov_samp, endpoint=False, dtype=np.float32)

        out[start_sample:end_sample] += y * win
        weight[start_sample:end_sample] += win

    nonzero = weight > 1e-8
    out[nonzero] = out[nonzero] / weight[nonzero]
    out = np.clip(out, -1.0, 1.0)
    audio = (out * MAX_WAV_VALUE).astype('int16')
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
    }


def measure_chunked(generator, wav_tensor, h, device, chunk_frames, overlap_frames, warmup, runs):
    for _ in range(warmup):
        mel = get_mel(wav_tensor, h)
        mel_np = mel.cpu().numpy()
        chunked_inference(generator, mel_np, h, device, chunk_frames, overlap_frames)

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
        audio, first_chunk_latency = chunked_inference(generator, mel_np, h, device,
                                                       chunk_frames, overlap_frames)
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
    }


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
        'model', 'method', 'chunk_frames', 'overlap_frames', 'params',
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
        'model', 'method', 'chunk_frames', 'overlap_frames', 'params',
        'flops', 'total_time_sec', 'vocoder_time_sec', 'latency_sec',
        'rtf', 'peak_allocated_mb', 'output_duration_sec'
    ]
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wav', required=True)
    parser.add_argument('--chunk_frames', default=128, type=int)
    parser.add_argument('--overlap_frames', default=32, type=int)
    parser.add_argument('--warmup', default=2, type=int)
    parser.add_argument('--runs', default=5, type=int)
    parser.add_argument('--csv_file', default=None)
    parser.add_argument('--v1_config', default='config_v1.json')
    parser.add_argument('--v2_config', default='config_v2.json')
    parser.add_argument('--v3_config', default='config_v3.json')
    parser.add_argument('--v1_checkpoint', default=None)
    parser.add_argument('--v2_checkpoint', default=None)
    parser.add_argument('--v3_checkpoint', default=None)
    args = parser.parse_args()

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
        chunked_flops = estimate_chunked_flops(generator, h, mel_frames, args.chunk_frames,
                                               args.overlap_frames, device)

        full_row = {
            'model': model_name,
            'method': 'full',
            'chunk_frames': None,
            'overlap_frames': None,
            'params': params,
            'flops': full_flops,
            'total_time_sec': None,
            'vocoder_time_sec': None,
            'latency_sec': None,
            'rtf': None,
            'peak_allocated_mb': None,
            'output_duration_sec': None,
        }
        chunked_row = {
            'model': model_name,
            'method': 'chunked',
            'chunk_frames': args.chunk_frames,
            'overlap_frames': args.overlap_frames,
            'params': params,
            'flops': chunked_flops,
            'total_time_sec': None,
            'vocoder_time_sec': None,
            'latency_sec': None,
            'rtf': None,
            'peak_allocated_mb': None,
            'output_duration_sec': None,
        }

        if checkpoint_file:
            full = measure_full(generator, wav_tensor, h, device, args.warmup, args.runs)
            full['rtf'] = full['total_time_sec'] / full['output_duration_sec']
            full_row.update(full)

            chunked = measure_chunked(generator, wav_tensor, h, device, args.chunk_frames,
                                      args.overlap_frames, args.warmup, args.runs)
            chunked['rtf'] = chunked['total_time_sec'] / chunked['output_duration_sec']
            chunked_row.update(chunked)

        rows.append(full_row)
        rows.append(chunked_row)

    print_rows(rows)
    write_csv(rows, args.csv_file)


if __name__ == '__main__':
    main()
