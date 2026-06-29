from __future__ import absolute_import, division, print_function, unicode_literals

import os
import glob
import argparse
import json
import numpy as np
import torch
from env import AttrDict
from models import Generator
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def hann_window(length):
    if length <= 1:
        return np.ones(length, dtype=np.float32)
    return np.hanning(length).astype(np.float32)


def chunked_inference_file(generator, x_np, h, device, chunk_frames=128, overlap_frames=32):
    # x_np expected shape: (1, num_mels, T) or (num_mels, T)
    if x_np.ndim == 2:
        x_np = x_np[np.newaxis, ...]
    if x_np.ndim == 3 and x_np.shape[0] != 1:
        # keep only first batch if somehow present
        x_np = x_np[0:1]

    x = torch.FloatTensor(x_np).to(device)
    _, n_mels, T = x.shape

    hop = h.hop_size
    total_samples = int(T * hop)
    out = np.zeros(total_samples, dtype=np.float32)
    weight = np.zeros(total_samples, dtype=np.float32)

    step = max(1, chunk_frames - overlap_frames)

    for start in range(0, T, step):
        s = max(0, start - overlap_frames)
        e = min(T, start + chunk_frames + overlap_frames)
        chunk = x[:, :, s:e]

        with torch.no_grad():
            y = generator(chunk)

        # y shape: (B, 1, S)
        y = y.squeeze(0).squeeze(0).cpu().numpy()
        # compute sample positions
        start_sample = int(s * hop)
        end_sample = start_sample + y.shape[0]

        # build window with fades on chunk edges only where overlap exists
        L = y.shape[0]
        win = np.ones(L, dtype=np.float32)
        ov_samp = int(overlap_frames * hop)
        if s > 0 and ov_samp > 0:
            # there is left overlap
            ramp = np.linspace(0.0, 1.0, ov_samp, endpoint=False, dtype=np.float32)
            win[:ov_samp] = ramp
        if e < T and ov_samp > 0:
            # there is right overlap
            ramp = np.linspace(1.0, 0.0, ov_samp, endpoint=False, dtype=np.float32)
            win[-ov_samp:] = ramp

        # accumulate
        out[start_sample:end_sample] += y * win
        weight[start_sample:end_sample] += win

    # normalize
    nonzero = weight > 1e-8
    out[nonzero] = out[nonzero] / weight[nonzero]

    # clip and convert
    out = np.clip(out, -1.0, 1.0)
    audio = (out * MAX_WAV_VALUE).astype('int16')
    return audio


def inference_mel_file(generator, mel_file, output_dir, chunk_frames, overlap_frames):
    x = np.load(mel_file)
    audio = chunked_inference_file(generator, x, h, device, chunk_frames, overlap_frames)

    from scipy.io.wavfile import write
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(mel_file))[0] + '_stream_generated.wav')
    write(output_file, h.sampling_rate, audio)
    print(output_file)


def inference_wav_file(generator, wav_file, output_dir, chunk_frames, overlap_frames):
    wav, sr = load_wav(wav_file)
    wav = wav / MAX_WAV_VALUE
    wav = torch.FloatTensor(wav).to(device)
    x = get_mel(wav.unsqueeze(0)).cpu().numpy()
    audio = chunked_inference_file(generator, x, h, device, chunk_frames, overlap_frames)

    from scipy.io.wavfile import write
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(wav_file))[0] + '_stream_generated.wav')
    write(output_file, h.sampling_rate, audio)
    print(output_file)


def inference(a):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()

    if a.input_wavs_dir:
        filelist = sorted(os.listdir(a.input_wavs_dir))
        for filname in filelist:
            if not filname.lower().endswith('.wav'):
                continue
            inference_wav_file(generator, os.path.join(a.input_wavs_dir, filname), a.output_dir,
                               a.chunk_frames, a.overlap_frames)
    else:
        filelist = sorted(os.listdir(a.input_mels_dir))
        for filname in filelist:
            if not filname.lower().endswith('.npy'):
                continue
            inference_mel_file(generator, os.path.join(a.input_mels_dir, filname), a.output_dir,
                               a.chunk_frames, a.overlap_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--input_wavs_dir', default=None,
                        help='Directory containing wav files. If set, wav input is used instead of mel input.')
    parser.add_argument('--output_dir', default='generated_stream')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--chunk_frames', default=128, type=int,
                        help='Number of mel frames per chunk (excludes overlap)')
    parser.add_argument('--overlap_frames', default=32, type=int,
                        help='Number of mel frames to overlap between chunks')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
