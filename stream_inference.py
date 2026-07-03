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


def normalize_mel(mel_np):
    if mel_np.ndim == 2:
        mel_np = mel_np[np.newaxis, ...]
    if mel_np.ndim != 3:
        raise ValueError('mel must have shape (num_mels, frames) or (1, num_mels, frames)')
    if mel_np.shape[0] != 1:
        mel_np = mel_np[0:1]
    return mel_np.astype(np.float32, copy=False)


class StreamingHiFiGAN:
    """带有限 mel 上下文和 lookahead 的增量 HiFi-GAN 推理器。"""

    def __init__(self, generator, h, device, context_frames=32):
        if context_frames < 0:
            raise ValueError('context_frames must be >= 0')
        self.generator = generator
        self.h = h
        self.device = device
        self.context_frames = context_frames
        self.hop_size = h.hop_size
        self.buffer = None
        self.buffer_start_frame = 0
        self.emitted_until_frame = 0
        self.finished = False

    def push(self, mel_chunk, final=False):
        if self.finished:
            raise RuntimeError('cannot push after final=True')

        mel_chunk = normalize_mel(mel_chunk)
        if mel_chunk.shape[1] != self.h.num_mels:
            raise ValueError('expected {} mel channels, got {}'.format(self.h.num_mels, mel_chunk.shape[1]))

        self._append(mel_chunk)
        emit_end_frame = self._next_emit_end_frame(final)

        audio = self._emit_until(emit_end_frame)
        self._trim_buffer()
        self.finished = final
        return audio

    def flush(self):
        # flush 不追加新 mel；空 chunk 只是触发 final=True 的收尾路径。
        return self.push(np.zeros((1, self.h.num_mels, 0), dtype=np.float32), final=True)

    def _append(self, mel_chunk):
        if self.buffer is None:
            self.buffer = mel_chunk
        elif mel_chunk.shape[-1] > 0:
            self.buffer = np.concatenate([self.buffer, mel_chunk], axis=-1)

    def _next_emit_end_frame(self, final):
        buffer_end_frame = self._buffer_end_frame()
        if final:
            return buffer_end_frame
        safe_until_frame = buffer_end_frame - self.context_frames
        return max(self.emitted_until_frame, safe_until_frame)

    def _emit_until(self, emit_end_frame):
        emit_start_frame = self.emitted_until_frame
        if emit_end_frame <= emit_start_frame:
            return np.zeros(0, dtype=np.float32)

        gen_start_frame, gen_end_frame = self._generator_frame_range(emit_start_frame, emit_end_frame)
        mel_context = self._buffer_slice(gen_start_frame, gen_end_frame)
        generated_audio = self._run_generator(mel_context)
        audio = self._crop_audio(generated_audio, gen_start_frame, emit_start_frame, emit_end_frame)

        self.emitted_until_frame = emit_end_frame
        return audio

    # 输出一段音频后，把以后用不到的旧 mel 从 buffer 前面删掉，避免 buffer 越来越长
    def _trim_buffer(self):
        keep_from_frame = max(self.buffer_start_frame, self.emitted_until_frame - self.context_frames)
        drop_frames = keep_from_frame - self.buffer_start_frame
        if drop_frames > 0:
            self.buffer = self.buffer[:, :, drop_frames:]
            self.buffer_start_frame = keep_from_frame

    # 计算当前已经收到的 mel 到全局第几帧结束
    def _buffer_end_frame(self):
        return self.buffer_start_frame + self.buffer.shape[-1]

    # 根据本次真正要输出的 mel 范围，计算实际应该送进 HiFi-GAN generator 的 mel 范围
    def _generator_frame_range(self, emit_start_frame, emit_end_frame):
        buffer_end_frame = self._buffer_end_frame()
        gen_start_frame = max(self.buffer_start_frame, emit_start_frame - self.context_frames)
        gen_end_frame = min(buffer_end_frame, emit_end_frame + self.context_frames)
        return gen_start_frame, gen_end_frame

    def _buffer_slice(self, start_frame, end_frame):
        # start_frame/end_frame 是全局帧编号，切 buffer 前要转换成局部下标。
        local_start = start_frame - self.buffer_start_frame
        local_end = end_frame - self.buffer_start_frame
        return self.buffer[:, :, local_start:local_end]

    def _run_generator(self, mel_context):
        x = torch.from_numpy(mel_context).to(self.device)
        with torch.no_grad():
            y = self.generator(x)
        return y.squeeze(0).squeeze(0).cpu().numpy()

    def _crop_audio(self, generated_audio, gen_start_frame, emit_start_frame, emit_end_frame):
        crop_start = (emit_start_frame - gen_start_frame) * self.hop_size
        crop_end = crop_start + (emit_end_frame - emit_start_frame) * self.hop_size
        return np.clip(generated_audio[crop_start:crop_end], -1.0, 1.0)


def streaming_inference_array(generator, x_np, h, device, chunk_frames=128, context_frames=32):
    if chunk_frames <= 0:
        raise ValueError('chunk_frames must be > 0')

    x_np = normalize_mel(x_np)
    streamer = StreamingHiFiGAN(generator, h, device, context_frames)
    chunks = []
    total_frames = x_np.shape[-1]

    for start in range(0, total_frames, chunk_frames):
        end = min(total_frames, start + chunk_frames)
        chunks.append(streamer.push(x_np[:, :, start:end]))
    chunks.append(streamer.flush())

    if chunks:
        audio = np.concatenate(chunks)
    else:
        audio = np.zeros(0, dtype=np.float32)
    return (audio * MAX_WAV_VALUE).astype('int16')


def inference_mel_file(generator, mel_file, output_dir, chunk_frames, overlap_frames):
    x = np.load(mel_file)
    audio = streaming_inference_array(generator, x, h, device, chunk_frames, overlap_frames)

    from scipy.io.wavfile import write
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(mel_file))[0] + '_stream_generated.wav')
    write(output_file, h.sampling_rate, audio)
    print(output_file)


def inference_wav_file(generator, wav_file, output_dir, chunk_frames, overlap_frames):
    wav, sr = load_wav(wav_file)
    wav = wav / MAX_WAV_VALUE
    wav = torch.FloatTensor(wav).to(device)
    x = get_mel(wav.unsqueeze(0)).cpu().numpy()
    audio = streaming_inference_array(generator, x, h, device, chunk_frames, overlap_frames)

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
                        help='Number of new mel frames to feed per streaming step')
    parser.add_argument('--overlap_frames', default=32, type=int,
                        help='Left/right mel context frames. Non-final output is delayed by this many frames.')
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
