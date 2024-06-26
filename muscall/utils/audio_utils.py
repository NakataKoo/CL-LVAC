import random

import torch
import torchaudio

from torch_audiomentations import (
    Compose,
    Gain,
    PolarityInversion,
    AddColoredNoise,
    PitchShift,
)

# データ拡張操作
def get_transform_chain(
    p_polarity,
    p_noise,
    p_gain,
    p_pitch_shift,
    sample_rate,
):
    train_transforms = [
        PolarityInversion(p=p_polarity), # 音声信号の極性を反転
        AddColoredNoise(p=p_noise), # 音声信号にランダムなノイズを追加
        Gain(p=p_gain), # 音声信号の振幅をランダムに変更（増幅または減衰）
        PitchShift(p=p_pitch_shift, sample_rate=sample_rate), # 音声信号のピッチ（音の高さ）をランダムに変更
    ]

    transform_chain = Compose(transforms=train_transforms)

    return transform_chain


class RandomResizedCrop(torch.nn.Module):
    def __init__(self, n_samples, pad=False):
        super().__init__()
        self.n_samples = n_samples
        self.pad = pad

    def forward(self, audio):
        max_samples = audio.shape[-1]
        start_idx = random.randint(0, max_samples - self.n_samples)
        audio = audio[..., start_idx : start_idx + self.n_samples]

        if self.pad:
            padding_size = max_samples - self.n_samples
            audio = torch.nn.functional.pad(audio, (0, padding_size), "constant", 0)

        return audio


def resample(waveform, source_sr, target_sr=16000):
    resampler = torchaudio.transforms.Resample(source_sr, target_sr)
    waveform = resampler(waveform)
    return waveform
