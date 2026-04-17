from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import parselmouth
import soundfile as sf
import torch


@dataclass
class Segment:
    phoneme: str
    phoneme_id: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    duration: float
    mean_confidence: float


AudioInput = Union[str, Path, np.ndarray, torch.Tensor]


class SharedAudioProcessor:
    """
    Shared audio utilities for articulatory and prosodic extractors.

    Supports:
      - file path input
      - numpy waveform input
      - torch waveform input

    Expected waveform shapes:
      - [T]
      - [1, T]
      - [C, T] -> averaged to mono
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        audio_path = Path(audio_path)
        wav, sr = sf.read(str(audio_path))

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        if sr != self.sample_rate:
            raise ValueError(
                f"Expected {self.sample_rate} Hz audio, got {sr} Hz. "
                "Resample audio before calling this pipeline."
            )

        return wav.astype(np.float32)

    def prepare_audio_array(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().float().numpy()

        wav = np.asarray(audio, dtype=np.float32)

        if wav.ndim == 2:
            if wav.shape[0] == 1:
                wav = wav[0]
            else:
                wav = wav.mean(axis=0)

        if wav.ndim != 1:
            raise ValueError(f"Expected waveform shape [T], [1, T], or [C, T], got {wav.shape}")

        return wav.astype(np.float32)

    def load_audio_any(self, audio_input: AudioInput) -> np.ndarray:
        if isinstance(audio_input, (str, Path)):
            return self.load_audio(audio_input)
        return self.prepare_audio_array(audio_input)

    def load_audio_torch_any(self, audio_input: AudioInput, device: str) -> torch.Tensor:
        wav = self.load_audio_any(audio_input)
        return torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0)

    def load_sound_any(self, audio_input: AudioInput) -> parselmouth.Sound:
        wav = self.load_audio_any(audio_input)
        return parselmouth.Sound(wav, sampling_frequency=self.sample_rate)

    @staticmethod
    def safe_mean(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.mean(x))

    @staticmethod
    def safe_std(x: np.ndarray) -> float:
        if x.size == 0:
            return 0.0
        return float(np.std(x))

    @staticmethod
    def finite_only(x: np.ndarray) -> np.ndarray:
        return x[np.isfinite(x)]

    @staticmethod
    def pooled_stats(sequence: np.ndarray) -> np.ndarray:
        if sequence.shape[0] == 0:
            return np.zeros(sequence.shape[1] * 2, dtype=np.float32)

        mean_vec = sequence.mean(axis=0)
        std_vec = sequence.std(axis=0)
        return np.concatenate([mean_vec, std_vec], axis=0).astype(np.float32)