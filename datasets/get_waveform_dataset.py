from pathlib import Path

import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset


class AudioDatasetFromCSV(Dataset):
    """
    Generic audio dataset that:
    - Reads file paths + labels from a CSV
    - Loads .wav files
    - Resamples to target sample rate
    - Converts to mono
    - Pads / truncates to fixed length

    Expected CSV columns:
        path,label,split
    """

    def __init__(
        self,
        csv_path,
        split,
        sample_rate=16000,
        max_seconds=4.0,
    ):
        self.csv_path = Path(csv_path)
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_seconds)

        df = pd.read_csv(self.csv_path)

        if "split" not in df.columns:
            raise ValueError("CSV must contain a 'split' column")

        self.df = df[df["split"] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No data found for split='{split}'")

    def __len__(self):
        return len(self.df)

    def _fix_length(self, wav):
        """Pad or truncate waveform to fixed length"""
        if wav.size(0) > self.max_len:
            wav = wav[:self.max_len]
        elif wav.size(0) < self.max_len:
            wav = F.pad(wav, (0, self.max_len - wav.size(0)))
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav_path = row["path"]
        label = int(row["label"])

        # --- Load audio safely ---
        try:
            waveform, sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"[WARNING] Failed to load: {wav_path}")
            # fallback: try next sample
            return self.__getitem__((idx + 1) % len(self))

        # --- Convert to mono ---
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # --- Resample if needed ---
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr,
                new_freq=self.sample_rate,
            )(waveform)

        # --- Remove channel dim ---
        waveform = waveform.squeeze(0)

        # --- Normalize (simple peak normalization) ---
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

        # --- Fix length ---
        waveform = self._fix_length(waveform)

        return {
            "input_values": waveform,                      # (num_samples,)
            "label": torch.tensor(label, dtype=torch.long),
            "path": wav_path,
        }

# to load datasets splits example
# from torch.utils.data import DataLoader
# from datasets.audio_dataset_csv import AudioDatasetFromCSV

# CSV_PATH = "data/speechfake_splits.csv"

# train_dataset = AudioDatasetFromCSV(CSV_PATH, split="train")
# val_dataset   = AudioDatasetFromCSV(CSV_PATH, split="val")
# test_dataset  = AudioDatasetFromCSV(CSV_PATH, split="test")

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
# val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
# test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)