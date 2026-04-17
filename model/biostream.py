import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from shared_audio_processor import Segment
from articulatory_features import ArticulatoryFeatureExtractor
from prosodic_features import ProsodicFeatureExtractor


class BioStreamLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
        pooling: str = "mean",
        standalone: bool = False,
    ):
        super().__init__()

        if pooling not in {"mean", "last"}:
            raise ValueError("pooling must be 'mean' or 'last'")

        self.pooling = pooling
        self.standalone = standalone
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.feature_dim = lstm_out_dim

        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, 1),
        )

    def _mean_pool(self, x, lengths):
        B, T, _ = x.shape
        device = x.device
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        denom = lengths.clamp(min=1).unsqueeze(1).float()
        return summed / denom

    def _last_pool(self, x, lengths):
        idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(x.size(0), device=x.device)
        return x[batch_idx, idx]

    def forward(self, x, lengths):
        x = self.input_norm(x)

        packed = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        if self.pooling == "mean":
            pooled = self._mean_pool(out, lengths)
        else:
            pooled = self._last_pool(out, lengths)

        if self.standalone:
            logits = self.classifier(pooled).squeeze(1)  # [B]
            return logits

        return pooled  # [B, feature_dim = hidden_dim * 2 (if bidirectional)]