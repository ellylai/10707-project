import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class GraphAttention():
    pass

class W2V2_AASIST(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        )

        self.proj = nn.Linear(1024, 256)

        self.temporal_gat = GraphAttention(256)
        self.spectral_gat = GraphAttention(256)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(256, 2)

    def forward(self, x):

        x = self.encoder(x).last_hidden_state   # (B,T,1024)
        x = self.proj(x)                        # (B,T,256)

        x = self.temporal_gat(x)

        x = x.transpose(1,2)
        x = self.spectral_gat(x)

        x = self.pool(x).squeeze(-1)

        return self.classifier(x)