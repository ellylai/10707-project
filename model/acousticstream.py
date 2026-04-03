import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class ConvolutionalTransformer(nn.Module):
    """
    Takes in embeddings of dimension 1024
    Wav2Vec2-XLSR -> (batch_size, sequence_length, 1024)
    """

    def __init__(self, embed_dim=1024, num_heads=8, ff_dim=2048):
        super().__init__()
        # Local context extraction
        self.conv_subsample = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Self-Attention
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Input: x, shape (batch, seq_len, 1024)
        """
        # x shape from Wav2vec: (batch, seq_len, 1024)
        x = x.transpose(1, 2)
        x = self.conv_subsample(x)
        x = x.transpose(1, 2)

        x = self.layer_norm(x)

        x = self.transformer(x)
        x = x.transpose(1, 2)
        return self.pool(x).squeeze(-1)  # Output: (batch, 1024)


class AcousticStream(nn.Module):
    def __init__(self, standalone: bool = False):
        super(AcousticStream, self).__init__()
        self.embedder = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        )  # -> (batch, seq_len, 1024)
        for param in self.embedder.parameters():
            param.requires_grad = False
        self.conv_transformer = ConvolutionalTransformer()  # -> (batch, 1024)
        self.standalone = standalone
        if standalone:
            self.classifier = BinaryClassifier()

    def forward(self, x):
        x = self.embedder(x).last_hidden_state  # (batch, seq_len, 1024)
        output = self.conv_transformer(x)

        if self.standalone:
            output = self.classifier(output)  # (batch, 2)
        return output  # (batch, 1024)


class BinaryClassifier(nn.Module):
    def __init__(
        self, input_size=1024, hidden_size=512
    ):  # chose 512 for no particular reason
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
