'''
DUALSTREAM: an audio deepfake detection model

Prosodic features (4) = fundamental frequency F_0, jitter, shimmer, harmonicity (HNR)
Digital features (embeddings) = wav2vec2 replaces mel-spectrogram

Fusion mechanism = multi-head cross attention

Input = .wav file

[digistream]                                            [biostream] 
Wav2vec2-XLSR → embedding                               Feature Extractor → prosodic features
Embedding → Convolutional Transformer → output          Prosodic features → LSTM → output

Prosodic output + Digital output → Fusion mechanism
Fusion mechanism → sigmoid → binary classification
Output = binary classification
'''
import torch
import torch.nn as nn

from acousticstream import AcousticStream, BinaryClassifier
from articulatorystream import ArticStream

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=2048, num_heads=8):
        super().__init__()
        # Multi-Head Attention where:
        # Query = Stream A
        # Key/Value = Stream B
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x_acoustic, x_prosodic):
        """
        x_acoustic: (batch, seq_len, 2048)
        x_prosodic: (batch, seq_len, 2048) 
        """
        # 1. Cross-Attention: Acoustic attends to Prosodic
        # Query: Acoustic, Key: Prosodic, Value: Prosodic
        attn_output, _ = self.multihead_attn(x_acoustic, x_prosodic, x_prosodic)
        
        # 2. Residual connection & Norm
        x = self.norm(attn_output + x_acoustic)
        
        # 3. Feed Forward
        x = self.ffn(x) + x
        
        # 4. Global Average Pooling to get (Batch, 2048)
        return x.mean(dim=1)

class DUALSTREAM(nn.Module):
    """
    Input: raw 1D .wav waveform
    Output: binary classification
    """
    def __init__(self, args):
        self.acoustic = AcousticStream() # pre-sigmoid output is (B, 2048)
        self.prosodic = ArticStream(...)
        self.fusion = CrossAttentionFusion()
        self.classifier = BinaryClassifier() # (B, 2048) -> (B, 2)
            
    def forward(self, x):
        acoustic_output = self.acoustic.forward(x) # (B, 2048)
        prosodic_output = self.prosodic.forward(x) # (B, 2048)
        
        # fuse the output of the two streams using cross attention
        features = self.fusion.forward(acoustic_output, prosodic_output) # (B, 2048)
        
        # binary classifier head on fused features
        output = self.classifier.forward(features) # (B, 2)
        return output
    