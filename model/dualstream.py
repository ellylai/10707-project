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
from biostream import BioStreamLSTM

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
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

    def forward(self, x_acoustic, x_biometric):
        # 1. Cross-Attention: Acoustic attends to Biometric
        # Query: Acoustic, Key: Biometric, Value: Biometric
        attn_output, _ = self.multihead_attn(x_acoustic, x_biometric, x_biometric)
        
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
    def __init__(self, projection_dim=512, standalone=False):
        super(DUALSTREAM, self).__init__()
        
        # ACOUSTIC STREAM
        self.acoustic = AcousticStream(standalone=standalone) # (B, 2048)
        self.acoustic_linear = nn.Linear(2048, projection_dim) # (B, 2048) -> (B, 512)
        
        # PROSODIC STREAM
        self.biostream = BioStreamLSTM(input_dim=404, hidden_dim=projection_dim/2) # (B, 512)
        
        # FUSTION
        self.fusion = CrossAttentionFusion(embed_dim=projection_dim) # (B, 512) -> (B, 512)
        
        # CLASSIFIER HEAD
        self.classifier = BinaryClassifier(input_size=projection_dim) # (B, 512) -> (B, 2)
            
    def forward(self, x_raw, x_bio, lengths):
        # ACOUSTIC STREAM
        a_out = self.acoustic(x_raw) # (B, 2048)
        acoustic_output = self.acoustic_linear(a_out) # (B, 2048) -> (B, 512)
        
        # BIO STREAM
        bio_output = self.biostream.forward(x_bio, lengths) # -> (B, 512)
        
        # fuse the output of the two streams using cross attention
        features = self.fusion.forward(acoustic_output, bio_output) # (B, 512)
        
        # binary classifier head on fused features
        output = self.classifier.forward(features) # (B, 2)
        return output
    