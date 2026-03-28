'''
DUALSTREAM: an audio deepfake detection model

Prosodic features (4) = fundamental frequency F_0, jitter, shimmer, harmonicity (HNR)
Digital features (embeddings) = wav2vec2 replaces mel-spectrogram

Fusion mechanism = gated multimodal unit (GMU) / cross-attention / cross-attention pooling

Input = .wav file

[digistream]                                            [biostream] 
Wav2vec2-XLSR → embedding                               Feature Extractor → prosodic features
Embedding → Convolutional Transformer → output          Prosodic features → LSTM → output

Prosodic output + Digital output → Fusion mechanism
Fusion mechanism → sigmoid → binary classification
Output = binary classification
'''

import torch.nn as nn
from acousticstream import AcousticStream
from prosodicstream import ProsodicStream

class DUALSTREAM(nn.Module):
    """
    Input: raw 1D .wav waveform
    Output: binary classification
    """
    def __init__(self, args):
        self.acoustic = AcousticStream(...)
        self.prosodic = ProsodicStream(...)
        self.fusion = ...
            
    def forward(self, x):
        acoustic_output = self.acoustic.forward(x)
        prosodic_output = self.prosodic.forward(x)
        
        output = self.fusion.forward(acoustic_output, prosodic_output)
        
        return output
