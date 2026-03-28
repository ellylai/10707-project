import parselmouth
import numpy as np
import torch
import torch.nn as nn

def extract_prosodic_features(audio_path):
    sound = parselmouth.Sound(audio_path)
    pitch = sound.to_pitch()
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    
    # 1. Fundamental Frequency (F0)
    f0 = np.nan_to_num(pitch.selected_array['frequency']).mean()
    
    # 2. Jitter (local)
    jitter = parselmouth.praat.call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    
    # 3. Shimmer (local)
    shimmer = parselmouth.praat.call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # 4. Harmonicity (HNR)
    harmonicity = sound.to_harmonicity_cc()
    hnr = np.nan_to_num(harmonicity.values).mean()
    
    return np.array([f0, jitter, shimmer, hnr])

class BiostreamProcessor(nn.Module):
    def __init__(self, input_size=4, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 128)
        
    def forward(self, x):
        # x shape: (batch, seq_len, 4)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state or pooling
        out = self.fc(lstm_out[:, -1, :]) 
        return out