from transformers import AutoProcessor, AutoModelForPreTraining

# wav2vec -> aasist head 

def load_wav2vec():
    """
    Input: raw audio waveform (type 1D pytorch tensor)
    """
    processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    
    return model

def load_aasist():
    ...