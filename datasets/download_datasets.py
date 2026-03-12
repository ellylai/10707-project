# Downloads datasets from HuggingFace
from datasets import load_dataset, Audio
'''
In domain:
    Train + Val: SpeechFake (Train+Val set)
    Test: SpeechFake (Test set)
Cross domain:
    Train: SpeechFake (Train+Val set)
    Test: XMAD-Bench (Cross-domain split) [sota]
'''

def get_speechfake(splits=[]):
    # SpeechFake has Train, Validation, and Test
    # splits is in train, val or test
    """
    DataType: .wav
    """
    speechfake = load_dataset("DeepFense/SpeechFake")
    if not splits:
        return speechfake
    else:
        return [speechfake[s] for s in splits]

def get_xmad(splits=[], data_dir="data/xmad-bench"):
    """
    XMAD-Bench cross-domain benchmark
    https://github.com/ristea/xmad-bench/
    """

    xmad = load_dataset(
        "csv",
        data_files=f"{data_dir}/metadata.csv"
    )

    xmad = xmad.cast_column("audio", Audio())

    if not splits:
        return xmad
    else:
        return [xmad[split] for split in splits]