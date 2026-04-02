from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
import soundfile as sf
from transformers import Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer


# ---------------------------------------------------
# Global model/tokenizer
# ---------------------------------------------------

CHECKPOINT = "facebook/wav2vec2-lv-60-espeak-cv-ft"

model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT)
tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(CHECKPOINT)

BLANK_ID = model.config.pad_token_id


# ---------------------------------------------------
# 1) Get phonemes from audio_path only
# ---------------------------------------------------

def get_phonemes(audio_path):
    """
    Input: path to audio file
    Return:
      - pred_ids: frame-level predicted phoneme IDs
      - phonemes: frame-level predicted phoneme tokens
      - wav: waveform tensor [1, N]
      - sr: sample rate
    """
    wav, sr = sf.read(str(audio_path))

    # stereo -> mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr}")

    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(wav).logits  # [1, T, vocab]

    pred_ids = torch.argmax(logits, dim=-1)[0]  # [T]
    phonemes = tokenizer.convert_ids_to_tokens(pred_ids.tolist())

    return pred_ids, phonemes, wav, sr


# ---------------------------------------------------
# 2) Segment phonemes from pred_ids, wav, sr only
# ---------------------------------------------------

def segment_phonemes(pred_ids, wav, sr):
    """
    Input: pred_ids (T,), wav (1, N), sr (int)
    Returns: 
        a list of dicts with: phoneme, start_frame, end_frame, start_sec, end_sec, duration
    """
    num_frames = pred_ids.shape[0]
    audio_duration = wav.shape[1] / sr
    seconds_per_frame = audio_duration / num_frames

    segments = []
    current_id = None
    start_frame = None

    for t, token_id in enumerate(pred_ids.tolist()):
        if token_id == BLANK_ID:
            if current_id is not None:
                start_sec = start_frame * seconds_per_frame
                end_sec = t * seconds_per_frame
                segments.append({
                    "phoneme": tokenizer.convert_ids_to_tokens(current_id),
                    "start_frame": start_frame,
                    "end_frame": t - 1,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration": end_sec - start_sec,
                })
                current_id = None
                start_frame = None
            continue

        if current_id is None:
            current_id = token_id
            start_frame = t
        elif token_id != current_id:
            start_sec = start_frame * seconds_per_frame
            end_sec = t * seconds_per_frame
            segments.append({
                "phoneme": tokenizer.convert_ids_to_tokens(current_id),
                "start_frame": start_frame,
                "end_frame": t - 1,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration": end_sec - start_sec,
            })
            current_id = token_id
            start_frame = t

    # flush final segment
    if current_id is not None:
        start_sec = start_frame * seconds_per_frame
        end_sec = audio_duration
        segments.append({
            "phoneme": tokenizer.convert_ids_to_tokens(current_id),
            "start_frame": start_frame,
            "end_frame": num_frames - 1,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": end_sec - start_sec,
        })

    return segments


# ---------------------------------------------------
# 3) Transition features
# ---------------------------------------------------

def get_transition_features(segments):
    """
    Return transition-level features from segmented phonemes.

    Output:
      {
        "sequence": [('AH', 'N'), ('N', 'T'), ...],
        "counts": Counter(...),
        "normalized": {('AH','N'): prob, ...}
      }
    """
    phoneme_seq = [seg["phoneme"] for seg in segments]
    transitions = [
        (phoneme_seq[i], phoneme_seq[i + 1])
        for i in range(len(phoneme_seq) - 1)
    ]

    counts = Counter(transitions)
    total = sum(counts.values())

    normalized = {
        tr: count / total for tr, count in counts.items()
    } if total > 0 else {}

    return {
        "sequence": transitions,
        "counts": counts,
        "normalized": normalized,
    }


# ---------------------------------------------------
# 4) Duration features
# ---------------------------------------------------

def get_duration_features(segments):
    """
    Return duration features per phoneme.

    Output:
      {
        "per_segment": [('AH', 0.08), ('N', 0.05), ...],
        "by_phoneme": {
            'AH': {'count': ..., 'total': ..., 'mean': ..., 'std': ..., ...},
            ...
        }
      }
    """
    per_segment = [(seg["phoneme"], seg["duration"]) for seg in segments]

    durations_by_phoneme = defaultdict(list)
    for seg in segments:
        durations_by_phoneme[seg["phoneme"]].append(seg["duration"])

    by_phoneme = {}
    for ph, ds in durations_by_phoneme.items():
        ds = np.array(ds, dtype=np.float32)
        by_phoneme[ph] = {
            "count": int(len(ds)),
            "total": float(ds.sum()),
            "mean": float(ds.mean()),
            "std": float(ds.std()) if len(ds) > 1 else 0.0,
            "min": float(ds.min()),
            "max": float(ds.max()),
        }

    return {
        "per_segment": per_segment,
        "by_phoneme": by_phoneme,
    }


# ---------------------------------------------------
# 5) Vectorization helpers
# ---------------------------------------------------

def build_phoneme_vocab(segments_list):
    phonemes = sorted({
        seg["phoneme"]
        for segments in segments_list
        for seg in segments
    })
    return {ph: i for i, ph in enumerate(phonemes)}


def build_transition_vocab(segments_list):
    transitions = set()

    for segments in segments_list:
        phoneme_seq = [seg["phoneme"] for seg in segments]
        for i in range(len(phoneme_seq) - 1):
            transitions.add((phoneme_seq[i], phoneme_seq[i + 1]))

    transitions = sorted(transitions)
    return {tr: i for i, tr in enumerate(transitions)}


# ---------------------------------------------------
# 6) Vectorize phoneme, transition, duration features
# ---------------------------------------------------

def vectorize_features(segments, phoneme_vocab, transition_vocab):
    """
    Vectorize:
      - phoneme counts
      - transition counts (normalized)
      - duration features: count, total, mean per phoneme

    Returns:
      dict with separate vectors + one concatenated feature vector
    """
    n_ph = len(phoneme_vocab)
    n_tr = len(transition_vocab)

    # ---- phoneme count vector ----
    phoneme_vec = np.zeros(n_ph, dtype=np.float32)
    for seg in segments:
        phoneme_vec[phoneme_vocab[seg["phoneme"]]] += 1.0

    # ---- transition vector ----
    transition_vec = np.zeros(n_tr, dtype=np.float32)
    phoneme_seq = [seg["phoneme"] for seg in segments]
    for i in range(len(phoneme_seq) - 1):
        tr = (phoneme_seq[i], phoneme_seq[i + 1])
        if tr in transition_vocab:
            transition_vec[transition_vocab[tr]] += 1.0

    if transition_vec.sum() > 0:
        transition_vec /= transition_vec.sum()

    # ---- duration vectors ----
    duration_count_vec = np.zeros(n_ph, dtype=np.float32)
    duration_total_vec = np.zeros(n_ph, dtype=np.float32)
    duration_mean_vec = np.zeros(n_ph, dtype=np.float32)

    durations_by_phoneme = defaultdict(list)
    for seg in segments:
        durations_by_phoneme[seg["phoneme"]].append(seg["duration"])

    for ph, ds in durations_by_phoneme.items():
        idx = phoneme_vocab[ph]
        ds = np.array(ds, dtype=np.float32)
        duration_count_vec[idx] = len(ds)
        duration_total_vec[idx] = ds.sum()
        duration_mean_vec[idx] = ds.mean()

    duration_vec = np.concatenate(
        [duration_count_vec, duration_total_vec, duration_mean_vec],
        axis=0
    )

    # ---- all in one ----
    full_vector = np.concatenate(
        [phoneme_vec, transition_vec, duration_vec],
        axis=0
    )

    return {
        "phoneme_vector": phoneme_vec,
        "transition_vector": transition_vec,
        "duration_vector": duration_vec,
        "full_vector": full_vector,
    }


# ---------------------------------------------------
# Example usage
# ---------------------------------------------------

if __name__ == "__main__":
    audio_path = Path("model/common_voice_en_18850744.wav")

    pred_ids, phonemes, wav, sr = get_phonemes(audio_path)
    segments = segment_phonemes(pred_ids, wav, sr)

    transition_features = get_transition_features(segments)
    duration_features = get_duration_features(segments)

    # For a real dataset, build vocabs from the training set.
    # Here we build them from this single utterance for demonstration.
    phoneme_vocab = build_phoneme_vocab([segments])
    transition_vocab = build_transition_vocab([segments])

    vectors = vectorize_features(segments, phoneme_vocab, transition_vocab)

    print("Frame-level phonemes:")
    print(phonemes)

    print("\nSegments:")
    for seg in segments:
        print(
            f"{seg['phoneme']:>6} "
            f"{seg['start_sec']:.3f} -> {seg['end_sec']:.3f} "
            f"dur={seg['duration']:.3f}"
        )

    print("\nTransition features:")
    print(transition_features["counts"])

    print("\nDuration features:")
    print(duration_features["by_phoneme"])

    print("\nVector shapes:")
    print("phoneme_vector:", vectors["phoneme_vector"].shape)
    print("transition_vector:", vectors["transition_vector"].shape)
    print("duration_vector:", vectors["duration_vector"].shape)
    print("full_vector:", vectors["full_vector"].shape)