from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer

from shared_audio_processor import Segment, SharedAudioProcessor


class ArticulatoryFeatureExtractor:
    """
    Segment-based phoneme/articulatory feature extractor.

    Produces:
      - segments
      - segment_feature_sequence: [num_segments, articulatory_dim]
      - utterance_feature_vector: pooled [2 * articulatory_dim]
    """

    def __init__(
        self,
        checkpoint: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        include_log_duration: bool = True,
        include_position: bool = True,
    ) -> None:
        self.shared = SharedAudioProcessor(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.include_log_duration = include_log_duration
        self.include_position = include_position

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = Wav2Vec2ForCTC.from_pretrained(checkpoint).to(self.device)
        self.model.eval()

        self.tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(checkpoint)

        self.blank_id = self.model.config.pad_token_id
        self.vocab_size = int(self.model.config.vocab_size)

    @torch.no_grad()
    def _infer_logits(self, wav_torch: torch.Tensor) -> torch.Tensor:
        return self.model(wav_torch).logits[0]  # [T, vocab]

    def _decode_frames(self, logits: torch.Tensor) -> Dict:
        posteriors = F.softmax(logits, dim=-1)
        pred_ids = torch.argmax(posteriors, dim=-1)
        phonemes = self.tokenizer.convert_ids_to_tokens(pred_ids.tolist())

        return {
            "pred_ids": pred_ids,
            "phonemes": phonemes,
            "posteriors": posteriors,
        }

    def _segment_phonemes(
        self,
        pred_ids: torch.Tensor,
        posteriors: torch.Tensor,
        num_samples: int,
    ) -> List[Segment]:
        num_frames = int(pred_ids.shape[0])
        audio_duration = num_samples / self.sample_rate
        seconds_per_frame = audio_duration / max(num_frames, 1)

        segments: List[Segment] = []
        current_id: Optional[int] = None
        start_frame: Optional[int] = None

        def flush_segment(end_frame_exclusive: int) -> None:
            nonlocal current_id, start_frame

            if current_id is None or start_frame is None:
                return

            seg_post = posteriors[start_frame:end_frame_exclusive, current_id]
            start_sec = start_frame * seconds_per_frame
            end_sec = end_frame_exclusive * seconds_per_frame

            segments.append(
                Segment(
                    phoneme=self.tokenizer.convert_ids_to_tokens(int(current_id)),
                    phoneme_id=int(current_id),
                    start_frame=int(start_frame),
                    end_frame=int(end_frame_exclusive - 1),
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    duration=float(end_sec - start_sec),
                    mean_confidence=float(seg_post.mean().item()),
                )
            )

            current_id = None
            start_frame = None

        for t, token_id in enumerate(pred_ids.tolist()):
            if token_id == self.blank_id:
                flush_segment(t)
                continue

            if current_id is None:
                current_id = token_id
                start_frame = t
            elif token_id != current_id:
                flush_segment(t)
                current_id = token_id
                start_frame = t

        if current_id is not None and start_frame is not None:
            flush_segment(num_frames)

        return segments

    def _build_segment_sequence(self, segments: List[Segment]) -> np.ndarray:
        if not segments:
            extra_dims = 2  # duration + confidence
            if self.include_log_duration:
                extra_dims += 1
            if self.include_position:
                extra_dims += 2
            return np.zeros((0, self.vocab_size + extra_dims), dtype=np.float32)

        total_end = max(seg.end_sec for seg in segments)
        total_end = max(total_end, 1e-8)

        rows = []
        for seg in segments:
            one_hot = np.zeros(self.vocab_size, dtype=np.float32)
            one_hot[seg.phoneme_id] = 1.0

            extras = [seg.duration]
            if self.include_log_duration:
                extras.append(np.log1p(seg.duration))
            extras.append(seg.mean_confidence)

            if self.include_position:
                extras.extend([
                    seg.start_sec / total_end,
                    seg.end_sec / total_end,
                ])

            row = np.concatenate([one_hot, np.asarray(extras, dtype=np.float32)], axis=0)
            rows.append(row)

        return np.stack(rows, axis=0).astype(np.float32)

    def extract(self, audio_path: Union[str, Path]) -> Dict:
        wav = self.shared.load_audio(audio_path)
        wav_torch = torch.tensor(wav, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits = self._infer_logits(wav_torch)
        decoded = self._decode_frames(logits)

        segments = self._segment_phonemes(
            pred_ids=decoded["pred_ids"],
            posteriors=decoded["posteriors"],
            num_samples=len(wav),
        )

        segment_feature_sequence = self._build_segment_sequence(segments)
        utterance_feature_vector = self.shared.pooled_stats(segment_feature_sequence)

        return {
            "frame_phonemes": decoded["phonemes"],
            "segments": segments,
            "segment_feature_sequence": segment_feature_sequence,
            "utterance_feature_vector": utterance_feature_vector,
        }

    def extract_feature_vector(self, audio_path: Union[str, Path]) -> np.ndarray:
        return self.extract(audio_path)["utterance_feature_vector"]

    def extract_segment_sequence(self, audio_path: Union[str, Path]) -> np.ndarray:
        return self.extract(audio_path)["segment_feature_sequence"]