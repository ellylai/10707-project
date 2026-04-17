from __future__ import annotations

from typing import Dict, List
import numpy as np

from shared_audio_processor import Segment, SharedAudioProcessor


class ProsodicFeatureExtractor:
    """
    Extract prosodic features aligned to externally provided segments.

    Segment features:
      - voiced_f0_mean
      - voiced_f0_std
      - f0_min
      - f0_max
      - voiced_fraction
      - intensity_mean
      - intensity_std
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        pitch_floor: float = 75.0,
        pitch_ceiling: float = 500.0,
        time_step: float = 0.01,
    ) -> None:
        self.shared = SharedAudioProcessor(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.time_step = time_step

    def _extract_frame_tracks(self, audio_input: AudioInput) -> Dict[str, np.ndarray]:
        sound = self.shared.load_sound_any(audio_input)

        pitch = sound.to_pitch(
            time_step=self.time_step,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
        )
        intensity = sound.to_intensity(
            time_step=self.time_step,
            minimum_pitch=self.pitch_floor,
        )

        f0 = pitch.selected_array["frequency"].astype(np.float32)
        pitch_times = np.asarray(
            [pitch.get_time_from_frame_number(i + 1) for i in range(len(f0))],
            dtype=np.float32,
        )

        intensity_vals = intensity.values.flatten().astype(np.float32)
        intensity_times = np.asarray(
            [intensity.get_time_from_frame_number(i + 1) for i in range(len(intensity_vals))],
            dtype=np.float32,
        )

        return {
            "f0": f0,
            "pitch_times": pitch_times,
            "intensity": intensity_vals,
            "intensity_times": intensity_times,
        }

    @staticmethod
    def _slice_track(values: np.ndarray, times: np.ndarray, start: float, end: float) -> np.ndarray:
        mask = (times >= start) & (times < end)
        return values[mask]

    def _segment_prosody(self, tracks: Dict[str, np.ndarray], segments: List[Segment]) -> np.ndarray:
        rows = []

        for seg in segments:
            f0_seg = self._slice_track(tracks["f0"], tracks["pitch_times"], seg.start_sec, seg.end_sec)
            intensity_seg = self._slice_track(
                tracks["intensity"], tracks["intensity_times"], seg.start_sec, seg.end_sec
            )

            voiced_f0 = f0_seg[f0_seg > 0]
            voiced_fraction = float((f0_seg > 0).mean()) if f0_seg.size > 0 else 0.0
            intensity_valid = self.shared.finite_only(intensity_seg)

            row = np.array(
                [
                    self.shared.safe_mean(voiced_f0),
                    self.shared.safe_std(voiced_f0),
                    float(np.min(voiced_f0)) if voiced_f0.size > 0 else 0.0,
                    float(np.max(voiced_f0)) if voiced_f0.size > 0 else 0.0,
                    voiced_fraction,
                    self.shared.safe_mean(intensity_valid),
                    self.shared.safe_std(intensity_valid),
                ],
                dtype=np.float32,
            )
            rows.append(row)

        if not rows:
            return np.zeros((0, 7), dtype=np.float32)

        return np.stack(rows, axis=0).astype(np.float32)

    def extract_from_segments(
        self,
        audio_input: AudioInput,
        segments: List[Segment],
    ) -> Dict:
        tracks = self._extract_frame_tracks(audio_input)
        segment_feature_sequence = self._segment_prosody(tracks, segments)
        utterance_feature_vector = self.shared.pooled_stats(segment_feature_sequence)

        return {
            "segment_feature_sequence": segment_feature_sequence,
            "utterance_feature_vector": utterance_feature_vector,
        }

    def combine_with_articulatory(
        self,
        articulatory_sequence: np.ndarray,
        prosodic_sequence: np.ndarray,
    ) -> np.ndarray:
        if articulatory_sequence.shape[0] != prosodic_sequence.shape[0]:
            raise ValueError(
                "Segment count mismatch between articulatory and prosodic sequences: "
                f"{articulatory_sequence.shape[0]} vs {prosodic_sequence.shape[0]}"
            )

        if articulatory_sequence.shape[0] == 0:
            return np.zeros(
                (0, articulatory_sequence.shape[1] + prosodic_sequence.shape[1]),
                dtype=np.float32,
            )

        return np.concatenate([articulatory_sequence, prosodic_sequence], axis=1).astype(np.float32)