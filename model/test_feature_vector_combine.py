import numpy as np
from articulatory_features import ArticulatoryFeatureExtractor
from prosodic_features import ProsodicFeatureExtractor

audio_path = "model/fake_1.wav"

art = ArticulatoryFeatureExtractor()
pro = ProsodicFeatureExtractor()

art_result = art.extract(audio_path)
pro_result = pro.extract_from_segments(audio_path, art_result["segments"])

combined_seq = pro.combine_with_articulatory(
    art_result["segment_feature_sequence"],
    pro_result["segment_feature_sequence"],
)

print("Articulatory:", art_result["segment_feature_sequence"].shape)
print("Prosodic:", pro_result["segment_feature_sequence"].shape)
print("Combined:", combined_seq.shape)

np.set_printoptions(suppress=True, precision=4, linewidth=200)

segments = art_result["segments"]
art_seq = art_result["segment_feature_sequence"]
pro_seq = pro_result["segment_feature_sequence"]
combined_seq = combined_seq 

# to observe the alignment of features with segments, print the first few segments and their corresponding feature vectors
for i, (seg, a_vec, p_vec, c_vec) in enumerate(zip(segments, art_seq, pro_seq, combined_seq)):
    print(f"\n--- Segment {i} ---")
    print(f"Phoneme: {seg.phoneme}")
    print(f"Time: {seg.start_sec:.3f}s → {seg.end_sec:.3f}s (dur={seg.duration:.3f})")

    print("\n[Articulatory]")
    print(a_vec)

    print("\n[Prosodic]")
    print(p_vec)

    print("\n[Combined]")
    print(c_vec)
    break