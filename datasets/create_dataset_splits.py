from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_stratified_split_csv(
    root_dir,
    output_csv="dataset.csv",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    label_map=None,
):
    root_dir = Path(root_dir)
    label_map = label_map or {"Real": 0, "Fake": 1}

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f}"
        )

    rows = []
    for class_name, label in label_map.items():
        class_dir = root_dir / class_name
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist, skipping")
            continue

        wav_files = sorted(class_dir.rglob("*.wav"))
        for wav_path in wav_files:
            rows.append(
                {
                    "path": str(wav_path.resolve()),
                    "label": label,
                    "class_name": class_name,
                }
            )

    if not rows:
        raise ValueError(f"No .wav files found under {root_dir}")

    df = pd.DataFrame(rows)

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=df["label"],
    )

    # Second split: val vs test from the remaining pool
    val_fraction_of_temp = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_fraction_of_temp),
        random_state=seed,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    print("\nSaved split CSV:")
    print(output_csv.resolve())

    print("\nSplit counts:")
    print(final_df["split"].value_counts())

    print("\nSplit/class counts:")
    print(final_df.groupby(["split", "class_name"]).size())

    print("\nTotal files:", len(final_df))


if __name__ == "__main__":
    make_stratified_split_csv(
        root_dir="es/commonvoice-es",
        output_csv="XMAD_Bench_es_commonvoice.csv",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
        label_map={"real": 0, "fake": 1},
    )