from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/splits")
IMG_DIR = RAW_DIR / "train_images"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# จำนวนต่อคลาสตามรายงานเดิม
TARGET_COUNTS = {
    0: {"train": 1264, "val": 271, "test": 270},  # No DR
    1: {"train": 259,  "val": 55,  "test": 56},   # Mild
    2: {"train": 699,  "val": 150, "test": 150},  # Moderate
    3: {"train": 135,  "val": 29,  "test": 29},   # Severe
    4: {"train": 206,  "val": 44,  "test": 45},   # Proliferative DR
}

LABELS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

df = pd.read_csv(RAW_DIR / "train.csv")

required_cols = {"id_code", "diagnosis"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"train.csv must contain columns: {required_cols}")

# เพิ่ม path รูปภาพ
df["image_path"] = df["id_code"].apply(lambda x: str(IMG_DIR / f"{x}.png"))
df["label_name"] = df["diagnosis"].map(LABELS)

missing = [p for p in df["image_path"] if not Path(p).exists()]
if missing:
    print(f"WARNING: Missing images: {len(missing)}")
    print("Example:", missing[:5])
else:
    print("All image files found.")

train_parts = []
val_parts = []
test_parts = []

for diagnosis, counts in TARGET_COUNTS.items():
    group = df[df["diagnosis"] == diagnosis].sample(
        frac=1,
        random_state=SEED + diagnosis
    ).reset_index(drop=True)

    expected_total = counts["train"] + counts["val"] + counts["test"]
    if len(group) != expected_total:
        raise ValueError(
            f"Class {diagnosis} expected {expected_total} rows, found {len(group)}"
        )

    n_train = counts["train"]
    n_val = counts["val"]

    train_parts.append(group.iloc[:n_train])
    val_parts.append(group.iloc[n_train:n_train + n_val])
    test_parts.append(group.iloc[n_train + n_val:])

train_df = pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df = pd.concat(val_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df = pd.concat(test_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

train_df.to_csv(OUT_DIR / "train.csv", index=False)
val_df.to_csv(OUT_DIR / "val.csv", index=False)
test_df.to_csv(OUT_DIR / "test.csv", index=False)

print("\nSaved:")
print(OUT_DIR / "train.csv")
print(OUT_DIR / "val.csv")
print(OUT_DIR / "test.csv")

print("\nSplit sizes:")
print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

summary = pd.DataFrame({
    "Train": train_df["diagnosis"].value_counts().sort_index(),
    "Validation": val_df["diagnosis"].value_counts().sort_index(),
    "Test": test_df["diagnosis"].value_counts().sort_index(),
})
summary["Total"] = summary.sum(axis=1)
summary["Class"] = summary.index.map(LABELS)
summary = summary[["Class", "Train", "Validation", "Test", "Total"]]

print("\nClass distribution:")
print(summary.to_string(index=False))