from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import DRDataset


PROJECT_DIR = Path(".")
SPLIT_DIR = PROJECT_DIR / "data" / "splits"
MODEL_PATH = PROJECT_DIR / "models" / "best_efficientnet_b3.pth"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

NUM_CLASSES = 5
BATCH_SIZE = 8

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


def build_model(device):
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model.to(device)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Testing"):
        images = images.to(device)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def compute_specificity(cm):
    specificities = []

    for i in range(NUM_CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)

    return specificities


def save_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)

    ax.set_title("Confusion Matrix on Test Set")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    out_path = OUTPUT_DIR / "confusion_matrix_test.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")


def save_precision_recall_curves(y_true, y_probs):
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])

    plt.figure(figsize=(8, 6))

    ap_scores = {}

    for i, class_name in enumerate(CLASS_NAMES):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_probs[:, i])
        ap = average_precision_score(y_bin[:, i], y_probs[:, i])
        ap_scores[class_name] = ap

        plt.plot(recall, precision, label=f"{class_name} AP={ap:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / "precision_recall_curves_test.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved: {out_path}")

    return ap_scores


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_dataset = DRDataset(SPLIT_DIR / "test.csv", train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model: {MODEL_PATH}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss')}")

    y_true, y_pred, y_probs = predict(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    print("\nOverall metrics")
    print(f"Accuracy: {acc:.4f}")
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    specificities = compute_specificity(cm)

    metrics_rows = []

    for i, class_name in enumerate(CLASS_NAMES):
        metrics_rows.append({
            "class_id": i,
            "class_name": class_name,
            "precision": report[class_name]["precision"],
            "sensitivity_recall": report[class_name]["recall"],
            "specificity": specificities[i],
            "f1_score": report[class_name]["f1-score"],
            "support": report[class_name]["support"],
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUTPUT_DIR / "test_class_metrics.csv", index=False)

    print("\nClass metrics")
    print(metrics_df.to_string(index=False))

    pred_df = pd.read_csv(SPLIT_DIR / "test.csv").copy()
    pred_df["true_label"] = y_true
    pred_df["pred_label"] = y_pred
    pred_df["pred_class_name"] = [CLASS_NAMES[i] for i in y_pred]

    for i, class_name in enumerate(CLASS_NAMES):
        safe_name = class_name.lower().replace(" ", "_").replace("/", "_")
        pred_df[f"prob_{safe_name}"] = y_probs[:, i]

    pred_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    save_confusion_matrix(cm)
    ap_scores = save_precision_recall_curves(y_true, y_probs)

    ap_df = pd.DataFrame([
        {"class_name": k, "average_precision": v}
        for k, v in ap_scores.items()
    ])
    ap_df.to_csv(OUTPUT_DIR / "test_average_precision.csv", index=False)

    summary_df = pd.DataFrame([{
        "accuracy": acc,
        "quadratic_weighted_kappa": qwk,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_loss": checkpoint.get("best_val_loss"),
    }])
    summary_df.to_csv(OUTPUT_DIR / "test_summary.csv", index=False)

    print("\nSaved files:")
    print(OUTPUT_DIR / "test_summary.csv")
    print(OUTPUT_DIR / "test_class_metrics.csv")
    print(OUTPUT_DIR / "test_predictions.csv")
    print(OUTPUT_DIR / "test_average_precision.csv")
    print(OUTPUT_DIR / "confusion_matrix_test.png")
    print(OUTPUT_DIR / "precision_recall_curves_test.png")


if __name__ == "__main__":
    main()