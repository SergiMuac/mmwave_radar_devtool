"""Evaluation and interpretability utilities for classification experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:  # Optional dependency.
    from sklearn.decomposition import PCA as _SklearnPCA
except Exception:  # pragma: no cover - exercised when sklearn is absent.
    _SklearnPCA = None


@dataclass(slots=True, frozen=True)
class ClassificationReport:
    """Structured per-class and aggregate classification metrics."""

    rows: tuple[dict[str, float | int | str], ...]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax for `[N, C]` logits."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected logits [N,C], got {x.shape}")
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    return (exp / np.maximum(denom, 1e-12)).astype(np.float32)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix as counts with shape `[C, C]`."""
    true_idx = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred_idx = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true_idx.shape != pred_idx.shape:
        raise ValueError(
            "y_true and y_pred shape mismatch. "
            f"y_true={true_idx.shape} y_pred={pred_idx.shape}"
        )
    if num_classes <= 0:
        raise ValueError(f"num_classes must be > 0, got {num_classes}")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (
        (true_idx >= 0)
        & (true_idx < num_classes)
        & (pred_idx >= 0)
        & (pred_idx < num_classes)
    )
    if not np.all(valid):
        bad = int(np.size(valid) - int(np.sum(valid)))
        raise ValueError(f"Found {bad} labels outside class range [0, {num_classes - 1}]")
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm


def normalize_confusion_matrix_rows(cm: np.ndarray) -> np.ndarray:
    """Return row-normalized confusion matrix where each row sums to 1."""
    counts = np.asarray(cm, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError(f"Expected square confusion matrix, got {counts.shape}")
    row_sums = np.sum(counts, axis=1, keepdims=True)
    out = np.zeros_like(counts, dtype=np.float64)
    np.divide(counts, row_sums, out=out, where=row_sums > 0)
    return out.astype(np.float32)


def compute_classification_report(
    cm: np.ndarray,
    *,
    class_names: list[str],
) -> ClassificationReport:
    """Compute per-class precision/recall/F1 from confusion-matrix counts."""
    counts = np.asarray(cm, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError(f"Expected square confusion matrix, got {counts.shape}")
    if len(class_names) != counts.shape[0]:
        raise ValueError(
            "class_names length must match confusion matrix size. "
            f"names={len(class_names)} cm={counts.shape}"
        )

    total = float(np.sum(counts))
    tp = np.diag(counts)
    support = np.sum(counts, axis=1)
    pred_support = np.sum(counts, axis=0)

    precision = np.divide(tp, pred_support, out=np.zeros_like(tp), where=pred_support > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )

    rows: list[dict[str, float | int | str]] = []
    for idx, label in enumerate(class_names):
        rows.append(
            {
                "class": label,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )

    macro_precision = float(np.mean(precision)) if precision.size else 0.0
    macro_recall = float(np.mean(recall)) if recall.size else 0.0
    macro_f1 = float(np.mean(f1)) if f1.size else 0.0

    weight = np.divide(
        support,
        np.sum(support),
        out=np.zeros_like(support),
        where=np.sum(support) > 0,
    )
    weighted_precision = float(np.sum(weight * precision))
    weighted_recall = float(np.sum(weight * recall))
    weighted_f1 = float(np.sum(weight * f1))
    accuracy = float(np.sum(tp) / total) if total > 0 else 0.0

    return ClassificationReport(
        rows=tuple(rows),
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
    )


def save_confusion_matrix_csv(
    cm: np.ndarray,
    *,
    class_names: list[str],
    output_path: Path,
    normalized: bool,
) -> None:
    """Save confusion matrix table to CSV with class-name headers."""
    matrix = normalize_confusion_matrix_rows(cm) if normalized else np.asarray(cm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for idx, class_name in enumerate(class_names):
            row = matrix[idx]
            if normalized:
                values = [f"{float(value):.6f}" for value in row]
            else:
                values = [str(int(value)) for value in row]
            writer.writerow([class_name, *values])


def save_classification_report_csv(
    report: ClassificationReport,
    *,
    output_path: Path,
) -> None:
    """Save per-class classification report as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        for row in report.rows:
            writer.writerow(row)
        writer.writerow(
            {
                "class": "macro_avg",
                "precision": report.macro_precision,
                "recall": report.macro_recall,
                "f1": report.macro_f1,
                "support": "",
            }
        )
        writer.writerow(
            {
                "class": "weighted_avg",
                "precision": report.weighted_precision,
                "recall": report.weighted_recall,
                "f1": report.weighted_f1,
                "support": "",
            }
        )
        writer.writerow(
            {
                "class": "accuracy",
                "precision": report.accuracy,
                "recall": report.accuracy,
                "f1": report.accuracy,
                "support": "",
            }
        )


def plot_training_curves(
    *,
    epochs: np.ndarray,
    train_loss: np.ndarray,
    val_loss: np.ndarray,
    train_acc: np.ndarray,
    val_acc: np.ndarray,
    output_path: Path,
) -> None:
    """Plot train/validation loss and accuracy across epochs."""
    x = np.asarray(epochs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(x, np.asarray(train_loss), label="train_loss", linewidth=1.6)
    axes[0].plot(x, np.asarray(val_loss), label="val_loss", linewidth=1.6)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(x, np.asarray(train_acc), label="train_acc", linewidth=1.6)
    axes[1].plot(x, np.asarray(val_acc), label="val_acc", linewidth=1.6)
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    class_names: list[str],
    output_path: Path,
    normalized: bool,
) -> None:
    """Plot confusion matrix heatmap with class labels on both axes."""
    matrix = normalize_confusion_matrix_rows(cm) if normalized else np.asarray(cm, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", origin="upper")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Normalized Confusion Matrix" if normalized else "Confusion Matrix")

    if matrix.size > 0:
        threshold = float(np.nanmax(matrix)) * 0.5
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = float(matrix[row, col])
                text = f"{value:.2f}" if normalized else str(int(value))
                ax.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pca_project_2d(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project feature matrix `[N,D]` into 2D using PCA."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected [N,D] feature matrix, got {x.shape}")
    if x.shape[0] == 0:
        raise ValueError("Cannot run PCA on zero samples.")
    if x.shape[1] == 0:
        raise ValueError("Cannot run PCA on zero-dimensional features.")

    if _SklearnPCA is not None and x.shape[0] >= 2:
        pca = _SklearnPCA(n_components=2)
        coords = pca.fit_transform(x).astype(np.float32)
        explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float32)
        if explained.size < 2:
            explained = np.pad(explained, (0, 2 - explained.size), constant_values=0.0)
        return coords, explained[:2]

    centered = x - np.mean(x, axis=0, keepdims=True)
    if centered.shape[1] == 1:
        coords = np.concatenate([centered, np.zeros_like(centered)], axis=1)
        return coords.astype(np.float32), np.array([1.0, 0.0], dtype=np.float32)

    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    coords = centered @ components.T
    var = singular_values**2
    var_total = float(np.sum(var))
    if var_total > 0.0:
        explained = (var[:2] / var_total).astype(np.float32)
    else:
        explained = np.zeros((2,), dtype=np.float32)
    if explained.size < 2:
        explained = np.pad(explained, (0, 2 - explained.size), constant_values=0.0)
    return coords.astype(np.float32), explained[:2]


def plot_pca_scatter(
    coords: np.ndarray,
    *,
    y_true: np.ndarray,
    class_names: list[str],
    output_path: Path,
    explained_variance: np.ndarray,
    title: str,
    y_pred: np.ndarray | None = None,
) -> None:
    """Plot PCA scatter colored by true class, optional X for errors."""
    xy = np.asarray(coords, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected PCA coords [N,2], got {xy.shape}")
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if labels.size != xy.shape[0]:
        raise ValueError(
            "y_true length must equal number of PCA points. "
            f"labels={labels.size} points={xy.shape[0]}"
        )
    pred = None if y_pred is None else np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if pred is not None and pred.size != labels.size:
        raise ValueError(
            "y_pred length must equal y_true length. "
            f"pred={pred.size} labels={labels.size}"
        )

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    palette = plt.get_cmap("tab10")
    for class_idx, class_name in enumerate(class_names):
        mask = labels == class_idx
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=26,
            alpha=0.75,
            color=palette(class_idx % 10),
            label=class_name,
            linewidths=0.0,
        )

    if pred is not None:
        wrong = pred != labels
        if np.any(wrong):
            ax.scatter(
                xy[wrong, 0],
                xy[wrong, 1],
                s=46,
                marker="x",
                color="black",
                linewidths=0.9,
                label="misclassified",
            )

    pc1_var = float(explained_variance[0]) if explained_variance.size > 0 else 0.0
    pc2_var = float(explained_variance[1]) if explained_variance.size > 1 else 0.0
    ax.set_xlabel(f"PC1 ({pc1_var * 100.0:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pc2_var * 100.0:.1f}% var)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_histogram(
    confidence: np.ndarray,
    *,
    correct_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Plot max-softmax confidence histogram for correct vs incorrect predictions."""
    conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
    correct = np.asarray(correct_mask, dtype=bool).reshape(-1)
    if conf.size != correct.size:
        raise ValueError(
            "confidence and correct_mask lengths must match. "
            f"confidence={conf.size} correct={correct.size}"
        )

    bins = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(
        conf[correct],
        bins=bins,
        alpha=0.75,
        label="correct",
        color="#22c55e",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.hist(
        conf[~correct],
        bins=bins,
        alpha=0.70,
        label="incorrect",
        color="#ef4444",
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_title("Prediction Confidence Histogram")
    ax.set_xlabel("Max softmax probability")
    ax.set_ylabel("Sample count")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_multiclass_predictions(
    logits: np.ndarray,
    y_true: np.ndarray,
    *,
    class_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ClassificationReport]:
    """Derive confusion matrix, predictions, confidence, and class report."""
    probs = softmax_numpy(logits)
    pred = np.argmax(probs, axis=1).astype(np.int64)
    true_idx = np.asarray(y_true, dtype=np.int64).reshape(-1)
    conf = np.max(probs, axis=1).astype(np.float32)
    cm = compute_confusion_matrix(true_idx, pred, num_classes=len(class_names))
    report = compute_classification_report(cm, class_names=class_names)
    correct = (pred == true_idx)
    return cm, pred, conf, correct, report


__all__ = [
    "ClassificationReport",
    "compute_classification_report",
    "compute_confusion_matrix",
    "normalize_confusion_matrix_rows",
    "pca_project_2d",
    "plot_confidence_histogram",
    "plot_confusion_matrix",
    "plot_pca_scatter",
    "plot_training_curves",
    "save_classification_report_csv",
    "save_confusion_matrix_csv",
    "softmax_numpy",
    "summarize_multiclass_predictions",
]
