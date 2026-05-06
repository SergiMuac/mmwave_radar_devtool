"""Run inference from trained radar ML checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..range_profile_dataset import (
    RadarTensorConfig,
    compute_range_axis_m,
    parse_tensor_cfg,
    select_useful_range_side,
)
from .data import (
    BACKGROUND_SUBTRACTION_NONE,
    FEATURE_MODE_COMPLEX_COHERENT,
    FEATURE_MODE_ZERO_DOPPLER_DB,
    NORMALIZATION_NONE,
    NORMALIZATION_PER_RECORDING,
    NORMALIZATION_TRAINSET,
    BackgroundReference,
    extract_frame_feature_tensor_by_rx_db,
    feature_tensor_to_samples,
    load_background_reference,
    load_baseline_zero_doppler_mean_by_rx_db,
    per_recording_standardize_features,
)
from .models import RadarMLP


@dataclass(slots=True, frozen=True)
class PredictionSummary:
    """Aggregated prediction over one or more radar frames."""

    task: str
    primary: str
    frame_count: int
    sample_count: int | None = None
    confidence: float | None = None
    value: float | None = None
    spread: float | None = None
    detail: str = ""
    probabilities: dict[str, float] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        payload: dict[str, object] = {
            "task": self.task,
            "primary": self.primary,
            "frame_count": int(self.frame_count),
            "detail": self.detail,
        }
        if self.confidence is not None:
            payload["confidence"] = float(self.confidence)
        if self.sample_count is not None:
            payload["sample_count"] = int(self.sample_count)
        if self.value is not None:
            payload["value"] = float(self.value)
        if self.spread is not None:
            payload["spread"] = float(self.spread)
        if self.probabilities is not None:
            payload["probabilities"] = {
                str(label): float(prob) for label, prob in self.probabilities.items()
            }
        return payload


@dataclass(slots=True)
class RadarCheckpointPredictor:
    """Loaded model plus its feature recipe and normalization state."""

    checkpoint_path: Path
    task: str
    model: RadarMLP
    device: torch.device
    cfg: RadarTensorConfig
    feature: dict[str, Any]
    mean: np.ndarray
    std: np.ndarray
    class_names: list[str] | None
    baseline_open_mean_db: np.ndarray | None
    baseline_blocked_mean_db: np.ndarray | None
    background_reference: BackgroundReference | None
    feature_mode: str
    normalization_mode: str
    background_subtraction: str
    range_mask: np.ndarray | None
    live_feature_history: list[np.ndarray] = field(default_factory=list)

    def predict_capture(
        self,
        capture_path: Path,
        *,
        max_frames: int | None = None,
    ) -> PredictionSummary:
        """Predict from a recorded `.bin` capture."""
        frames = extract_frame_feature_tensor_by_rx_db(
            capture_path,
            self.cfg,
            range_side=str(self.feature.get("range_side", "positive")),
            window_kind=str(self.feature.get("window_kind", "hann")),
            eps=float(self.feature.get("eps", 1e-9)),
            max_frames=max_frames,
            baseline_open_mean_db=self.baseline_open_mean_db,
            baseline_blocked_mean_db=self.baseline_blocked_mean_db,
            range_min_m=_optional_float(self.feature.get("range_min_m")),
            range_max_m=_optional_float(self.feature.get("range_max_m")),
            feature_mode=self.feature_mode,
            target_range_m=float(self.feature.get("target_range_m", 0.40)),
            range_gate_bins=int(self.feature.get("range_gate_bins", 2)),
            background_subtraction=self.background_subtraction,
            background_reference=self.background_reference,
        )
        return self.predict_feature_tensor(frames)

    def predict_frame_db(
        self,
        frame_db: np.ndarray,
        *,
        live_window_frames: int = 16,
    ) -> PredictionSummary:
        """Predict from one live raw zero-Doppler dB frame shaped `[RX, B]`."""
        if live_window_frames <= 0:
            raise ValueError("live_window_frames must be > 0")
        if self.feature_mode != FEATURE_MODE_ZERO_DOPPLER_DB:
            raise ValueError(
                "Live prediction currently supports feature_mode='zero_doppler_db' only. "
                f"Loaded checkpoint uses {self.feature_mode!r}."
            )
        frame = np.asarray(frame_db, dtype=np.float32)
        if frame.ndim != 2:
            raise ValueError(f"Expected live frame with shape [RX,B], got {frame.shape}")
        features = self._transform_raw_frame_db(frame[None, ...])
        self.live_feature_history.append(features[0].astype(np.float32, copy=False))
        if len(self.live_feature_history) > live_window_frames:
            del self.live_feature_history[: len(self.live_feature_history) - live_window_frames]
        window = np.stack(self.live_feature_history, axis=0).astype(np.float32, copy=False)
        return self.predict_feature_tensor(window)

    def predict_feature_tensor(self, frames_rx_bins: np.ndarray) -> PredictionSummary:
        """Predict from already transformed feature frames."""
        features = np.asarray(frames_rx_bins, dtype=np.float32)
        if features.ndim not in {3, 4}:
            raise ValueError(
                "Expected feature tensor [F,RX,B] or [F,RX,B,2], "
                f"got {features.shape}"
            )
        if features.shape[0] == 0:
            raise RuntimeError("No frames were available for prediction.")

        sample_mode = str(self.feature.get("sample_mode", "frames"))
        frame_count = int(features.shape[0])
        x = feature_tensor_to_samples(features, sample_mode=sample_mode)
        if x.shape[1] != self.mean.size:
            raise ValueError(
                "Feature dimension does not match checkpoint normalization. "
                f"got={x.shape[1]}, expected={self.mean.size}"
            )
        if self.normalization_mode == NORMALIZATION_TRAINSET:
            x = ((x - self.mean[None, :]) / self.std[None, :]).astype(np.float32)
        elif self.normalization_mode == NORMALIZATION_PER_RECORDING:
            x = per_recording_standardize_features(x)
        elif self.normalization_mode == NORMALIZATION_NONE:
            x = np.asarray(x, dtype=np.float32)
        else:
            raise ValueError(
                "Unsupported normalization mode in checkpoint: "
                f"{self.normalization_mode!r}"
            )
        tensor = torch.from_numpy(x).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)

        if self.task == "classification":
            return self._summarize_classification(
                output,
                frame_count=frame_count,
                sample_count=x.shape[0],
            )
        return self._summarize_regression(
            output,
            frame_count=frame_count,
            sample_count=x.shape[0],
        )

    def _transform_raw_frame_db(self, frames_db: np.ndarray) -> np.ndarray:
        """Apply range gate and baseline deltas to raw live dB frames."""
        frames = np.asarray(frames_db, dtype=np.float32)
        if self.range_mask is not None:
            frames = frames[:, :, self.range_mask]

        target_shape = frames.shape[1:]
        deltas: list[np.ndarray] = []
        if self.baseline_open_mean_db is not None:
            open_mean = np.asarray(self.baseline_open_mean_db, dtype=np.float32)
            if open_mean.shape != target_shape:
                raise ValueError(
                    "Open baseline shape does not match live feature shape. "
                    f"baseline={open_mean.shape} expected={target_shape}"
                )
            deltas.append((frames - open_mean[None, ...]).astype(np.float32, copy=False))
        if self.baseline_blocked_mean_db is not None:
            blocked_mean = np.asarray(self.baseline_blocked_mean_db, dtype=np.float32)
            if blocked_mean.shape != target_shape:
                raise ValueError(
                    "Blocked baseline shape does not match live feature shape. "
                    f"baseline={blocked_mean.shape} expected={target_shape}"
                )
            deltas.append((frames - blocked_mean[None, ...]).astype(np.float32, copy=False))

        if not deltas:
            return frames
        if len(deltas) == 1:
            return deltas[0]
        return np.concatenate(deltas, axis=2).astype(np.float32, copy=False)

    def _summarize_classification(
        self,
        output: torch.Tensor,
        *,
        frame_count: int,
        sample_count: int,
    ) -> PredictionSummary:
        """Aggregate per-frame classifier logits."""
        if self.class_names is None:
            raise RuntimeError("Classification checkpoint is missing class names.")
        probs = torch.softmax(output, dim=1).detach().cpu().numpy()
        mean_probs = np.mean(probs, axis=0)
        best_idx = int(np.argmax(mean_probs))
        per_frame_idx = np.argmax(probs, axis=1)
        counts = np.bincount(per_frame_idx, minlength=len(self.class_names))
        vote_idx = int(np.argmax(counts))

        probabilities = {
            label: float(mean_probs[idx]) for idx, label in enumerate(self.class_names)
        }
        detail = (
            f"vote={self.class_names[vote_idx]} "
            f"{int(counts[vote_idx])}/{int(sample_count)} samples"
        )
        return PredictionSummary(
            task="classification",
            primary=self.class_names[best_idx],
            frame_count=int(frame_count),
            sample_count=int(sample_count),
            confidence=float(mean_probs[best_idx]),
            detail=detail,
            probabilities=probabilities,
        )

    def _summarize_regression(
        self,
        output: torch.Tensor,
        *,
        frame_count: int,
        sample_count: int,
    ) -> PredictionSummary:
        """Aggregate per-frame regression outputs."""
        values = output.squeeze(1).detach().cpu().numpy().astype(np.float32)
        mean_value = float(np.mean(values))
        spread = float(np.std(values))
        return PredictionSummary(
            task="regression",
            primary=f"{mean_value:.4g}",
            frame_count=int(frame_count),
            sample_count=int(sample_count),
            value=mean_value,
            spread=spread,
            detail=f"frame_std={spread:.4g}",
        )


def load_radar_predictor(
    checkpoint_path: Path,
    *,
    cfg_path: Path | None = None,
    baseline_open_capture: Path | None = None,
    baseline_blocked_capture: Path | None = None,
    background_capture: Path | None = None,
    device_name: str = "auto",
) -> RadarCheckpointPredictor:
    """Load a checkpoint and reconstruct its feature pipeline."""
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary, got {type(checkpoint).__name__}")

    feature = dict(checkpoint.get("feature") or {})
    feature_mode = str(feature.get("feature_mode", FEATURE_MODE_ZERO_DOPPLER_DB))
    normalization_mode = str(feature.get("normalization_mode", NORMALIZATION_TRAINSET))
    background_subtraction = str(feature.get("background_subtraction", BACKGROUND_SUBTRACTION_NONE))
    resolved_cfg_path = cfg_path or _path_or_none(feature.get("cfg_path"))
    if resolved_cfg_path is None:
        raise RuntimeError("Checkpoint does not include cfg_path; pass --cfg explicitly.")
    cfg = parse_tensor_cfg(resolved_cfg_path)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    class_names_raw = checkpoint.get("class_names")
    class_names = None if class_names_raw is None else [str(item) for item in class_names_raw]
    task = "classification" if class_names is not None else "regression"
    output_dim = len(class_names) if class_names is not None else 1

    hidden_dims = tuple(int(value) for value in checkpoint["hidden_dims"])
    model = RadarMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    normalization = checkpoint.get("normalization") or {}
    input_dim = int(checkpoint["input_dim"])
    mean_raw = normalization.get("mean")
    std_raw = normalization.get("std")
    if mean_raw is None or std_raw is None:
        mean = np.zeros((input_dim,), dtype=np.float32)
        std = np.ones((input_dim,), dtype=np.float32)
    else:
        mean = np.asarray(mean_raw, dtype=np.float32)
        std = np.asarray(std_raw, dtype=np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32, copy=False)

    baseline_open_mean_db = None
    baseline_blocked_mean_db = None
    background_reference = None
    range_mask = None

    if feature_mode == FEATURE_MODE_ZERO_DOPPLER_DB:
        baseline_open_path = baseline_open_capture or _path_or_none(
            feature.get("baseline_open_capture")
        )
        baseline_blocked_path = baseline_blocked_capture or _path_or_none(
            feature.get("baseline_blocked_capture")
        )
        baseline_kwargs = {
            "range_side": str(feature.get("range_side", "positive")),
            "window_kind": str(feature.get("window_kind", "hann")),
            "eps": float(feature.get("eps", 1e-9)),
            "range_min_m": _optional_float(feature.get("range_min_m")),
            "range_max_m": _optional_float(feature.get("range_max_m")),
        }
        baseline_open_mean_db = (
            None
            if baseline_open_path is None
            else load_baseline_zero_doppler_mean_by_rx_db(
                baseline_open_path,
                cfg,
                **baseline_kwargs,
            )
        )
        baseline_blocked_mean_db = (
            None
            if baseline_blocked_path is None
            else load_baseline_zero_doppler_mean_by_rx_db(
                baseline_blocked_path, cfg, **baseline_kwargs
            )
        )
        range_mask = _build_range_mask_from_feature(cfg, feature)
    elif feature_mode == FEATURE_MODE_COMPLEX_COHERENT:
        background_path = background_capture or _path_or_none(feature.get("background_capture"))
        background_reference = load_background_reference(
            background_path,
            cfg,
            window_kind=str(feature.get("window_kind", "hann")),
            eps=float(feature.get("eps", 1e-9)),
            background_subtraction=background_subtraction,
        )
    else:
        raise ValueError(f"Unsupported feature_mode in checkpoint: {feature_mode!r}")

    return RadarCheckpointPredictor(
        checkpoint_path=Path(checkpoint_path),
        task=task,
        model=model,
        device=device,
        cfg=cfg,
        feature=feature,
        mean=mean,
        std=std,
        class_names=class_names,
        baseline_open_mean_db=baseline_open_mean_db,
        baseline_blocked_mean_db=baseline_blocked_mean_db,
        background_reference=background_reference,
        feature_mode=feature_mode,
        normalization_mode=normalization_mode,
        background_subtraction=background_subtraction,
        range_mask=range_mask,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a trained radar classifier/regressor on one or more `.bin` captures."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Capture `.bin` file(s) to predict.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cfg",
        type=Path,
        default=None,
        help="Optional cfg override. Defaults to cfg path stored in the checkpoint.",
    )
    parser.add_argument(
        "--baseline-open-capture",
        type=Path,
        default=None,
        help="Optional open-baseline override. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--baseline-blocked-capture",
        type=Path,
        default=None,
        help="Optional blocked-baseline override. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--background-capture",
        type=Path,
        default=None,
        help="Optional empty-scene capture override for complex background subtraction.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device selection.",
    )
    return parser.parse_args()


def main() -> int:
    """Run checkpoint inference on recorded captures."""
    args = parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0 when set")

    predictor = load_radar_predictor(
        args.checkpoint,
        cfg_path=args.cfg,
        baseline_open_capture=args.baseline_open_capture,
        baseline_blocked_capture=args.baseline_blocked_capture,
        background_capture=args.background_capture,
        device_name=args.device,
    )

    outputs: list[dict[str, object]] = []
    for capture_path in args.inputs:
        summary = predictor.predict_capture(capture_path, max_frames=args.max_frames)
        payload = {"input": str(capture_path), **summary.to_dict()}
        outputs.append(payload)
        if not args.json:
            print(_format_prediction_line(capture_path, summary))

    if args.json:
        print(json.dumps(outputs, indent=2))
    return 0


def _format_prediction_line(capture_path: Path, summary: PredictionSummary) -> str:
    """Format one human-readable prediction line."""
    if summary.task == "classification":
        confidence = 0.0 if summary.confidence is None else summary.confidence
        return (
            f"{capture_path}: class={summary.primary} "
            f"confidence={confidence:.3f} frames={summary.frame_count} "
            f"samples={summary.sample_count or summary.frame_count} {summary.detail}"
        )

    value = 0.0 if summary.value is None else summary.value
    spread = 0.0 if summary.spread is None else summary.spread
    return (
        f"{capture_path}: value={value:.4g} "
        f"frame_std={spread:.4g} frames={summary.frame_count} "
        f"samples={summary.sample_count or summary.frame_count}"
    )


def _build_range_mask_from_feature(
    cfg: RadarTensorConfig, feature: dict[str, Any]
) -> np.ndarray | None:
    """Build live range gate mask matching a checkpoint feature recipe."""
    range_min_m = _optional_float(feature.get("range_min_m"))
    range_max_m = _optional_float(feature.get("range_max_m"))
    if range_min_m is None and range_max_m is None:
        return None
    axis_m = compute_range_axis_m(cfg)
    range_side = str(feature.get("range_side", "positive"))
    if range_side != "full":
        axis_m = select_useful_range_side(axis_m, side=range_side)
    distances_m = np.abs(np.asarray(axis_m, dtype=np.float32))
    lo = 0.0 if range_min_m is None else range_min_m
    hi = np.inf if range_max_m is None else range_max_m
    mask = (distances_m >= lo) & (distances_m <= hi)
    if not np.any(mask):
        raise RuntimeError(
            "Checkpoint range gate removed all bins. "
            f"Requested [{lo}, {hi}] m with range_side={range_side!r}."
        )
    return mask


def _path_or_none(raw: object) -> Path | None:
    """Convert checkpoint path metadata into a Path, preserving None."""
    if raw is None:
        return None
    text = str(raw)
    if not text:
        return None
    return Path(text)


def _optional_float(raw: object) -> float | None:
    """Convert optional numeric checkpoint metadata into float."""
    if raw is None:
        return None
    return float(raw)


if __name__ == "__main__":
    raise SystemExit(main())
