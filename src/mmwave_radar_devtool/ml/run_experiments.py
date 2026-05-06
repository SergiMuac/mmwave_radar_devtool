"""Run a grid of classifier experiments from a JSON/YAML config."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from itertools import product
from pathlib import Path
from typing import Any

from .data import BACKGROUND_SUBTRACTION_NONE, FEATURE_MODE_COMPLEX_COHERENT
from .train_classification import parse_args as parse_train_args
from .train_classification import train_classifier

_PATH_PASSTHROUGH_KEYS = {
    "baseline_open_capture",
    "baseline_blocked_capture",
    "background_capture",
}


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _safe_token(name: str, value: object) -> str:
    text = str(value)
    text = text.replace(" ", "_").replace("/", "-")
    text = text.replace(".", "p")
    return f"{name}-{text}"


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency path.
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Use JSON config or install PyYAML."
            ) from exc
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("YAML config root must be a mapping.")
        return payload
    raise RuntimeError("Config file must end with .json, .yaml, or .yml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse experiment-runner arguments."""
    parser = argparse.ArgumentParser(description="Run classification experiment grid.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute experiment combinations and save per-run summaries."""
    args = parse_args(argv)
    payload = _load_config(args.config)
    cfg = payload.get("experiments", payload)
    if not isinstance(cfg, dict):
        raise RuntimeError("Experiment config must be a mapping.")

    dataset_dir = cfg.get("dataset_dir")
    cfg_path = cfg.get("cfg")
    out_dir = cfg.get("out_dir")
    if dataset_dir is None or cfg_path is None or out_dir is None:
        raise RuntimeError("Config requires dataset_dir, cfg, and out_dir.")

    target_ranges = _as_list(cfg.get("target_range_m", [0.40]))
    range_gate_bins = _as_list(cfg.get("range_gate_bins", [2]))
    feature_modes = _as_list(cfg.get("feature_mode", ["zero_doppler_db"]))
    background_modes = _as_list(cfg.get("background_subtraction", ["none"]))
    normalization_modes = _as_list(cfg.get("normalization_mode", ["trainset_standardize"]))
    seeds = _as_list(cfg.get("seeds", [42]))

    out_root = Path(str(out_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    scalar_passthrough = {
        "batch_size": cfg.get("batch_size"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "dropout": cfg.get("dropout"),
        "hidden_dims": cfg.get("hidden_dims"),
        "val_ratio": cfg.get("val_ratio"),
        "test_ratio": cfg.get("test_ratio"),
        "early_stop_patience": cfg.get("early_stop_patience"),
        "min_epochs": cfg.get("min_epochs"),
        "range_side": cfg.get("range_side"),
        "window_kind": cfg.get("window_kind"),
        "eps": cfg.get("eps"),
        "sample_mode": cfg.get("sample_mode"),
        "max_frames_per_capture": cfg.get("max_frames_per_capture"),
        "flat_label_regex": cfg.get("flat_label_regex"),
        "device": cfg.get("device"),
        "range_min_m": cfg.get("range_min_m"),
        "range_max_m": cfg.get("range_max_m"),
        "baseline_open_capture": cfg.get("baseline_open_capture"),
        "baseline_blocked_capture": cfg.get("baseline_blocked_capture"),
        "background_capture": cfg.get("background_capture"),
        "epochs": cfg.get("epochs", cfg.get("max_epochs")),
    }

    save_best_val_cfg = cfg.get("save_best_val")

    combos = list(
        product(
            feature_modes,
            target_ranges,
            range_gate_bins,
            background_modes,
            normalization_modes,
            seeds,
        )
    )
    if args.limit is not None:
        combos = combos[: max(0, int(args.limit))]

    completed = 0
    skipped = 0
    failed = 0
    summaries: list[dict[str, object]] = []

    for feature_mode, target_range_m, gate_bins, bg_mode, norm_mode, seed in combos:
        if feature_mode != FEATURE_MODE_COMPLEX_COHERENT and bg_mode != BACKGROUND_SUBTRACTION_NONE:
            print(
                "Skipping unsupported combo "
                f"feature_mode={feature_mode} background_subtraction={bg_mode}"
            )
            skipped += 1
            continue

        run_name = "_".join(
            [
                _safe_token("fm", feature_mode),
                _safe_token("tr", target_range_m),
                _safe_token("gate", gate_bins),
                _safe_token("bg", bg_mode),
                _safe_token("norm", norm_mode),
                _safe_token("seed", seed),
            ]
        )
        out_path = out_root / f"{run_name}.pt"
        out_summary = out_path.with_suffix(".json")

        if not args.force and out_path.exists() and out_summary.exists():
            print(f"Skipping existing run: {run_name}")
            skipped += 1
            continue

        train_args = parse_train_args(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--cfg",
                str(cfg_path),
                "--out",
                str(out_path),
            ]
        )
        train_args.feature_mode = str(feature_mode)
        train_args.target_range_m = float(target_range_m)
        train_args.range_gate_bins = int(gate_bins)
        train_args.background_subtraction = str(bg_mode)
        train_args.normalization_mode = str(norm_mode)
        train_args.seed = int(seed)
        train_args.plots = bool(args.plots)

        if save_best_val_cfg is not None:
            train_args.save_best_val = bool(save_best_val_cfg)

        for key, value in scalar_passthrough.items():
            if value is None:
                continue
            if key in _PATH_PASSTHROUGH_KEYS:
                setattr(train_args, key, Path(str(value)))
                continue
            setattr(train_args, key, value)

        try:
            summary = train_classifier(train_args)
            summaries.append(summary)
            completed += 1
            print(f"Completed run: {run_name}")
        except Exception as exc:
            failed += 1
            print(f"Run failed: {run_name}: {exc}")

    manifest = {
        "config": str(args.config.resolve()),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "runs": summaries,
    }
    manifest_path = out_root / "experiment_runs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "Experiment sweep finished "
        f"completed={completed} skipped={skipped} failed={failed}"
    )
    print(f"Saved manifest: {manifest_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
