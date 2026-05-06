from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

from pathlib import Path

import numpy as np

from mmwave_radar_devtool.ml.visualize_dataset import _plot_range_profile_overlays_by_label


def test_plot_range_profile_overlays_by_label(tmp_path: Path) -> None:
    """Overlay plot helper should save a figure for stacked capture runs."""
    range_axis_m = np.linspace(0.0, 0.5, 6, dtype=np.float32)
    profiles_by_label = {
        "water-0": [
            np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]], dtype=np.float32),
            np.array([[2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]], dtype=np.float32),
        ],
        "water-5": [
            np.array([[3, 3, 3, 3, 3, 3], [4, 4, 4, 4, 4, 4]], dtype=np.float32),
            np.array([[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5]], dtype=np.float32),
        ],
    }

    output_path = tmp_path / "overlay.png"
    _plot_range_profile_overlays_by_label(
        profiles_by_label,
        title="Overlay",
        output_path=output_path,
        range_axis_m=range_axis_m,
        eps=1e-9,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0