"""Minimal plotting utilities for recorded captures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .parser import read_raw_int16


def plot_raw_iq(path: str | Path, sample_count: int = 4096) -> None:
    """Plot the first raw I and Q values from a capture file."""
    raw = read_raw_int16(path=path, max_values=sample_count * 2)
    plt.figure()
    plt.plot(raw.i[:sample_count], label="I")
    plt.plot(raw.q[:sample_count], label="Q")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.title(Path(path).name)
    plt.legend()
    plt.tight_layout()
    plt.show()
