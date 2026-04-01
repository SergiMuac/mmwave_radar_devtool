"""Helpers for interpreting recorded raw capture files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True, frozen=True)
class RawInt16Samples:
    """Minimal representation of raw interleaved int16 samples."""

    samples: np.ndarray

    @property
    def i(self) -> np.ndarray:
        """Return even-indexed values."""
        return self.samples[0::2]

    @property
    def q(self) -> np.ndarray:
        """Return odd-indexed values."""
        return self.samples[1::2]


def read_raw_int16(path: str | Path, max_values: int | None = None) -> RawInt16Samples:
    """Read a capture file as little-endian signed 16-bit samples."""
    data = np.fromfile(Path(path), dtype="<i2", count=max_values if max_values is not None else -1)
    return RawInt16Samples(samples=data)
