"""Tests for TI USB telemetry parsing and CLI integration."""

from __future__ import annotations

import io
import struct
from contextlib import redirect_stdout

import pytest

from mmwave_radar_devtool.config import CaptureConfig, RadarDataSerialConfig
from mmwave_radar_devtool.exceptions import ConfigurationError
from mmwave_radar_devtool.cli import _handle_profiles
from mmwave_radar_devtool.usb_telemetry import (
    TI_MMWAVE_MAGIC_WORD,
    TiUsbTelemetryStream,
    TiUsbTelemetryParser,
)


def _build_frame() -> bytes:
    range_profile_payload = struct.pack("<4H", 10, 20, 30, 40)
    points_payload = struct.pack("<ffff", 1.0, 2.0, 3.0, 0.5)
    points_tlv = struct.pack("<II", 1, 8 + len(points_payload)) + points_payload
    range_profile_tlv = (
        struct.pack("<II", 2, 8 + len(range_profile_payload)) + range_profile_payload
    )
    side_info_payload = struct.pack("<HH", 120, 45)
    side_info_tlv = struct.pack("<II", 7, 8 + len(side_info_payload)) + side_info_payload
    total_packet_length = 40 + len(points_tlv) + len(range_profile_tlv) + len(side_info_tlv)
    header = TI_MMWAVE_MAGIC_WORD + struct.pack(
        "<8I",
        0x03060000,
        total_packet_length,
        0xA6843,
        7,
        1234,
        1,
        3,
        0,
    )
    return header + points_tlv + range_profile_tlv + side_info_tlv


def test_ti_usb_parser_decodes_detected_points_and_side_info() -> None:
    """The parser should decode standard TI detected-point telemetry packets."""
    parser = TiUsbTelemetryParser()
    frame_bytes = _build_frame()

    frames = parser.feed(frame_bytes[:19])
    frames.extend(parser.feed(frame_bytes[19:]))

    assert len(frames) == 1
    frame = frames[0]
    assert frame.frame_number == 7
    assert frame.detected_points == 1
    assert frame.points[0].range_m > 3.7
    assert frame.points[0].snr_db == 12.0
    assert frame.points[0].noise_db == 4.5
    assert frame.range_profile == (10.0, 20.0, 30.0, 40.0)


def test_no_frame_error_explains_missing_bytes() -> None:
    """The stream should explain when no UART bytes were received."""
    with pytest.raises(ConfigurationError, match="No bytes were received"):
        raise ConfigurationError(TiUsbTelemetryStream._build_no_frame_error(0))


def test_no_frame_error_explains_undecodable_bytes() -> None:
    """The stream should explain when bytes arrive but no TI frames decode."""
    with pytest.raises(ConfigurationError, match="Received bytes from the radar data UART"):
        raise ConfigurationError(TiUsbTelemetryStream._build_no_frame_error(128))


class _FakeTelemetrySerial:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    @property
    def in_waiting(self) -> int:
        return len(self._chunks[0]) if self._chunks else 0

    def read(self, _: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""

    def close(self) -> None:
        return None


def test_stream_times_out_when_bytes_arrive_but_no_frame_decodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The telemetry stream should fail fast on undecodable byte streams."""
    stream = TiUsbTelemetryStream(
        RadarDataSerialConfig(
            data_port="/dev/null", first_frame_timeout_s=0.01, poll_interval_s=0.001
        )
    )
    stream._serial = _FakeTelemetrySerial([b"abc", b"def", b""])  # type: ignore[assignment]

    moments = iter([0.0, 0.0, 0.02, 0.02, 0.03])
    monkeypatch.setattr(
        "mmwave_radar_devtool.usb_telemetry.time.monotonic",
        lambda: next(moments),
    )

    with pytest.raises(ConfigurationError, match="could not decode any TI telemetry frames"):
        stream.stream(CaptureConfig())


def test_profiles_cli_lists_usb_profile() -> None:
    """The profiles command should print the new USB telemetry profile."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _handle_profiles(object())
    output = buffer.getvalue()
    assert "iwr6843aop-usb" in output
    assert "supports_usb_telemetry=True" in output
