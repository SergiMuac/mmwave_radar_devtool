from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mmwave_radar_devtool.config import RadarSerialConfig
from mmwave_radar_devtool.serial_control import RadarSerialController


@dataclass
class FakeSerial:
    chunks: list[bytes]
    writes: list[bytes] = field(default_factory=list)
    is_open: bool = True

    @property
    def in_waiting(self) -> int:
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, _: int) -> bytes:
        if not self.chunks:
            return b""
        return self.chunks.pop(0)

    def write(self, payload: bytes) -> int:
        self.writes.append(payload)
        return len(payload)

    def flush(self) -> None:
        return None

    def reset_input_buffer(self) -> None:
        return None

    def reset_output_buffer(self) -> None:
        return None

    def close(self) -> None:
        self.is_open = False


def build_controller(chunks: Iterable[bytes]) -> RadarSerialController:
    controller = RadarSerialController(
        RadarSerialConfig(
            cli_port="/dev/null",
            command_timeout_s=0.2,
            prompt_idle_timeout_s=0.02,
            post_config_settle_s=0.01,
        )
    )
    controller._serial = FakeSerial(list(chunks))
    return controller


def test_send_command_writes_ascii_crlf() -> None:
    controller = build_controller([b"sensorStop\r\nDone\r\nmmwDemo:/> "])
    response = controller.send_command("sensorStop")
    assert response.endswith("mmwDemo:/> ")
    assert controller._serial.writes == [b"sensorStop\r\n"]


def test_send_command_returns_after_terminal_marker_without_prompt() -> None:
    controller = build_controller([b"Ignored: Sensor is already stopped\r\n", b"Done\r\n"])
    response = controller.send_command("sensorStop")
    assert "Ignored:" in response
    assert "Done" in response


def test_send_command_discards_stale_prompt_before_new_command() -> None:
    controller = build_controller(
        [
            b"mmwDemo:/> ",
            b"flushCfg\r\nDone\r\nmmwDemo:/> ",
        ]
    )
    response = controller.send_command("flushCfg")
    assert "flushCfg" in response
    assert response.endswith("mmwDemo:/> ")
    assert controller._serial.writes == [b"flushCfg\r\n"]


def test_send_command_waits_for_prompt_after_split_chunks() -> None:
    controller = build_controller(
        [
            b"flushCfg\r\nDon",
            b"e\r\nmmwDemo:/> ",
        ]
    )
    response = controller.send_command("flushCfg")
    assert "Done" in response
    assert response.endswith("mmwDemo:/> ")
