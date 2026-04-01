"""UART controller for the radar CLI port."""

from __future__ import annotations

import time

import serial

from .cfg_parser import RadarCliConfig
from .config import RadarSerialConfig
from .exceptions import RadarSerialError


class RadarSerialController:
    """Send CLI commands and configuration lines to the radar."""

    _PROMPT_MARKERS = ("mmwDemo:/>", "/>")
    _SUCCESS_MARKERS = ("Done", "Ignored:")
    _ERROR_MARKERS = ("Error", "not recognized as a CLI command")

    def __init__(self, config: RadarSerialConfig) -> None:
        """Initialize the controller."""
        self._config = config
        self._serial: serial.Serial | None = None

    def open(self) -> None:
        """Open the CLI serial port."""
        if self._serial is not None:
            return
        self._serial = serial.Serial(
            port=self._config.cli_port,
            baudrate=self._config.cli_baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self._config.cli_timeout_s,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()
        self._drain_startup_banner()

    def close(self) -> None:
        """Close the CLI serial port."""
        if self._serial is None:
            return
        self._serial.close()
        self._serial = None

    def __enter__(self) -> RadarSerialController:
        """Open the controller context."""
        self.open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close the controller context."""
        self.close()

    def send_command(self, command: str, readback_timeout_s: float | None = None) -> str:
        """Send one CLI command and collect immediate readback."""
        if self._serial is None:
            raise RadarSerialError("Radar serial port is not open")

        sanitized = command.strip()
        if not sanitized:
            raise RadarSerialError("Refusing to send an empty CLI command")

        payload = sanitized.encode("ascii", errors="strict") + b"\r\n"
        if self._config.debug_serial:
            print(f"radar_cli tx: {sanitized}")
            print(f"radar_cli tx_hex: {payload.hex(' ')}")

        self._serial.write(payload)
        self._serial.flush()
        response = self._read_command_response(
            command=sanitized,
            timeout_s=readback_timeout_s or self._config.command_timeout_s,
        )

        if self._config.debug_serial and response:
            print(f"radar_cli rx: {response.rstrip()}")

        return response

    def send_cfg(self, cfg: RadarCliConfig) -> list[tuple[str, str]]:
        """Send a full CLI configuration file."""
        return self.send_cfg_lines(cfg.texts())

    def send_cfg_lines(self, commands: list[str]) -> list[tuple[str, str]]:
        """Send a sequence of already-sanitized CLI commands."""
        results: list[tuple[str, str]] = []
        total = len(commands)
        for index, line in enumerate(commands, start=1):
            if self._config.verbose:
                print(f"radar_cli[{index}/{total}]: {line}")
            response = self.send_command(line)
            if self._config.verbose and response.strip():
                summary = " ".join(part.strip() for part in response.splitlines() if part.strip())
                print(f"radar_cli[{index}/{total}] rx: {summary}")
            results.append((line, response))
        self._wait_after_configuration()
        return results

    def sensor_start(self) -> str:
        """Start radar sensing."""
        return self.send_command("sensorStart")

    def sensor_stop(self) -> str:
        """Stop radar sensing."""
        return self.send_command("sensorStop")

    def flush_cfg(self) -> str:
        """Reset active CLI configuration in the radar firmware."""
        return self.send_command("flushCfg")

    def _drain_startup_banner(self) -> None:
        """Read any prompt or banner emitted when the port opens."""
        if self._serial is None:
            raise RadarSerialError("Radar serial port is not open")
        self._read_until_idle(timeout_s=self._config.startup_drain_timeout_s)

    def _read_command_response(self, command: str, timeout_s: float) -> str:
        """Read one command response until a trailing prompt or stable terminal state arrives."""
        if self._serial is None:
            raise RadarSerialError("Radar serial port is not open")

        deadline = time.monotonic() + timeout_s
        idle_deadline = time.monotonic() + self._config.prompt_idle_timeout_s
        chunks: list[bytes] = []
        saw_terminal_marker = False
        saw_command_echo = False

        while time.monotonic() < deadline:
            waiting = self._serial.in_waiting
            if waiting > 0:
                chunks.append(self._serial.read(waiting))
                idle_deadline = time.monotonic() + self._config.prompt_idle_timeout_s
                continue

            if chunks and time.monotonic() >= idle_deadline:
                decoded = b"".join(chunks).decode("utf-8", errors="replace")
                saw_command_echo = saw_command_echo or command in decoded
                saw_terminal_marker = saw_terminal_marker or any(
                    marker in decoded for marker in self._SUCCESS_MARKERS + self._ERROR_MARKERS
                )
                has_trailing_prompt = any(
                    decoded.rstrip().endswith(marker) for marker in self._PROMPT_MARKERS
                )

                if saw_command_echo and has_trailing_prompt:
                    return decoded
                if saw_terminal_marker and not self._serial.in_waiting:
                    return decoded

                idle_deadline = time.monotonic() + self._config.prompt_idle_timeout_s

            time.sleep(self._config.poll_interval_s)

        return b"".join(chunks).decode("utf-8", errors="replace")

    def _wait_after_configuration(self) -> None:
        """Allow the firmware to settle after the full cfg has been applied."""
        if self._serial is None:
            raise RadarSerialError("Radar serial port is not open")
        deadline = time.monotonic() + self._config.post_config_settle_s
        while time.monotonic() < deadline:
            self._read_until_idle(timeout_s=self._config.prompt_idle_timeout_s)
            time.sleep(self._config.poll_interval_s)

    def _read_until_idle(self, timeout_s: float) -> str:
        """Read all currently available serial data for a short idle window."""
        if self._serial is None:
            raise RadarSerialError("Radar serial port is not open")

        deadline = time.monotonic() + timeout_s
        idle_deadline = time.monotonic() + self._config.prompt_idle_timeout_s
        chunks: list[bytes] = []

        while time.monotonic() < deadline:
            waiting = self._serial.in_waiting
            if waiting > 0:
                chunks.append(self._serial.read(waiting))
                idle_deadline = time.monotonic() + self._config.prompt_idle_timeout_s
                continue

            if chunks and time.monotonic() >= idle_deadline:
                break

            time.sleep(self._config.poll_interval_s)

        return b"".join(chunks).decode("utf-8", errors="replace")
