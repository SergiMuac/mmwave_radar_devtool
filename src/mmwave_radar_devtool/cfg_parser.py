"""Parser and validator for TI mmWave CLI configuration files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DCA1000DataLoggingMode
from .exceptions import ConfigurationError


@dataclass(slots=True, frozen=True)
class RadarCliLine:
    """A single CLI instruction line."""

    line_number: int
    text: str


@dataclass(slots=True, frozen=True)
class AdcCfg:
    """Decoded adcCfg settings relevant for DCA1000 capture."""

    num_adc_bits_code: int
    adc_output_format_code: int

    @property
    def data_format_mode(self) -> int:
        """Return the DCA1000 data format code derived from adcCfg."""
        return self.num_adc_bits_code + 1

    @property
    def is_complex(self) -> bool:
        """Return whether the ADC output is complex."""
        return self.adc_output_format_code in {1, 2}


@dataclass(slots=True, frozen=True)
class LvdsStreamCfg:
    """Decoded lvdsStreamCfg settings relevant for DCA1000 capture."""

    subframe_idx: int
    enable_header: bool
    enable_hw_stream: bool
    enable_sw_stream: bool

    @property
    def data_logging_mode(self) -> DCA1000DataLoggingMode:
        """Return the required DCA1000 logging mode for this LVDS setup."""
        if self.enable_header:
            return DCA1000DataLoggingMode.MULTI
        return DCA1000DataLoggingMode.RAW


@dataclass(slots=True, frozen=True)
class ProfileCfg:
    """Decoded profileCfg fields used for lightweight live post-processing."""

    start_frequency_ghz: float
    idle_time_us: float
    adc_start_time_us: float
    ramp_end_time_us: float
    freq_slope_mhz_per_us: float
    num_adc_samples: int
    dig_out_sample_rate_ksps: int


@dataclass(slots=True, frozen=True)
class AdcBufCfg:
    """Decoded adcbufCfg fields relevant for IQ sample ordering."""

    subframe_idx: int
    adc_output_format: int
    sample_swap: int
    channel_interleave: int
    chirp_threshold: int

    @property
    def q_first(self) -> bool:
        """Return whether the stream orders Q samples before I samples."""
        return self.sample_swap == 1


@dataclass(slots=True, frozen=True)
class ChannelCfg:
    """Decoded channelCfg fields used for channel count metadata."""

    rx_channel_enable_mask: int
    tx_channel_enable_mask: int
    cascading: int

    @property
    def num_enabled_rx(self) -> int:
        """Return how many RX channels are enabled by the mask."""
        return int(bin(self.rx_channel_enable_mask & 0xF).count("1"))


@dataclass(slots=True, frozen=True)
class RadarCaptureRequirements:
    """Capture settings derived from the radar cfg."""

    adc_cfg: AdcCfg
    lvds_stream_cfg: LvdsStreamCfg

    @property
    def data_format_mode(self) -> int:
        """Return the DCA1000 data format mode value."""
        return self.adc_cfg.data_format_mode

    @property
    def data_logging_mode(self) -> DCA1000DataLoggingMode:
        """Return the DCA1000 data logging mode value."""
        return self.lvds_stream_cfg.data_logging_mode


@dataclass(slots=True, frozen=True)
class RadarCliConfig:
    """Parsed CLI configuration file."""

    path: Path
    commands: tuple[RadarCliLine, ...]

    def has_command_prefix(self, prefix: str) -> bool:
        """Return whether any line starts with the given prefix."""
        return any(line.text.startswith(prefix) for line in self.commands)

    def texts(self) -> list[str]:
        """Return plain command texts in order."""
        return [line.text for line in self.commands]

    def command_texts_excluding(self, prefixes: tuple[str, ...]) -> list[str]:
        """Return command texts excluding commands with selected prefixes."""
        return [
            line.text
            for line in self.commands
            if not any(line.text.startswith(prefix) for prefix in prefixes)
        ]

    def find_first(self, prefix: str) -> RadarCliLine | None:
        """Return the first CLI line whose text starts with the given prefix."""
        for line in self.commands:
            if line.text.startswith(prefix):
                return line
        return None

    def parse_adc_cfg(self) -> AdcCfg:
        """Parse the adcCfg command."""
        line = self.find_first("adcCfg")
        if line is None:
            raise ConfigurationError("Configuration file does not contain adcCfg.")

        parts = line.text.split()
        if len(parts) != 3:
            raise ConfigurationError(
                f"Malformed adcCfg command at line {line.line_number}: {line.text}"
            )

        try:
            num_adc_bits_code = int(parts[1])
            adc_output_format_code = int(parts[2])
        except ValueError as exc:
            raise ConfigurationError(
                f"Malformed adcCfg numeric fields at line {line.line_number}: {line.text}"
            ) from exc

        if num_adc_bits_code not in {0, 1, 2}:
            raise ConfigurationError(
                f"Unsupported adcCfg numADCBits value at line {line.line_number}: {num_adc_bits_code}"
            )

        return AdcCfg(
            num_adc_bits_code=num_adc_bits_code,
            adc_output_format_code=adc_output_format_code,
        )

    def parse_profile_cfg(self) -> ProfileCfg:
        """Parse the profileCfg command used by the active frame."""
        line = self.find_first("profileCfg")
        if line is None:
            raise ConfigurationError("Configuration file does not contain profileCfg.")

        parts = line.text.split()
        if len(parts) < 12:
            raise ConfigurationError(
                f"Malformed profileCfg command at line {line.line_number}: {line.text}"
            )

        try:
            return ProfileCfg(
                start_frequency_ghz=float(parts[2]),
                idle_time_us=float(parts[3]),
                adc_start_time_us=float(parts[4]),
                ramp_end_time_us=float(parts[5]),
                freq_slope_mhz_per_us=float(parts[8]),
                num_adc_samples=int(parts[10]),
                dig_out_sample_rate_ksps=int(parts[11]),
            )
        except (ValueError, IndexError) as exc:
            raise ConfigurationError(
                f"Malformed profileCfg numeric fields at line {line.line_number}: {line.text}"
            ) from exc

    def parse_adcbuf_cfg(self) -> AdcBufCfg:
        """Parse adcbufCfg for sample ordering metadata."""
        line = self.find_first("adcbufCfg")
        if line is None:
            raise ConfigurationError("Configuration file does not contain adcbufCfg.")

        parts = line.text.split()
        if len(parts) != 6:
            raise ConfigurationError(
                f"Malformed adcbufCfg command at line {line.line_number}: {line.text}"
            )

        try:
            return AdcBufCfg(
                subframe_idx=int(parts[1]),
                adc_output_format=int(parts[2]),
                sample_swap=int(parts[3]),
                channel_interleave=int(parts[4]),
                chirp_threshold=int(parts[5]),
            )
        except ValueError as exc:
            raise ConfigurationError(
                f"Malformed adcbufCfg numeric fields at line {line.line_number}: {line.text}"
            ) from exc

    def parse_channel_cfg(self) -> ChannelCfg:
        """Parse channelCfg for enabled RX/TX masks."""
        line = self.find_first("channelCfg")
        if line is None:
            raise ConfigurationError("Configuration file does not contain channelCfg.")

        parts = line.text.split()
        if len(parts) != 4:
            raise ConfigurationError(
                f"Malformed channelCfg command at line {line.line_number}: {line.text}"
            )

        try:
            rx_channel_enable_mask = int(parts[1])
            tx_channel_enable_mask = int(parts[2])
            cascading = int(parts[3])
        except ValueError as exc:
            raise ConfigurationError(
                f"Malformed channelCfg numeric fields at line {line.line_number}: {line.text}"
            ) from exc

        return ChannelCfg(
            rx_channel_enable_mask=rx_channel_enable_mask,
            tx_channel_enable_mask=tx_channel_enable_mask,
            cascading=cascading,
        )

    def parse_lvds_stream_cfg(self) -> LvdsStreamCfg:
        """Parse the lvdsStreamCfg command."""
        line = self.find_first("lvdsStreamCfg")
        if line is None:
            raise ConfigurationError("Configuration file does not contain lvdsStreamCfg.")

        parts = line.text.split()
        if len(parts) != 5:
            raise ConfigurationError(
                f"Malformed lvdsStreamCfg command at line {line.line_number}: {line.text}"
            )

        try:
            subframe_idx = int(parts[1])
            enable_header = bool(int(parts[2]))
            enable_hw_stream = bool(int(parts[3]))
            enable_sw_stream = bool(int(parts[4]))
        except ValueError as exc:
            raise ConfigurationError(
                f"Malformed lvdsStreamCfg numeric fields at line {line.line_number}: {line.text}"
            ) from exc

        return LvdsStreamCfg(
            subframe_idx=subframe_idx,
            enable_header=enable_header,
            enable_hw_stream=enable_hw_stream,
            enable_sw_stream=enable_sw_stream,
        )

    def validate_for_dca_capture(self) -> RadarCaptureRequirements:
        """Validate that the cfg enables LVDS capture for DCA1000."""
        adc_cfg = self.parse_adc_cfg()
        lvds_cfg = self.parse_lvds_stream_cfg()

        if not adc_cfg.is_complex:
            raise ConfigurationError(
                "This package currently expects complex ADC output from adcCfg for raw capture."
            )

        if not lvds_cfg.enable_hw_stream:
            raise ConfigurationError(
                "lvdsStreamCfg does not enable the hardware LVDS stream required for raw ADC capture."
            )

        return RadarCaptureRequirements(adc_cfg=adc_cfg, lvds_stream_cfg=lvds_cfg)


def _validate_cfg_line(raw_line: str, line_number: int, cfg_path: Path) -> str:
    """Validate and sanitize one cfg line for CLI transmission."""
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("%"):
        return ""

    if any(ord(character) > 127 for character in stripped):
        raise ConfigurationError(
            f"Non-ASCII character found in configuration file {cfg_path} at line {line_number}: {raw_line!r}"
        )

    if any(ord(character) < 32 and character != "\t" for character in stripped):
        raise ConfigurationError(
            f"Control character found in configuration file {cfg_path} at line {line_number}: {raw_line!r}"
        )

    return stripped


def parse_radar_cfg(path: str | Path) -> RadarCliConfig:
    """Parse a TI radar CLI configuration file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise ConfigurationError(f"Configuration file does not exist: {cfg_path}")

    try:
        # Decode as UTF-8 so non-ASCII comment text is accepted.
        raw_lines = cfg_path.read_text(encoding="utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise ConfigurationError(
            f"Configuration file must be valid UTF-8 text: {cfg_path}"
        ) from exc

    commands: list[RadarCliLine] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        sanitized = _validate_cfg_line(
            raw_line=raw_line, line_number=line_number, cfg_path=cfg_path
        )
        if not sanitized:
            continue
        commands.append(RadarCliLine(line_number=line_number, text=sanitized))

    if not commands:
        raise ConfigurationError(f"No CLI commands found in configuration file: {cfg_path}")

    return RadarCliConfig(path=cfg_path, commands=tuple(commands))


def create_capture_cfg(
    source_path: str | Path,
    target_path: str | Path,
    *,
    enable_header: bool,
    enable_hw_stream: bool,
    enable_sw_stream: bool,
) -> Path:
    """Create a capture-ready cfg by rewriting lvdsStreamCfg in an existing cfg."""
    source = Path(source_path)
    target = Path(target_path)

    lines = source.read_text(encoding="ascii", errors="strict").splitlines()
    updated_lines: list[str] = []
    found_lvds_stream_cfg = False

    replacement = (
        f"lvdsStreamCfg -1 {int(enable_header)} {int(enable_hw_stream)} {int(enable_sw_stream)}"
    )

    for line in lines:
        if line.strip().startswith("lvdsStreamCfg"):
            updated_lines.append(replacement)
            found_lvds_stream_cfg = True
            continue
        updated_lines.append(line)

    if not found_lvds_stream_cfg:
        raise ConfigurationError("Cannot create capture cfg because lvdsStreamCfg was not found.")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(updated_lines) + "\n", encoding="ascii")
    return target
