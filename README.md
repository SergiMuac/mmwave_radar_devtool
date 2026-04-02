# mmwave_radar_devtool

Python toolkit for working with TI mmWave radar hardware using the TI CLI and `DCA1000EVM` capture card.

Supported board profiles currently include:

- `IWR1843 + DCA1000`
- `IWR6843 + DCA1000`
- `IWR6843 USB Telemetry`
- `IWR6843AOP + DCA1000`
- `IWR6843AOP USB Telemetry`
- `generic-ti-dca1000` for other compatible TI setups

## Features

- Configure the radar over the CLI serial port
- Probe the radar and capture-card setup before acquisition
- Record raw ADC data to disk
- Run a live terminal view during capture
- Run a live terminal view from TI USB telemetry output
- Generate capture-ready radar configuration files

## Installation

```bash
uv sync
```

For development dependencies:

```bash
uv sync --extra dev
```

## CLI

The package installs the `mmw` command.

### List supported profiles

```bash
uv run mmw profiles
```

### Probe hardware

```bash
uv run mmw probe \
  --radar-cli-port /dev/ttyACM0 \
  --profile iwr1843-dca1000 \
  --cfg config/xwr18xx_profile_raw_capture.cfg
```

### Capture to file

```bash
uv run mmw capture \
  --radar-cli-port /dev/ttyACM0 \
  --profile iwr1843-dca1000 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output capture.bin \
  --duration 3
```

### Live terminal dashboard

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --profile iwr1843-dca1000 \
  --cfg config/xwr18xx_profile_raw_capture.cfg
```

Optional bounded run:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --profile iwr1843-dca1000 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --duration 10
```

### Live terminal dashboard while recording

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --profile iwr1843-dca1000 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output capture.bin
```

### Plot a raw capture

```bash
uv run mmw plot \
  --input capture.bin \
  --samples 4096
```

### Generate a capture-ready config

```bash
uv run mmw make-capture-cfg \
  --input config/xwr18xx_profile_raw_capture.cfg \
  --output config/generated_capture.cfg
```

Example for `IWR6843AOP`:

```bash
uv run mmw probe \
  --radar-cli-port /dev/ttyACM0 \
  --profile iwr6843aop-dca1000 \
  --cfg config/iwr6843aop_profile_raw_capture.cfg
```

Example for `IWR6843AOP` over USB telemetry only:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --radar-data-port /dev/ttyACM1 \
  --profile iwr6843aop-usb \
  --cfg config/iwr6843aop_usb_point_cloud.cfg
```

For USB telemetry profiles, `live` reads the factory TI demo output stream from the data UART. This is not raw ADC capture, so `capture` is only supported on DCA1000-based profiles.

Example for `IWR6843` over USB telemetry:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyUSB0 \
  --radar-data-port /dev/ttyUSB1 \
  --profile iwr6843-usb \
  --cfg config/xwr68xx_profile_2026_04_02T12_19_57_603.cfg
```

## Development

Run tests with:

```bash
uv run pytest
```

## License

This project is licensed under the Apache License 2.0. See `LICENSE`.
