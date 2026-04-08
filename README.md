# mmwave_radar_devtool

Utilities for recording raw TI mmWave radar captures and visualizing saved radar `.bin`
files.

This README covers only:

- `dataset_recorder.bash`
- the `mmw capture` single capture command
- the `mmw live` terminal viewer
- `radar_bin_visualizer_fixed.py`

## Installation

Install UV if you don't have it yet:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repo and install the Python dependencies with:

```bash
uv sync
```

The recorder uses the `mmw` command provided by this project, so run it from the
repository root after installing the environment.

## Single Capture

Use `mmw capture` to record one raw radar `.bin` file for a fixed duration:

```bash
uv run mmw capture \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --duration 5 \
  --output capture.bin
```

The command prints capture statistics after the recording finishes, including
received packets, bytes written, elapsed time, and detected sequence gaps.

Use `--keep-dca-header` if the DCA1000 packet headers should stay in the output
file instead of being stripped.

## Live Viewer

Use `mmw live` to run the terminal live viewer with the radar configuration:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg
```

Run the live viewer for a fixed duration:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --duration 10
```

Record while using the live viewer:

```bash
uv run mmw live \
  --radar-cli-port /dev/ttyACM0 \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output live_capture.bin
```

## Dataset Recorder

`dataset_recorder.bash` repeatedly records 5 second raw radar captures using:

- config file: `config/xwr18xx_profile_raw_capture.cfg`
- radar CLI serial port: `/dev/ttyACM0`
- output format: `<label>_<record_num>.bin`

Run it with:

```bash
bash dataset_recorder.bash
```

The script prompts for a recording label, then writes files such as:

```text
person_0.bin
person_1.bin
person_2.bin
```

After each capture, choose the next action:

- press `Enter` or type `c` to continue with the same label
- type `l` to enter a new label and restart numbering at `0`
- type `q` to quit

## Radar Bin Visualizer

`radar_bin_visualizer_fixed.py` loads a raw radar `.bin` capture, reshapes it
using a TI mmWave configuration, prints a capture summary, and plots:

- raw waveform by RX channel
- intensity envelope by RX channel
- beat spectrum by RX channel
- range profile by RX channel
- combined range profile
- range-Doppler map for one RX channel

Typical usage:

```bash
uv run python radar_bin_visualizer_fixed.py person_0.bin \
  --cfg config/xwr18xx_profile_raw_capture.cfg
```

Save plots as PNG files without opening interactive windows:

```bash
uv run python radar_bin_visualizer_fixed.py person_0.bin \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --output-dir plots \
  --no-show
```

The saved files are:

```text
raw_waveform_by_rx.png
intensity_by_rx.png
spectrum_by_rx.png
range_profile_by_rx.png
range_profile_combined.png
range_doppler_rx1.png
```

Use a different RX channel for the range-Doppler plot with `--rx-for-rd`:

```bash
uv run python radar_bin_visualizer_fixed.py person_0.bin \
  --cfg config/xwr18xx_profile_raw_capture.cfg \
  --rx-for-rd 2
```

If a `.cfg` file is not available, provide the required capture layout manually:

```bash
uv run python radar_bin_visualizer_fixed.py person_0.bin \
  --num-rx 4 \
  --samples-per-chirp 256 \
  --chirps-per-frame 128
```

Optional manual parameters include:

- `--num-tx`
- `--frame-period-ms`
- `--adc-sample-rate-ksps`
- `--freq-slope-mhz-per-us`
- `--start-freq-ghz`
- `--idle-time-us`
- `--ramp-end-time-us`
- `--real-only`

## License

This project is licensed under the Apache License 2.0. See `LICENSE`.
