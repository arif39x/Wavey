# Byakugan

**See the invisible.** Real-time Wi-Fi signal heatmapper that turns your
laptop into an RF vision tool. Walk around a building, click your
position, and watch the heatmap build like a live thermal camera.

The Byakugan — the "white eye" — sees chakra, detects hidden threats,
and perceives the world beyond normal sight. This tool does the same for
Wi-Fi: rendering the invisible radio landscape of your environment in
real time.

## Architecture

Three decoupled pipelines running on separate threads:

1. **Ingestion** (`ingestion/networking.py`) — polls the OS for live
   RSSI readings on a background `QThread`.
2. **Math** (`processing/matrix_math.py`) — computes a physics-based
   prediction (log-distance path loss + wall attenuation via 2D ray
   tracing), auto-estimates router position from signal data, and
   corrects with residual interpolation (RBF) from measured points.
3. **Presentation** (`presentation/gui.py` / `main.py`) — PyQt6 window
   with embedded matplotlib canvas, live scan mode, and instant heatmap
   updates.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Per-OS Setup

### Linux

Reads `/proc/net/wireless` — no special permissions needed. Falls back
to `iw dev` or `iwconfig`. Most distros work out of the box.

```bash
cat /proc/net/wireless   # Verify interface is visible
```

### Windows

Runs `netsh wlan show interfaces`, converts percentage to dBm via
`dBm = (percent / 2) - 100` (rough heuristic). No admin needed.

### macOS

Uses the `airport` binary at:
```
/System/Library/PrivateFrameworks/Apple80211.framework/
  Versions/Current/Resources/airport -I
```
May be missing on Ventura+ — app falls back to simulation automatically.

## How to Use

### Live Scan (recommended)

1. Select **"Live Scan"** mode.
2. Click your physical position on the floor plan — each click records
   the current RSSI and the heatmap updates instantly.
3. Walk to another spot and click again. The heatmap grows with every
   point.
4. Enable **"Continuous record"** to auto-sample every 1 second.

### Auto-Detect Router

The router is automatically estimated from your data points using a
signal-strength-weighted centroid. No manual placement required.

### Drawing Walls

1. Select **"Draw Wall"** mode.
2. Choose a material: Drywall (-3 dB), Brick (-12 dB), Reinforced
   Concrete (-25 dB).
3. Click and drag to draw wall segments.

## Calibration

Constants in `processing/matrix_math.py`:

- **CELL_METERS** (default 0.3): measure a known distance on your floor
  plan, count grid cells, set `CELL_METERS = meters / cells`.
- **TX_POWER_DBM** (default -30.0): stand 1 cell from the router, note
  live RSSI, set to that value.

## Simulated vs. Real

- **Real**: reads live RSSI from OS (interface name shown in UI).
- **Simulated**: after 5 consecutive failures, falls back to a realistic
  sine+walk+noise model. UI always labels the source.

## Limitations

- Ray tracing models direct-path blocking only (no reflection/diffraction).
  Residual correction from measurements compensates where data exists.
- Single router only.
- 2.4 GHz reference attenuation values.
- RBF interpolation used (not Kriging) — zero extra dependency.
- Router centroid estimate biased when data comes from one side only.
