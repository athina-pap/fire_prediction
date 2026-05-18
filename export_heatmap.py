"""
export_heatmap.py
=================
Run this AFTER training to generate a self-contained interactive HTML
heatmap dashboard from real model predictions.

Usage
-----
    python export_heatmap.py                         # auto-picks top fire days
    python export_heatmap.py --dates 2021-08-06 2022-11-03 2023-08-22
    python export_heatmap.py --top 5                 # top-5 fire days
    python export_heatmap.py --out my_dashboard.html

Requirements
------------
    - unet_fire_improved_meta.json   (saved by train_unet_improved.py)
    - unet_fire_improved.weights.h5  (saved by train_unet_improved.py)
    - training_dataset_final_prepared.csv
    - tensorflow, pandas, numpy
"""

import argparse
import json
import os
import sys
import warnings
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input, Model, layers

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Export UNet heatmap dashboard")
parser.add_argument("--csv",     default="training_dataset_final_prepared.csv")
parser.add_argument("--meta",    default="unet_fire_improved_meta.json")
parser.add_argument("--weights", default="unet_fire_improved.weights.h5")
parser.add_argument("--out",     default="unet_heatmap_dashboard.html")
parser.add_argument("--dates",   nargs="+", default=None,
                    help="Specific dates to include (YYYY-MM-DD)")
parser.add_argument("--top",     type=int, default=3,
                    help="If --dates not given, use top-N fire days")
args = parser.parse_args()

# ── 1. LOAD META ─────────────────────────────────────────────────────────────

if not os.path.exists(args.meta):
    sys.exit(f"ERROR: {args.meta} not found. Run train_unet_improved.py first.")

with open(args.meta) as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
mean         = np.array(meta["mean"],  dtype=np.float32)
std          = np.array(meta["std"],   dtype=np.float32)
WINDOW       = int(meta["WINDOW"])
LAT_BIN_DEG  = float(meta["LAT_BIN_DEG"])
LON_BIN_DEG  = float(meta["LON_BIN_DEG"])
H            = int(meta["H"])
W            = int(meta["W"])
Cin          = int(meta["Cin"])

print(f"Meta loaded: grid {H}×{W}, {WINDOW}-day window, {len(feature_cols)} features")

# ── 2. REBUILD GRIDS FROM CSV ────────────────────────────────────────────────

df = pd.read_csv(args.csv)
df["date"] = pd.to_datetime(df["date"])

# Feature engineering (must match training exactly)
spei_cols = [c for c in ["SPEI_30", "SPEI_90", "SPEI_180"] if c in df.columns]
for col in spei_cols:
    df[f"{col}_obs"] = (~df[col].isnull()).astype(np.float32)
    df[col] = df[col].fillna(0.0)

base_cols = [c for c in feature_cols
             if not c.endswith("_obs") and c not in ("doy_sin", "doy_cos")]
df[base_cols] = df[base_cols].fillna(df[base_cols].median(numeric_only=True))
df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365).astype(np.float32)
df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365).astype(np.float32)

# Verify all expected features are present
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    sys.exit(f"ERROR: features missing from CSV: {missing}")

# Spatial grid
lat_min = df["lat"].min()
lon_min = df["lon"].min()
df["lat_bin"] = ((df["lat"] - lat_min) / LAT_BIN_DEG).round().astype(int)
df["lon_bin"] = ((df["lon"] - lon_min) / LON_BIN_DEG).round().astype(int)
lat_bins = np.sort(df["lat_bin"].unique())
lon_bins = np.sort(df["lon_bin"].unique())
latbin_to_i = {b: i for i, b in enumerate(lat_bins)}
lonbin_to_j = {b: j for j, b in enumerate(lon_bins)}
df["i"] = df["lat_bin"].map(latbin_to_i).astype(int)
df["j"] = df["lon_bin"].map(lonbin_to_j).astype(int)

lats = (lat_min + lat_bins * LAT_BIN_DEG).round(3).tolist()
lons = (lon_min + lon_bins * LON_BIN_DEG).round(3).tolist()

C = len(feature_cols)
dates_sorted = np.sort(df["date"].unique())
T_total      = len(dates_sorted)
date_to_t    = {d: i for i, d in enumerate(dates_sorted)}
df["t"]      = df["date"].map(date_to_t).astype(int)

# Daily tensors
frames     = np.zeros((T_total, H, W, C),  dtype=np.float32)
labels     = np.zeros((T_total, H, W),      dtype=np.float32)
valid_mask = np.zeros((T_total, H, W),      dtype=np.float32)

t_arr = df["t"].to_numpy()
i_arr = df["i"].to_numpy()
j_arr = df["j"].to_numpy()

frames    [t_arr, i_arr, j_arr, :]  = df[feature_cols].to_numpy(dtype=np.float32)
labels    [t_arr, i_arr, j_arr]     = df["y_fire"].to_numpy(dtype=np.float32)
valid_mask[t_arr, i_arr, j_arr]     = 1.0

# FWI and t2m raw (for display)
fwi_idx = feature_cols.index("fwi") if "fwi" in feature_cols else None
t2m_idx = feature_cols.index("t2m") if "t2m" in feature_cols else None

print(f"Grids built: {frames.shape}")

# ── 3. SELECT SHOWCASE DATES ─────────────────────────────────────────────────

if args.dates:
    showcase_dates = [pd.Timestamp(d) for d in args.dates]
    # Validate
    all_dates_set = set(pd.Timestamp(d) for d in dates_sorted)
    for d in showcase_dates:
        if d not in all_dates_set:
            sys.exit(f"ERROR: date {d.date()} not found in dataset")
else:
    # Auto-pick top-N by fire count, with enough history for the window
    fire_counts = (
        df.groupby("date")["y_fire"].sum()
        .reset_index()
        .rename(columns={"y_fire": "fires"})
        .sort_values("fires", ascending=False)
    )
    # Gap-aware: only keep dates that have a full contiguous 7-day window
    day_gaps = np.diff([pd.Timestamp(d).toordinal() for d in dates_sorted])
    valid_ends = set(
        t for t in range(WINDOW - 1, T_total)
        if np.all(day_gaps[t - WINDOW + 1 : t] == 1)
    )
    valid_date_set = {dates_sorted[t] for t in valid_ends}

    showcase_dates = []
    for _, row in fire_counts.iterrows():
        ts = row["date"]
        if ts in valid_date_set:
            showcase_dates.append(ts)
        if len(showcase_dates) == args.top:
            break

    if not showcase_dates:
        sys.exit("ERROR: no valid dates found (need full 7-day contiguous window)")

print(f"Showcase dates: {[str(d.date()) for d in showcase_dates]}")

# ── 4. BUILD MODEL ───────────────────────────────────────────────────────────

def conv_block(x, filters, dropout=0.0, name="cb"):
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_a1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_c2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_a2")(x)
    if dropout > 0:
        x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    return x

def build_unet(H, W, Cin, base=32, dropout=0.1):
    inp = Input(shape=(H, W, Cin), name="unet_input")
    factor = 16
    pad_h  = (factor - (H % factor)) % factor
    pad_w  = (factor - (W % factor)) % factor

    def pad_hw(x):
        if pad_h == 0 and pad_w == 0:
            return x
        return tf.pad(x, [[0,0],[0,pad_h],[0,pad_w],[0,0]])

    x  = layers.Lambda(pad_hw, name="pad_hw")(inp)
    c1 = conv_block(x,  base,    dropout, "enc1"); p1 = layers.MaxPooling2D(name="pool1")(c1)
    c2 = conv_block(p1, base*2,  dropout, "enc2"); p2 = layers.MaxPooling2D(name="pool2")(c2)
    c3 = conv_block(p2, base*4,  dropout, "enc3"); p3 = layers.MaxPooling2D(name="pool3")(c3)
    c4 = conv_block(p3, base*8,  dropout, "enc4"); p4 = layers.MaxPooling2D(name="pool4")(c4)
    bn = conv_block(p4, base*16, dropout, "bottleneck")
    u4 = layers.UpSampling2D(name="up4")(bn);  u4 = layers.Concatenate(name="cat4")([u4,c4]); d4 = conv_block(u4, base*8,  dropout, "dec4")
    u3 = layers.UpSampling2D(name="up3")(d4);  u3 = layers.Concatenate(name="cat3")([u3,c3]); d3 = conv_block(u3, base*4,  dropout, "dec3")
    u2 = layers.UpSampling2D(name="up2")(d3);  u2 = layers.Concatenate(name="cat2")([u2,c2]); d2 = conv_block(u2, base*2,  dropout, "dec2")
    u1 = layers.UpSampling2D(name="up1")(d2);  u1 = layers.Concatenate(name="cat1")([u1,c1]); d1 = conv_block(u1, base,    dropout, "dec1")
    out = layers.Conv2D(1, 1, activation="sigmoid", name="fire_prob")(d1)
    if pad_h != 0 or pad_w != 0:
        out = layers.Cropping2D(((0,pad_h),(0,pad_w)), name="crop_back")(out)
    return Model(inp, out, name="UNet")

unet = build_unet(H=H, W=W, Cin=Cin,
                  base=meta.get("unet_base", 32),
                  dropout=meta.get("unet_dropout", 0.1))
unet.load_weights(args.weights)
print(f"Model loaded from {args.weights}")

# ── 5. BUILD DATE PAYLOAD ─────────────────────────────────────────────────────

SENTINEL = -999  # marks unobserved cells in JSON

def grid_to_list(grid_2d, mask_2d, sentinel=SENTINEL, precision=3):
    """Convert (H,W) numpy array to nested list; sentinel where mask==0."""
    out = []
    for i in range(grid_2d.shape[0]):
        row = []
        for j in range(grid_2d.shape[1]):
            if mask_2d[i, j] == 0:
                row.append(sentinel)
            else:
                v = float(grid_2d[i, j])
                row.append(round(v, precision))
        out.append(row)
    return out

def build_unet_input(t_today):
    """Build normalized, zeroed (H, W, Cin) input for a given time index."""
    window_frames = frames[t_today - WINDOW + 1 : t_today + 1].copy()  # (7, H, W, C)
    window_norm   = (window_frames - mean[None, None, None, :]) / std[None, None, None, :]
    for day in range(WINDOW):
        t = t_today - WINDOW + 1 + day
        m = valid_mask[t]
        window_norm[day][m == 0] = 0.0
    return window_norm.transpose(1, 2, 0, 3).reshape(H, W, WINDOW * C)

dates_payload = {}

for ts in showcase_dates:
    date_str = str(ts.date())
    t_today  = date_to_t[ts]
    mask_t   = valid_mask[t_today]   # (H, W)
    label_t  = labels[t_today]        # (H, W)

    # Model prediction
    X_in      = build_unet_input(t_today)[None, ...].astype(np.float32)
    prob_map  = unet.predict(X_in, verbose=0)[0, :, :, 0]

    # Raw FWI and temperature (convert t2m from K to °C if needed)
    fwi_grid = frames[t_today, :, :, fwi_idx] if fwi_idx is not None else np.zeros((H, W))
    t2m_raw  = frames[t_today, :, :, t2m_idx] if t2m_idx is not None else np.zeros((H, W))
    t2m_grid = t2m_raw - 273.15 if t2m_raw.mean() > 100 else t2m_raw  # auto K→°C

    # Stats (valid cells only)
    n_valid = int(mask_t.sum())
    n_fires = int(label_t[mask_t == 1].sum())
    fwi_mean = round(float(fwi_grid[mask_t == 1].mean()), 1) if n_valid else 0
    t2m_mean = round(float(t2m_grid[mask_t == 1].mean()), 1) if n_valid else 0

    # Encode grids: label uses -1 for unobserved (categorical), others use SENTINEL
    label_export = np.where(mask_t == 1, label_t, -1.0)

    dates_payload[date_str] = {
        "n":         n_valid,
        "fires":     n_fires,
        "fwi_mean":  fwi_mean,
        "t2m_mean":  t2m_mean,
        "prob_grid":  grid_to_list(prob_map,  mask_t, precision=3),
        "label_grid": grid_to_list(label_export, mask_t, sentinel=-1, precision=0),
        "fwi_grid":   grid_to_list(fwi_grid,  mask_t, precision=1),
        "t2m_grid":   grid_to_list(t2m_grid,  mask_t, precision=1),
        "mask_grid":  grid_to_list(mask_t,    np.ones((H, W)), precision=0),
    }

    print(f"  {date_str}: {n_valid} cells, {n_fires} fires, "
          f"prob range [{prob_map[mask_t==1].min():.3f}, {prob_map[mask_t==1].max():.3f}]")

# ── 6. ASSEMBLE DATA OBJECT ──────────────────────────────────────────────────

data_obj = {
    "H":    H,
    "W":    W,
    "lats": [round(v, 3) for v in lats],
    "lons": [round(v, 3) for v in lons],
    "meta": {
        "model":    meta.get("model_type", "UNet"),
        "features": len(feature_cols),
        "window":   WINDOW,
        "threshold": round(meta.get("best_threshold", 0.5), 4),
        "val_f1":    round(meta.get("best_val_f1", 0.0), 4),
        "pr_auc":    round((meta.get("val_metrics") or {}).get("pr_auc", 0.0), 4),
    },
    "dates": dates_payload,
}

data_json = json.dumps(data_obj, separators=(",", ":"))
print(f"\nData payload: {len(data_json) / 1024:.0f} KB")

# ── 7. HTML TEMPLATE ─────────────────────────────────────────────────────────
# Clean, documented, production-quality HTML — no inline data hacks,
# all rendering logic in clearly named functions.

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wildfire UNet — Prediction Dashboard</title>
<style>
/* ── Reset & base ────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: system-ui, -apple-system, sans-serif;
  background: #0d1117;
  color: #e2e8f0;
  min-height: 100vh;
}}

/* ── Layout ──────────────────────────────────────────── */
.page-header  {{ padding: 20px 28px 14px; border-bottom: 1px solid #1e2433; }}
.controls     {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
                 padding: 12px 28px; background: #0f141c; border-bottom: 1px solid #1a2030; }}
.stats-bar    {{ display: flex; gap: 20px; flex-wrap: wrap;
                 padding: 10px 28px; background: #090d14; border-bottom: 1px solid #151a25; }}
.canvas-area  {{ padding: 20px 28px; display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start; }}
.footer-note  {{ padding: 6px 28px 14px; font-size: 11px; color: #2d3748; }}

/* ── Typography ──────────────────────────────────────── */
.page-header h1 {{ font-size: 18px; font-weight: 600; color: #f1f5f9; }}
.page-header p  {{ font-size: 12px; color: #475569; margin-top: 3px; }}

/* ── Button groups ───────────────────────────────────── */
.btn-group {{ display: flex; gap: 4px; }}
.btn {{
  padding: 5px 14px;
  border-radius: 6px;
  font-size: 12px;
  font-family: inherit;
  cursor: pointer;
  border: 1px solid #2d3448;
  background: transparent;
  color: #64748b;
  transition: all .15s;
}}
.btn:hover                {{ border-color: #475569; color: #cbd5e1; }}
.btn.active               {{ background: #1d4ed8; border-color: #2563eb; color: #fff; }}
.btn.active.layer-btn     {{ background: #1e2d3d; border-color: #334155; color: #e2e8f0; }}
.divider {{ width: 1px; height: 24px; background: #1e2433; margin: 0 4px; align-self: center; }}

/* ── Stats ───────────────────────────────────────────── */
.stat         {{ display: flex; flex-direction: column; gap: 2px; }}
.stat-label   {{ font-size: 10px; color: #334155; text-transform: uppercase; letter-spacing: .5px; }}
.stat-value   {{ font-size: 15px; font-weight: 700; }}
.c-fire {{ color: #f97316; }}
.c-fwi  {{ color: #eab308; }}
.c-temp {{ color: #60a5fa; }}
.c-muted{{ color: #94a3b8; }}

/* ── Map canvas ──────────────────────────────────────── */
.map-wrap  {{ flex: 1; min-width: 300px; }}
.map-title {{ font-size: 11px; color: #475569; text-transform: uppercase;
               letter-spacing: .4px; margin-bottom: 8px; }}
canvas {{
  display: block;
  border-radius: 8px;
  border: 1px solid #1e2433;
  cursor: crosshair;
  width: 100%;
  height: auto;
}}

/* ── Legend ──────────────────────────────────────────── */
.legend        {{ min-width: 130px; padding-top: 22px; }}
.legend-title  {{ font-size: 10px; color: #475569; text-transform: uppercase;
                  letter-spacing: .4px; margin-bottom: 6px; }}
.gradient-row  {{ display: flex; align-items: stretch; gap: 8px; }}
.gradient-bar  {{ width: 16px; border-radius: 4px; }}
.tick-col      {{ display: flex; flex-direction: column; justify-content: space-between; }}
.tick          {{ font-size: 11px; color: #64748b; }}
.cat-list      {{ display: flex; flex-direction: column; gap: 5px; margin-top: 8px; }}
.cat-item      {{ display: flex; align-items: center; gap: 7px;
                  font-size: 11px; color: #94a3b8; }}
.swatch        {{ width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; }}

/* ── Tooltip ─────────────────────────────────────────── */
.tooltip {{
  position: fixed;
  background: #1e2433;
  border: 1px solid #334155;
  border-radius: 8px;
  padding: 10px 14px;
  font-size: 12px;
  pointer-events: none;
  display: none;
  z-index: 999;
  box-shadow: 0 8px 24px rgba(0,0,0,.6);
  min-width: 160px;
}}
.tooltip-title {{ font-weight: 600; color: #f1f5f9; margin-bottom: 6px; }}
.tooltip-row   {{ display: flex; justify-content: space-between; gap: 14px;
                  color: #64748b; margin-top: 3px; }}
.tooltip-val   {{ color: #e2e8f0; font-weight: 500; }}

/* ── Model badge ─────────────────────────────────────── */
.model-badge {{
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 12px;
  background: #0f2744; border: 1px solid #1e3a5f;
  font-size: 11px; color: #60a5fa;
  margin-left: auto;
}}
</style>
</head>
<body>

<!-- ── Header ──────────────────────────────────────────────────────── -->
<div class="page-header">
  <h1>Wildfire Prediction Dashboard — Greece</h1>
  <p>
    UNet model &nbsp;·&nbsp; 0.25° grid ({H}&times;{W} cells) &nbsp;·&nbsp;
    {WINDOW}-day rolling window &nbsp;·&nbsp; {n_features} input features &nbsp;·&nbsp;
    Masked evaluation (valid cells only)
  </p>
</div>

<!-- ── Controls ────────────────────────────────────────────────────── -->
<div class="controls">

  <!-- Date selector -->
  <div class="btn-group" id="date-btns"></div>

  <div class="divider"></div>

  <!-- Layer selector -->
  <div class="btn-group" id="layer-btns">
    <button class="btn layer-btn active" data-layer="prob">Fire probability</button>
    <button class="btn layer-btn" data-layer="label">True label</button>
    <button class="btn layer-btn" data-layer="fwi">FWI</button>
    <button class="btn layer-btn" data-layer="t2m">Temperature (°C)</button>
    <button class="btn layer-btn" data-layer="mask">Valid mask</button>
  </div>

  <!-- Model info badge -->
  <div class="model-badge" id="model-badge"></div>

</div>

<!-- ── Stats bar ────────────────────────────────────────────────────── -->
<div class="stats-bar" id="stats-bar"></div>

<!-- ── Map + legend ─────────────────────────────────────────────────── -->
<div class="canvas-area">
  <div class="map-wrap">
    <div class="map-title" id="map-title">Fire probability</div>
    <canvas id="map-canvas"></canvas>
  </div>
  <div class="legend" id="legend"></div>
</div>

<div class="footer-note">
  Grey cells = not observed on this date (outside valid mask) &nbsp;&middot;&nbsp;
  Hover any cell for coordinates and values
</div>

<!-- ── Tooltip ──────────────────────────────────────────────────────── -->
<div class="tooltip" id="tooltip"></div>

<!-- ── Data + rendering logic ───────────────────────────────────────── -->
<script>
"use strict";

// ── DATA ──────────────────────────────────────────────────────────────────
// Generated by export_heatmap.py from real model predictions.
// Do not edit manually — re-run the script to update.
const DATA = {DATA_JSON};

// ── LAYER CONFIG ──────────────────────────────────────────────────────────
const LAYERS = {{
  prob:  {{ label: "Fire probability",       grid: "prob_grid",  type: "gradient", min: 0,   max: 1,   cmap: "fire" }},
  label: {{ label: "True fire label",         grid: "label_grid", type: "categorical",
            cats: ["No fire", "Fire", "Not observed"],
            colors: ["#166534", "#dc2626", "#1e2433"] }},
  fwi:   {{ label: "Fire Weather Index",      grid: "fwi_grid",   type: "gradient", min: 0,   max: 100, cmap: "heat" }},
  t2m:   {{ label: "Temperature (°C)",        grid: "t2m_grid",   type: "gradient", min: 0,   max: 40,  cmap: "temp" }},
  mask:  {{ label: "Valid mask",              grid: "mask_grid",  type: "categorical",
            cats: ["Observed", "Not observed"],
            colors: ["#3b82f6", "#1e2433"] }},
}};

// ── COLOUR MAPS ───────────────────────────────────────────────────────────
// Each map is an array of [r,g,b] stops; values are interpolated linearly.

const CMAPS = {{
  // black → purple → red → orange → yellow → near-white  (fire probability)
  fire: [[0,0,0], [80,0,80], [180,0,0], [220,80,0], [240,180,0], [255,255,200]],
  // blue → cyan → green → yellow → orange → dark-red     (FWI)
  heat: [[0,0,128], [0,180,180], [0,160,0], [200,200,0], [220,100,0], [180,0,0]],
  // blue → light-blue → green → yellow → orange → red    (temperature)
  temp: [[0,80,180], [0,160,220], [0,200,100], [220,220,0], [220,100,0], [160,0,0]],
}};

/**
 * Linearly interpolate a value t ∈ [0,1] through a colour map.
 * Returns [r, g, b] as floats.
 */
function sampleCmap(cmap, t) {{
  const stops = CMAPS[cmap];
  const n     = stops.length - 1;
  const idx   = Math.max(0, Math.min(n - 0.001, t * n));
  const i     = Math.floor(idx);
  const f     = idx - i;
  return stops[i].map((v, k) => v + (stops[i + 1][k] - v) * f);
}}

// ── CANVAS SETUP ──────────────────────────────────────────────────────────

const H   = DATA.H;
const W   = DATA.W;
const CV_W = 660;
const CV_H = Math.round(CV_W * H / W);

const canvas = document.getElementById("map-canvas");
const ctx    = canvas.getContext("2d");
canvas.width  = CV_W;
canvas.height = CV_H;

const cellW = CV_W / W;
const cellH = CV_H / H;

// Background colour for unobserved cells
const BG = [30, 32, 51];

// ── STATE ─────────────────────────────────────────────────────────────────

let activeDate  = Object.keys(DATA.dates)[0];
let activeLayer = "prob";

// ── RENDER ────────────────────────────────────────────────────────────────

/**
 * Main render loop — called whenever date or layer changes.
 * Writes directly to ImageData for performance.
 */
function render() {{
  const cfg       = LAYERS[activeLayer];
  const dateData  = DATA.dates[activeDate];
  const grid      = dateData[cfg.grid];
  const maskGrid  = dateData.mask_grid;
  const imgData   = ctx.createImageData(CV_W, CV_H);
  const pixels    = imgData.data;

  for (let row = 0; row < H; row++) {{
    // Flip row so latitude increases upward (south at bottom)
    const srcRow = H - 1 - row;

    for (let col = 0; col < W; col++) {{
      const val  = grid[srcRow][col];
      const mask = maskGrid[srcRow][col];

      let r, g, b;

      if (cfg.type === "gradient") {{
        if (mask === 1 && val !== -999) {{
          const t = Math.max(0, Math.min(1, (val - cfg.min) / (cfg.max - cfg.min)));
          [r, g, b] = sampleCmap(cfg.cmap, t);
        }} else {{
          [r, g, b] = BG;
        }}
      }} else {{
        // Categorical
        if (activeLayer === "mask") {{
          [r, g, b] = mask === 1
            ? hexToRgb(cfg.colors[0])
            : hexToRgb(cfg.colors[1]);
        }} else {{
          // label: -1 = not observed, 0 = no fire, 1 = fire
          if (val === 1)       [r, g, b] = hexToRgb(cfg.colors[1]);
          else if (val === 0)  [r, g, b] = hexToRgb(cfg.colors[0]);
          else                 [r, g, b] = hexToRgb(cfg.colors[2]);
        }}
      }}

      // Write all pixels in this cell block
      const px0 = Math.round(col * cellW);
      const py0 = Math.round(row * cellH);
      const pw  = Math.round((col + 1) * cellW) - px0;
      const ph  = Math.round((row + 1) * cellH) - py0;

      for (let dy = 0; dy < ph; dy++) {{
        for (let dx = 0; dx < pw; dx++) {{
          const base = 4 * ((py0 + dy) * CV_W + (px0 + dx));
          pixels[base]     = Math.round(r);
          pixels[base + 1] = Math.round(g);
          pixels[base + 2] = Math.round(b);
          pixels[base + 3] = 255;
        }}
      }}
    }}
  }}

  ctx.putImageData(imgData, 0, 0);

  // Subtle grid lines
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth   = 0.5;
  for (let c = 0; c <= W; c++) {{
    ctx.beginPath(); ctx.moveTo(c * cellW, 0); ctx.lineTo(c * cellW, CV_H); ctx.stroke();
  }}
  for (let r = 0; r <= H; r++) {{
    ctx.beginPath(); ctx.moveTo(0, r * cellH); ctx.lineTo(CV_W, r * cellH); ctx.stroke();
  }}

  updateMapTitle();
  updateLegend();
  updateStats();
}}

// ── UI UPDATES ────────────────────────────────────────────────────────────

function updateMapTitle() {{
  document.getElementById("map-title").textContent = LAYERS[activeLayer].label;
}}

function updateStats() {{
  const d = DATA.dates[activeDate];
  const fireRate = d.n > 0 ? (d.fires / d.n * 100).toFixed(1) : "0.0";

  const statsEl = document.getElementById("stats-bar");
  statsEl.innerHTML = `
    <div class="stat">
      <span class="stat-label">Date</span>
      <span class="stat-value c-muted">${{activeDate}}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Valid cells</span>
      <span class="stat-value c-muted">${{d.n}}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Fire cells</span>
      <span class="stat-value c-fire">${{d.fires}}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Fire rate</span>
      <span class="stat-value c-fire">${{fireRate}}%</span>
    </div>
    <div class="stat">
      <span class="stat-label">Mean FWI</span>
      <span class="stat-value c-fwi">${{d.fwi_mean}}</span>
    </div>
    <div class="stat">
      <span class="stat-label">Mean temp</span>
      <span class="stat-value c-temp">${{d.t2m_mean}}&thinsp;&deg;C</span>
    </div>
  `;
}}

function updateLegend() {{
  const cfg = LAYERS[activeLayer];
  const el  = document.getElementById("legend");

  if (cfg.type === "categorical") {{
    el.innerHTML = `
      <div class="legend-title">${{cfg.label}}</div>
      <div class="cat-list">
        ${{cfg.cats.map((name, i) =>
          `<div class="cat-item">
             <div class="swatch" style="background:${{cfg.colors[i]}}"></div>
             ${{name}}
           </div>`
        ).join("")}}
      </div>`;
  }} else {{
    const barImg = buildGradientBar(cfg.cmap, 16, 140);
    const mid    = Math.round((cfg.min + cfg.max) / 2);
    el.innerHTML = `
      <div class="legend-title">${{cfg.label}}</div>
      <div class="gradient-row">
        <img src="${{barImg}}" class="gradient-bar" style="height:140px">
        <div class="tick-col" style="height:140px">
          <span class="tick">${{cfg.max}}</span>
          <span class="tick">${{mid}}</span>
          <span class="tick">${{cfg.min}}</span>
        </div>
      </div>
      <div class="cat-list" style="margin-top:10px">
        <div class="cat-item">
          <div class="swatch" style="background:#1e2433"></div>Not observed
        </div>
      </div>`;
  }}
}}

/**
 * Render a vertical gradient bar to a tiny off-screen canvas
 * and return it as a data URL for use in an <img> element.
 */
function buildGradientBar(cmap, width, height) {{
  const offscreen = document.createElement("canvas");
  offscreen.width  = width;
  offscreen.height = height;
  const octx = offscreen.getContext("2d");
  for (let y = 0; y < height; y++) {{
    const t   = 1 - y / height;  // top = max
    const [r, g, b] = sampleCmap(cmap, t);
    octx.fillStyle = `rgb(${{Math.round(r)}},${{Math.round(g)}},${{Math.round(b)}})`;
    octx.fillRect(0, y, width, 1);
  }}
  return offscreen.toDataURL();
}}

function buildModelBadge() {{
  const m  = DATA.meta;
  document.getElementById("model-badge").innerHTML =
    `${{m.model}} &nbsp;&middot;&nbsp; threshold ${{m.threshold}} &nbsp;&middot;&nbsp; val F1 ${{m.val_f1}} &nbsp;&middot;&nbsp; val PR-AUC ${{m.pr_auc}}`;
}}

// ── BUTTON WIRING ─────────────────────────────────────────────────────────

function buildDateButtons() {{
  const container = document.getElementById("date-btns");
  Object.keys(DATA.dates).forEach(dateStr => {{
    const btn = document.createElement("button");
    btn.className = "btn" + (dateStr === activeDate ? " active" : "");
    btn.textContent = dateStr;
    btn.addEventListener("click", () => {{
      activeDate = dateStr;
      container.querySelectorAll(".btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      render();
    }});
    container.appendChild(btn);
  }});
}}

function wireLayerButtons() {{
  document.getElementById("layer-btns").querySelectorAll(".btn").forEach(btn => {{
    btn.addEventListener("click", () => {{
      activeLayer = btn.dataset.layer;
      document.getElementById("layer-btns")
              .querySelectorAll(".btn")
              .forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      render();
    }});
  }});
}}

// ── TOOLTIP ───────────────────────────────────────────────────────────────

const tooltip = document.getElementById("tooltip");

canvas.addEventListener("mousemove", e => {{
  const rect = canvas.getBoundingClientRect();
  const scaleX = CV_W / rect.width;
  const scaleY = CV_H / rect.height;
  const px     = (e.clientX - rect.left) * scaleX;
  const py     = (e.clientY - rect.top)  * scaleY;

  const col    = Math.floor(px / cellW);
  const row    = Math.floor(py / cellH);

  if (col < 0 || col >= W || row < 0 || row >= H) {{
    tooltip.style.display = "none";
    return;
  }}

  const srcRow = H - 1 - row;  // flip back
  const d      = DATA.dates[activeDate];
  const mask   = d.mask_grid[srcRow][col];
  const lat    = DATA.lats[srcRow].toFixed(2);
  const lon    = DATA.lons[col].toFixed(2);

  if (mask === 0) {{
    tooltip.innerHTML = `
      <div class="tooltip-title">${{lat}}&deg;N, ${{lon}}&deg;E</div>
      <div class="tooltip-row"><span>Status</span><span class="tooltip-val">Not observed</span></div>`;
  }} else {{
    const lv  = d.label_grid[srcRow][col];
    const pv  = d.prob_grid[srcRow][col];
    const fv  = d.fwi_grid[srcRow][col];
    const tv  = d.t2m_grid[srcRow][col];
    const fireColor = lv === 1 ? "#f97316" : "#4ade80";
    const fireLabel = lv === 1 ? "FIRE" : "no fire";

    tooltip.innerHTML = `
      <div class="tooltip-title">${{lat}}&deg;N, ${{lon}}&deg;E</div>
      <div class="tooltip-row">
        <span>Fire label</span>
        <span class="tooltip-val" style="color:${{fireColor}}">${{fireLabel}}</span>
      </div>
      <div class="tooltip-row">
        <span>Probability</span>
        <span class="tooltip-val">${{pv !== -999 ? (pv * 100).toFixed(1) + "%" : "—"}}</span>
      </div>
      <div class="tooltip-row">
        <span>FWI</span>
        <span class="tooltip-val">${{fv !== -999 ? fv.toFixed(1) : "—"}}</span>
      </div>
      <div class="tooltip-row">
        <span>Temp</span>
        <span class="tooltip-val">${{tv !== -999 ? tv.toFixed(1) + " °C" : "—"}}</span>
      </div>`;
  }}

  tooltip.style.display = "block";
  tooltip.style.left    = (e.clientX + 16) + "px";
  tooltip.style.top     = (e.clientY - 10) + "px";
}});

canvas.addEventListener("mouseleave", () => {{
  tooltip.style.display = "none";
}});

// ── HELPERS ───────────────────────────────────────────────────────────────

/** Parse a CSS hex colour string to [r, g, b]. */
function hexToRgb(hex) {{
  const v = parseInt(hex.replace("#", ""), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}}

// ── INIT ──────────────────────────────────────────────────────────────────

buildDateButtons();
wireLayerButtons();
buildModelBadge();
render();

</script>
</body>
</html>
"""

# ── 8. WRITE HTML ─────────────────────────────────────────────────────────

html = HTML_TEMPLATE.format(
    H          = H,
    W          = W,
    WINDOW     = WINDOW,
    n_features = len(feature_cols),
    DATA_JSON  = data_json,
)

with open(args.out, "w", encoding="utf-8") as f:
    f.write(html)

size_kb = os.path.getsize(args.out) / 1024
print(f"\nDone. Dashboard saved to: {args.out}  ({size_kb:.0f} KB)")
print(f"Open it in any browser — no server required.")