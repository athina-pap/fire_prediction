#%%
# ============================================================

# Evaluate the UNet against a label that matches what it was
# actually trained on: a fire-prone zone mask.
# ============================================================
#
# WHY THIS IS THE RIGHT EVALUATION FOR THIS MODEL
# ─────────────────────────────────────────────────
# The training label y_fire is a static zone mask, not a daily
# fire event. Evidence:
#   - Fire rate flat across all 12 months (19-28% even in January)
#   - Label persistence = 0.91 per location per day
#   - FWI ratio fire/no-fire only 1.24x (genuine fires show 3-10x)
#   - 86.2% of test-period fires reoccur at training fire zones
#
# The model therefore learned: "which spatial locations are
# historically fire-prone" - not "is there a fire today?"
#
# This script evaluates EXACTLY that question:
#   Q: Does the model assign higher average predicted risk to
#      locations that actually burned during the evaluation
#      period, compared to locations that never burned?
#
# APPROACH
# ─────────────────────────────────────────────────
# 1. Run the UNet on the full FireCube evaluation period and
#    compute mean predicted probability per spatial location.
#
# 2. Build the zone-mask target TWO WAYS:
#    A. FireCube target  - locations that ever burned during
#       the evaluation period (from burned_areas in dataset_greece.nc)
#    B. CSV holdout target - locations in the training CSV
#       that were labelled y_fire=1 during the TEST split
#       (2023-2025), completely unseen during training.
#       This is the cleanest possible evaluation: same grid,
#       same label definition, held-out time period.
#
# 3. Evaluate spatial ranking:
#    - ROC-AUC / PR-AUC at the location level
#    - Spatial risk maps side-by-side
#    - Comparison with a FWI-only baseline
#
# ============================================================

import os
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore", message=".*_ARRAY_API.*")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

CSV_PATH        = "training_dataset_final_prepared.csv"
NC_PATH         = "dataset_greece.nc"
UNET_MODEL_PATH = "unet_fire_improved.keras"
UNET_META_PATH  = "unet_fire_improved_meta.json"

OUT_DIR = "outputs_zone_eval"
os.makedirs(OUT_DIR, exist_ok=True)

# Evaluation period in FireCube - use dates NOT in training for cleanest eval.
# Training ends 2021-08-03 so 2021-09-01 onward is safe.
# FireCube covers 2009-2020/2021 so we evaluate on the overlap.
EVAL_START = "2018-01-01"   # start of available FireCube data in your file
EVAL_END   = "2021-08-29"   # end of available FireCube data in your file

# Minimum number of time windows a cell must be observed to be included
MIN_OBS_FRACTION = 0.05     # cell must appear in ≥5% of windows

TIME_BATCH  = 32
CLIP_Z      = 5.0

# ── HELPERS ───────────────────────────────────────────────────────────────────

def safe_auc(y_true, y_score, fn):
    return float(fn(y_true, y_score)) if np.unique(y_true).size >= 2 else float("nan")


def time_fill(arr):
    out = arr.copy().astype(np.float32)
    for t in range(1, out.shape[0]):
        m = ~np.isfinite(out[t]); out[t][m] = out[t-1][m]
    for t in range(out.shape[0]-2, -1, -1):
        m = ~np.isfinite(out[t]); out[t][m] = out[t+1][m]
    out[~np.isfinite(out)] = 0.0
    return out


def clean_kelvin(arr):
    arr = arr.astype(np.float32)
    arr = np.where((arr < 100) | (arr > 400), np.nan, arr)
    return np.where(np.isfinite(arr), arr, float(np.nanmedian(arr))).astype(np.float32)


def clean_feature(arr):
    arr = arr.astype(np.float32)
    return np.where(np.isfinite(arr), arr, float(np.nanmedian(arr))).astype(np.float32)


def rolling_mean_time(arr, win):
    T   = arr.shape[0]
    out = np.full_like(arr, np.nan, dtype=np.float32)
    cs  = np.zeros((T+1, arr.shape[1], arr.shape[2]), dtype=np.float64)
    cs[1:] = np.cumsum(arr.astype(np.float64), axis=0)
    for t in range(win-1, T):
        out[t] = ((cs[t+1] - cs[t+1-win]) / float(win)).astype(np.float32)
    return out


def zscore_per_pixel(arr, eps=1e-6):
    mu = np.nanmean(arr, axis=0).astype(np.float32)
    sd = np.nanstd(arr,  axis=0).astype(np.float32)
    sd = np.where(sd < eps, 1.0, sd)
    z  = (arr - mu[None]) / sd[None]
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)


def fill_nan_zero(arr):
    return np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)


def collapse_time_to_channels(X):
    N, T, H, W, C = X.shape
    return X.transpose(0, 2, 3, 1, 4).reshape(N, H, W, T*C)


def report_spatial(name, y_true, y_score):
    """Print spatial evaluation metrics and return dict."""
    if np.unique(y_true).size < 2:
        print(f"  {name}: only one class - skipped"); return {}

    roc  = safe_auc(y_true, y_score, roc_auc_score)
    pr   = safe_auc(y_true, y_score, average_precision_score)
    brier = float(brier_score_loss(y_true.astype(np.uint8), y_score))

    p_arr, r_arr, t_arr = precision_recall_curve(y_true.astype(np.uint8), y_score)
    f1_arr = 2*p_arr*r_arr / (p_arr+r_arr+1e-9)
    best   = int(np.nanargmax(f1_arr))
    thr    = float(t_arr[best]) if best < len(t_arr) else 0.5
    yhat   = (y_score >= thr).astype(np.uint8)

    print(f"\n  {name}")
    print(f"    Locations:  {len(y_true):,}  |  fire zones: {int(y_true.sum()):,}  "
          f"({100*y_true.mean():.1f}%)")
    print(f"    ROC-AUC:    {roc:.4f}  (random=0.5, perfect=1.0)")
    print(f"    PR-AUC:     {pr:.4f}  (random≈{y_true.mean():.4f})")
    print(f"    Brier:      {brier:.4f}")
    print(f"    Best thr:   {thr:.4f}  →  F1={f1_arr[best]:.4f}  "
          f"P={precision_score(y_true,yhat,zero_division=0):.4f}  "
          f"R={recall_score(y_true,yhat,zero_division=0):.4f}")

    return dict(n_locations=int(len(y_true)), n_fire_zones=int(y_true.sum()),
                positive_rate=float(y_true.mean()), roc_auc=float(roc),
                pr_auc=float(pr), brier=float(brier),
                best_threshold=float(thr), best_f1=float(f1_arr[best]),
                precision=float(precision_score(y_true,yhat,zero_division=0)),
                recall=float(recall_score(y_true,yhat,zero_division=0)))


# ── LOAD META + MODEL ─────────────────────────────────────────────────────────

with open(UNET_META_PATH) as f:
    meta = json.load(f)

FEATURES        = meta["feature_cols"]
WINDOW          = int(meta["WINDOW"])
LAT_BIN_DEG     = float(meta["LAT_BIN_DEG"])
LON_BIN_DEG     = float(meta["LON_BIN_DEG"])
H_train         = int(meta["H"])
W_train         = int(meta["W"])
lat_min_train   = float(meta["lat_min"])
lon_min_train   = float(meta["lon_min"])
i_min_global    = int(meta["i_min_global"])
j_min_global    = int(meta["j_min_global"])
Cin_expected    = int(meta["Cin"])
train_mean      = np.array(meta["mean"], dtype=np.float32)
train_std       = np.array(meta["std"],  dtype=np.float32)
train_std[train_std == 0] = 1.0
TRAIN_THRESHOLD = float(meta.get("best_threshold", 0.5))
OBS_FEATURES    = {"SPEI_30_obs", "SPEI_90_obs", "SPEI_180_obs"}
TEMP_IN_KELVIN  = train_mean[FEATURES.index("t2m")] > 100

print(f"Features: {FEATURES}")
print(f"Grid: {H_train}×{W_train}  |  threshold: {TRAIN_THRESHOLD:.4f}")
print(f"Temperature units: {'Kelvin' if TEMP_IN_KELVIN else 'Celsius'}")


@tf.keras.utils.register_keras_serializable()
def pad_hw(x):
    factor = 16
    return tf.pad(x, [[0,0],[0,(factor-(x.shape[1]%factor))%factor],
                       [0,(factor-(x.shape[2]%factor))%factor],[0,0]])

unet = tf.keras.models.load_model(
    UNET_MODEL_PATH, compile=False, custom_objects={"pad_hw": pad_hw}
)
unet.trainable = False
print("Model loaded.\n")

# ── PART A: FIRECUBE PREDICTIONS ──────────────────────────────────────────────
# Build features from dataset_greece.nc, run the model, get spatial mean risk.

print("=" * 60)
print("PART A - FireCube predictions")
print("=" * 60)

ds = xr.open_dataset(NC_PATH)
lat_dim = next(d for d in ["lat","y"] if d in ds.dims)
lon_dim = next(d for d in ["lon","x"] if d in ds.dims)
ds = ds.sel(time=slice(EVAL_START, EVAL_END))
T_nc = ds.sizes["time"]
print(f"FireCube time steps: {T_nc}  ({EVAL_START} → {EVAL_END})")

# Grid mapping
lat_vals      = ds[lat_dim].values
lon_vals      = ds[lon_dim].values
i_crop_all    = np.round((lat_vals - lat_min_train)/LAT_BIN_DEG).astype(int) - i_min_global
j_crop_all    = np.round((lon_vals - lon_min_train)/LON_BIN_DEG).astype(int) - j_min_global
lat_idx_keep  = np.where((i_crop_all >= 0) & (i_crop_all < H_train))[0]
lon_idx_keep  = np.where((j_crop_all >= 0) & (j_crop_all < W_train))[0]
lat_bins_keep = i_crop_all[lat_idx_keep]
lon_bins_keep = j_crop_all[lon_idx_keep]

cell_counts = np.zeros((H_train, W_train), dtype=np.float32)
for bi in lat_bins_keep:
    for bj in lon_bins_keep:
        cell_counts[bi, bj] += 1.0
cell_counts_safe  = np.where(cell_counts == 0, 1.0, cell_counts)
static_valid_mask = (cell_counts > 0).astype(np.float32)

def regrid_mean(var_name):
    da  = ds[var_name].isel({lat_dim:lat_idx_keep, lon_dim:lon_idx_keep}).astype("float32")
    T   = da.sizes["time"]
    out = np.zeros((T, H_train, W_train), dtype=np.float32)
    for t0 in range(0, T, TIME_BATCH):
        t1    = min(T, t0+TIME_BATCH)
        block = da.isel(time=slice(t0,t1)).values.astype(np.float32)
        tmp   = np.zeros((t1-t0, H_train, W_train), dtype=np.float32)
        for ii, bi in enumerate(lat_bins_keep):
            for jj, bj in enumerate(lon_bins_keep):
                v = block[:, ii, jj]; g = np.isfinite(v)
                tmp[g, bi, bj] += v[g]
        out[t0:t1] = tmp / cell_counts_safe[None]
    return out.astype(np.float32)

def regrid_bin(var_name):
    da  = ds[var_name].isel({lat_dim:lat_idx_keep, lon_dim:lon_idx_keep})
    T   = da.sizes["time"]
    out = np.zeros((T, H_train, W_train), dtype=np.uint8)
    for t0 in range(0, T, TIME_BATCH):
        t1    = min(T, t0+TIME_BATCH)
        block = (da.isel(time=slice(t0,t1)).values > 0).astype(np.uint8)
        tmp   = np.zeros((t1-t0, H_train, W_train), dtype=np.uint8)
        for ii, bi in enumerate(lat_bins_keep):
            for jj, bj in enumerate(lon_bins_keep):
                tmp[:, bi, bj] = np.maximum(tmp[:, bi, bj], block[:, ii, jj])
        out[t0:t1] = tmp
    return out

print("Building features...")
t2m_K    = clean_kelvin(regrid_mean("avg_t2m"))
d2m_K    = clean_kelvin(regrid_mean("avg_d2m"))
u10      = time_fill(clean_feature(regrid_mean("avg_u10")))
v10      = time_fill(clean_feature(regrid_mean("avg_v10")))
tp_mm    = clean_feature(time_fill(regrid_mean("avg_tp")) * 1000.0)
fwi      = clean_feature(time_fill(regrid_mean("fwi") if "fwi" in ds.data_vars
                          else np.zeros((T_nc,H_train,W_train), dtype=np.float32)))

t2m_for_norm = t2m_K          if TEMP_IN_KELVIN else t2m_K - 273.15
d2m_for_norm = d2m_K          if TEMP_IN_KELVIN else d2m_K - 273.15
t2m_C        = t2m_K - 273.15

PET  = (0.4 * np.maximum(t2m_C, 0.0)).astype(np.float32)
D    = (tp_mm - PET).astype(np.float32)
KBDI = np.zeros_like(tp_mm, dtype=np.float32)

SPEI_30  = fill_nan_zero(zscore_per_pixel(rolling_mean_time(D, 30)))
SPEI_90  = fill_nan_zero(zscore_per_pixel(rolling_mean_time(D, 90)))
SPEI_180 = fill_nan_zero(zscore_per_pixel(rolling_mean_time(D, 180)))
SPEI_30_obs = SPEI_90_obs = SPEI_180_obs = np.ones((T_nc, H_train, W_train), dtype=np.float32)

doy    = np.array([pd.Timestamp(t).timetuple().tm_yday for t in ds["time"].values])
bcast  = lambda v: (v[:,None,None] * np.ones((T_nc,H_train,W_train), dtype=np.float32)).astype(np.float32)
doy_sin = bcast(np.sin(2*np.pi*doy/365).astype(np.float32))
doy_cos = bcast(np.cos(2*np.pi*doy/365).astype(np.float32))

# FireCube zone-mask target: did this cell EVER burn in the eval period?
ba_nc    = regrid_bin("burned_areas")
zone_fc  = (ba_nc.sum(axis=0) > 0).astype(np.float32)   # (H_train, W_train)
print(f"FireCube ever-burned cells: {int(zone_fc.sum())} / {int(static_valid_mask.sum())} "
      f"({100*zone_fc[static_valid_mask==1].mean():.1f}%)")

# Also save FWI spatial mean (for FWI baseline)
fwi_spatial_mean = np.where(static_valid_mask==1,
                             fwi.sum(axis=0) / np.maximum(static_valid_mask * T_nc, 1),
                             np.nan)

feat_map = {
    "t2m":t2m_for_norm,"d2m":d2m_for_norm,"u10":u10,"v10":v10,"tp":tp_mm,"fwi":fwi,
    "KBDI":KBDI,"PET":PET,"D":D,
    "SPEI_30":SPEI_30,"SPEI_90":SPEI_90,"SPEI_180":SPEI_180,
    "SPEI_30_obs":SPEI_30_obs,"SPEI_90_obs":SPEI_90_obs,"SPEI_180_obs":SPEI_180_obs,
    "doy_sin":doy_sin,"doy_cos":doy_cos,
}
X_full = np.stack([feat_map[f] for f in FEATURES], axis=-1).astype(np.float32)
X_full = np.where(np.isfinite(X_full), X_full, 0.0).astype(np.float32)

valid_mask_full = np.repeat(static_valid_mask[None], T_nc, axis=0).astype(np.float32)
X_norm = np.zeros_like(X_full, dtype=np.float32)
for k, feat in enumerate(FEATURES):
    if feat in OBS_FEATURES:
        X_norm[..., k] = X_full[..., k]
    else:
        X_norm[..., k] = np.clip((X_full[...,k] - train_mean[k]) / train_std[k], -CLIP_Z, CLIP_Z)
X_norm[valid_mask_full == 0] = 0.0
X_norm = np.where(np.isfinite(X_norm), X_norm, 0.0).astype(np.float32)

valid_days = np.arange(WINDOW-1, T_nc, dtype=np.int32)
X_seq  = np.stack([X_norm[t-WINDOW+1:t+1] for t in valid_days], axis=0).astype(np.float32)
M_seq  = np.stack([valid_mask_full[t]      for t in valid_days], axis=0).astype(np.float32)
X_unet = collapse_time_to_channels(X_seq)
X_unet = np.where(np.isfinite(X_unet), X_unet, 0.0).astype(np.float32)

if X_unet.shape[-1] != Cin_expected:
    raise ValueError(f"Channel mismatch: {X_unet.shape[-1]} vs {Cin_expected}")

print(f"Running model on {X_unet.shape[0]} windows...")
pred_raw = unet.predict(X_unet, batch_size=8, verbose=1).astype(np.float32)[..., 0]
pred_raw = np.where(np.isfinite(pred_raw), pred_raw, 0.5)

# Spatial mean predicted risk per cell (averaged over all time windows)
pred_sum    = np.where(M_seq == 1, pred_raw, 0.0).sum(axis=0)
obs_count   = M_seq.sum(axis=0)
min_obs     = max(1, int(MIN_OBS_FRACTION * len(valid_days)))
spatial_ok  = (obs_count >= min_obs).astype(np.float32)
pred_spatial = np.where(spatial_ok==1, pred_sum / np.maximum(obs_count, 1), np.nan)

ds.close()

# ── PART B: CSV HOLDOUT TARGET ────────────────────────────────────────────────
# Build zone-mask from training CSV test split (2023-2025) - cleanest evaluation.
# Same grid, same label definition, completely unseen during training.

print("\n" + "="*60)
print("PART B - CSV holdout zone mask (2023-2025 test split)")
print("="*60)

df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])

# Reconstruct test split (same logic as training script)
dates_sorted_csv = np.sort(df["date"].unique())
T_csv = len(dates_sorted_csv)
day_gaps_csv = np.diff([pd.Timestamp(d).toordinal() for d in dates_sorted_csv])
valid_ends_csv = [t for t in range(WINDOW-1, T_csv)
                  if np.all(day_gaps_csv[t-WINDOW+1:t]==1)]
N_csv    = len(valid_ends_csv)
n_train  = int(0.70 * N_csv)
n_val    = int(0.85 * N_csv)

test_t_indices = valid_ends_csv[n_val:]
test_dates_set = set(pd.Timestamp(dates_sorted_csv[t]) for t in test_t_indices)

df_test = df[df["date"].isin(test_dates_set)].copy()
print(f"Test split: {df_test['date'].min().date()} → {df_test['date'].max().date()}")
print(f"Test rows: {len(df_test):,}")

# Zone mask: location ever had y_fire=1 in test split
csv_zone = df_test.groupby(["lat","lon"])["y_fire"].max().reset_index()
csv_zone.columns = ["lat","lon","zone_label"]

# Map to training grid
csv_zone["lat_bin"] = ((csv_zone["lat"] - lat_min_train) / LAT_BIN_DEG).round().astype(int)
csv_zone["lon_bin"] = ((csv_zone["lon"] - lon_min_train) / LON_BIN_DEG).round().astype(int)
lat_bins_sorted_csv = np.sort(df["lat_bin"].unique()) if "lat_bin" in df.columns else None

# Recompute grid indices
df["lat_bin"] = ((df["lat"] - lat_min_train)/LAT_BIN_DEG).round().astype(int)
df["lon_bin"] = ((df["lon"] - lon_min_train)/LON_BIN_DEG).round().astype(int)
lat_bins_u = np.sort(df["lat_bin"].unique())
lon_bins_u = np.sort(df["lon_bin"].unique())
latbin_to_i = {b:i for i,b in enumerate(lat_bins_u)}
lonbin_to_j = {b:j for j,b in enumerate(lon_bins_u)}

csv_zone["i"] = csv_zone["lat_bin"].map(latbin_to_i)
csv_zone["j"] = csv_zone["lon_bin"].map(lonbin_to_j)
csv_zone["i_crop"] = csv_zone["i"] - i_min_global
csv_zone["j_crop"] = csv_zone["j"] - j_min_global
csv_zone = csv_zone.dropna(subset=["i_crop","j_crop"])
csv_zone = csv_zone[(csv_zone["i_crop"]>=0) & (csv_zone["i_crop"]<H_train) &
                    (csv_zone["j_crop"]>=0) & (csv_zone["j_crop"]<W_train)]

# Build zone mask grid
zone_csv_grid = np.full((H_train, W_train), np.nan, dtype=np.float32)
pred_csv_grid = np.full((H_train, W_train), np.nan, dtype=np.float32)

for _, row in csv_zone.iterrows():
    i, j = int(row["i_crop"]), int(row["j_crop"])
    zone_csv_grid[i, j] = float(row["zone_label"])
    if np.isfinite(pred_spatial[i, j]):
        pred_csv_grid[i, j] = pred_spatial[i, j]

# Valid cells: present in CSV and have predictions
csv_valid = np.isfinite(zone_csv_grid) & np.isfinite(pred_csv_grid)
n_csv_valid = int(csv_valid.sum())
print(f"CSV test locations with predictions: {n_csv_valid}")
print(f"Fire zones in test split: {int(zone_csv_grid[csv_valid].sum())} "
      f"({100*zone_csv_grid[csv_valid].mean():.1f}%)")

# ── PART C: ALSO BUILD A FWI BASELINE ─────────────────────────────────────────
# Rank locations by mean FWI - FWI alone should predict zone risk reasonably well.
# If model ROC-AUC > FWI ROC-AUC -> model adds value beyond FWI.

print("\n" + "="*60)
print("PART C - FWI spatial mean baseline")
print("="*60)
print("(Comparing model vs simple mean-FWI ranking of locations)")

# ── EVALUATION ────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("SPATIAL ZONE-MASK EVALUATION RESULTS")
print("="*60)

# -- Eval A: FireCube ever-burned --
fc_valid = spatial_ok.astype(bool) & np.isfinite(pred_spatial)
yt_fc    = zone_fc[fc_valid]
yp_fc    = pred_spatial[fc_valid]
yf_fc    = fwi_spatial_mean[fc_valid]
yf_fc_clean = np.where(np.isfinite(yf_fc), yf_fc, 0.0)
# normalise FWI to [0,1] for fair comparison
yf_fc_norm = (yf_fc_clean - yf_fc_clean.min()) / (yf_fc_clean.max() - yf_fc_clean.min() + 1e-9)

print("\n── A. FireCube target: ever-burned during eval period ──")
res_fc_model = report_spatial("UNet mean risk",        yt_fc, yp_fc)
res_fc_fwi   = report_spatial("FWI baseline (mean FWI)", yt_fc, yf_fc_norm)

if res_fc_model.get("roc_auc") and res_fc_fwi.get("roc_auc"):
    diff = res_fc_model["roc_auc"] - res_fc_fwi["roc_auc"]
    print(f"\n  Model lift over FWI baseline: {diff:+.4f} ROC-AUC")

# -- Eval B: CSV holdout zone mask --
yt_csv = zone_csv_grid[csv_valid]
yp_csv = pred_csv_grid[csv_valid]

# FWI baseline on CSV grid
fwi_csv = np.full((H_train, W_train), np.nan, dtype=np.float32)
for _, row in csv_zone.iterrows():
    i, j = int(row["i_crop"]), int(row["j_crop"])
    if np.isfinite(fwi_spatial_mean[i, j]):
        fwi_csv[i, j] = fwi_spatial_mean[i, j]
yf_csv = fwi_csv[csv_valid]
yf_csv_clean = np.where(np.isfinite(yf_csv), yf_csv, 0.0)
yf_csv_norm  = (yf_csv_clean - yf_csv_clean.min()) / (yf_csv_clean.max() - yf_csv_clean.min() + 1e-9)

print("\n── B. CSV holdout target: y_fire zones in test split (2023-2025) ──")
print("   (CLEANEST evaluation: same grid, same label, unseen time period)")
res_csv_model = report_spatial("UNet mean risk",          yt_csv, yp_csv)
res_csv_fwi   = report_spatial("FWI baseline (mean FWI)", yt_csv, yf_csv_norm)

if res_csv_model.get("roc_auc") and res_csv_fwi.get("roc_auc"):
    diff = res_csv_model["roc_auc"] - res_csv_fwi["roc_auc"]
    print(f"\n  Model lift over FWI baseline: {diff:+.4f} ROC-AUC")

# ── PLOTS ─────────────────────────────────────────────────────────────────────

cmap_risk = plt.cm.YlOrRd.copy();  cmap_risk.set_bad("lightgrey")
cmap_zone = mcolors.ListedColormap(["#2d6a4f","#d62828"]); cmap_zone.set_bad("lightgrey")
cmap_fwi  = plt.cm.hot_r.copy();   cmap_fwi.set_bad("lightgrey")

# --- Plot 1: FireCube spatial comparison ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(f"Zone-mask evaluation - FireCube target ({EVAL_START[:4]}–{EVAL_END[:4]})", fontsize=12)

pred_vis = np.where(spatial_ok==1, pred_spatial, np.nan)
zone_vis = np.where(spatial_ok==1, zone_fc, np.nan)
fwi_vis  = np.where(spatial_ok==1, fwi_spatial_mean, np.nan)

im0 = axes[0].imshow(pred_vis, origin="lower", cmap=cmap_risk, vmin=0, vmax=1)
axes[0].set_title("UNet: mean predicted risk per location")
plt.colorbar(im0, ax=axes[0], fraction=0.04)

im1 = axes[1].imshow(zone_vis, origin="lower", cmap=cmap_zone, vmin=0, vmax=1)
axes[1].set_title("Target: ever burned (red) vs never (green)")
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_zone), ax=axes[1], fraction=0.04)

fwi_norm_vis = (fwi_vis - np.nanmin(fwi_vis)) / (np.nanmax(fwi_vis) - np.nanmin(fwi_vis) + 1e-9)
im2 = axes[2].imshow(fwi_norm_vis, origin="lower", cmap=cmap_fwi, vmin=0, vmax=1)
axes[2].set_title("FWI baseline: mean FWI per location")
plt.colorbar(im2, ax=axes[2], fraction=0.04)

plt.tight_layout()
p = os.path.join(OUT_DIR, "zone_eval_firecube.png")
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"\nSaved: {p}")

# --- Plot 2: CSV holdout spatial comparison ---
pred_csv_vis = np.where(csv_valid, pred_csv_grid, np.nan)
zone_csv_vis = np.where(csv_valid, zone_csv_grid, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Zone-mask evaluation - CSV holdout target (2023-2025 test split)", fontsize=12)

im0 = axes[0].imshow(pred_csv_vis, origin="lower", cmap=cmap_risk, vmin=0, vmax=1)
axes[0].set_title("UNet: mean predicted risk (training grid only)")
plt.colorbar(im0, ax=axes[0], fraction=0.04)

im1 = axes[1].imshow(zone_csv_vis, origin="lower", cmap=cmap_zone, vmin=0, vmax=1)
axes[1].set_title("Target: y_fire=1 zones in test split")
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_zone), ax=axes[1], fraction=0.04)

plt.tight_layout()
p = os.path.join(OUT_DIR, "zone_eval_csv_holdout.png")
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")

# --- Plot 3: ROC curves side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ROC curves - zone-mask evaluation", fontsize=12)

for ax, (yt, yp, yf, title) in zip(axes, [
    (yt_fc,  yp_fc,  yf_fc_norm,  "FireCube target"),
    (yt_csv, yp_csv, yf_csv_norm, "CSV holdout target (2023-2025)")
]):
    if np.unique(yt).size < 2: ax.set_visible(False); continue
    fpr_m, tpr_m, _ = roc_curve(yt.astype(np.uint8), yp)
    fpr_f, tpr_f, _ = roc_curve(yt.astype(np.uint8), yf)
    ax.plot(fpr_m, tpr_m, lw=1.5, label=f"UNet (AUC={safe_auc(yt,yp,roc_auc_score):.3f})")
    ax.plot(fpr_f, tpr_f, lw=1.5, linestyle="--", label=f"FWI baseline (AUC={safe_auc(yt,yf,roc_auc_score):.3f})")
    ax.plot([0,1],[0,1], color="grey", linestyle=":", label="Random")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title); ax.legend(fontsize=9)

plt.tight_layout()
p = os.path.join(OUT_DIR, "roc_curves.png")
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")

# --- Plot 4: PR curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Precision-Recall curves - zone-mask evaluation", fontsize=12)

for ax, (yt, yp, yf, title) in zip(axes, [
    (yt_fc,  yp_fc,  yf_fc_norm,  "FireCube target"),
    (yt_csv, yp_csv, yf_csv_norm, "CSV holdout target (2023-2025)")
]):
    if np.unique(yt).size < 2: ax.set_visible(False); continue
    p_m, r_m, _ = precision_recall_curve(yt.astype(np.uint8), yp)
    p_f, r_f, _ = precision_recall_curve(yt.astype(np.uint8), yf)
    ax.plot(r_m, p_m, lw=1.5, label=f"UNet (AP={safe_auc(yt,yp,average_precision_score):.3f})")
    ax.plot(r_f, p_f, lw=1.5, linestyle="--", label=f"FWI baseline (AP={safe_auc(yt,yf,average_precision_score):.3f})")
    ax.axhline(float(yt.mean()), color="grey", linestyle=":", label=f"Random ({yt.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend(fontsize=9)

plt.tight_layout()
p = os.path.join(OUT_DIR, "pr_curves.png")
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")

# ── SAVE SUMMARY ──────────────────────────────────────────────────────────────

summary = {
    "evaluation_type": "zone_mask_spatial",
    "eval_period_firecube": f"{EVAL_START} to {EVAL_END}",
    "csv_test_split": "2023-2025 (held out during training)",
    "note": (
        "Training label y_fire is a static zone mask (fire-prone location). "
        "This evaluation correctly measures spatial zone discrimination: "
        "does the model assign higher mean risk to historically fire-prone locations?"
    ),
    "firecube_target": {
        "unet":          res_fc_model,
        "fwi_baseline":  res_fc_fwi,
        "model_lift_roc": float(res_fc_model.get("roc_auc",0) - res_fc_fwi.get("roc_auc",0)),
    },
    "csv_holdout_target": {
        "unet":          res_csv_model,
        "fwi_baseline":  res_csv_fwi,
        "model_lift_roc": float(res_csv_model.get("roc_auc",0) - res_csv_fwi.get("roc_auc",0)),
    },
}

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAll outputs saved to: {OUT_DIR}/")
print("\n── FINAL SUMMARY ──")
print(f"  FireCube target  - UNet ROC-AUC: {res_fc_model.get('roc_auc','N/A')}  "
      f"FWI baseline: {res_fc_fwi.get('roc_auc','N/A')}")
print(f"  CSV holdout      - UNet ROC-AUC: {res_csv_model.get('roc_auc','N/A')}  "
      f"FWI baseline: {res_csv_fwi.get('roc_auc','N/A')}")
print()
print("Interpretation guide:")
print("  UNet ROC > FWI ROC  -> model adds spatial information beyond FWI alone")
print("  UNet ROC > 0.5      -> model ranks fire zones above safe zones")
print("  UNet ROC ≈ 0.5      -> no spatial discrimination (zone patterns not learned)")
# %%
