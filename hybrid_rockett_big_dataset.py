#%%
# ============================================================
# EVAL SAVED Hybrid MiniRocket model on FireCube (4-year subset)
# MEMORY-SAFE STREAMING VERSION
# ============================================================

from __future__ import annotations

import os
import json
import joblib
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# -----------------------------
# PATHS
# -----------------------------
NC_PATH = "dataset_greece.nc"

MODEL_PATH = "minirocket_cell_model_v3.joblib"
CONFIG_PATH = "minirocket_cell_config_v3.json"

OUT_DIR = "outputs_minirocket_firecube_4y"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# SUBSET / TARGET
# -----------------------------
START_DATE = "2018-01-01"
END_DATE   = "2021-08-29"

USE_MONTH_FILTER = True
MONTHS_KEEP = {5, 6, 7, 8, 9}

TARGET_PREF = "ignition_points"   # fallback burned_areas

# -----------------------------
# MEMORY
# -----------------------------
TIME_BATCH = 32
PRED_BATCH = 20000   # lower if needed

# -----------------------------
# HELPERS
# -----------------------------
def safe_auc(y_true, y_score, fn):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(fn(y_true, y_score))

def month_from_npdt64(dt64):
    s = str(dt64)
    return int(s[5:7])

def panelize(X_w_c):
    """(N, WINDOW, C) -> (N, C, WINDOW)"""
    return np.transpose(X_w_c, (0, 2, 1))

def time_fill(arr):
    out = arr.copy()
    T = out.shape[0]
    for t in range(1, T):
        m = ~np.isfinite(out[t])
        out[t][m] = out[t - 1][m]
    for t in range(T - 2, -1, -1):
        m = ~np.isfinite(out[t])
        out[t][m] = out[t + 1][m]
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)

def rolling_mean_time(arr, win):
    T = arr.shape[0]
    out = np.full_like(arr, np.nan, dtype=np.float32)
    csum = np.zeros((T + 1, arr.shape[1], arr.shape[2]), dtype=np.float64)
    csum[1:] = np.cumsum(arr.astype(np.float64), axis=0)
    for t in range(win - 1, T):
        out[t] = ((csum[t + 1] - csum[t + 1 - win]) / float(win)).astype(np.float32)
    return out

def zscore_per_pixel(arr, eps=1e-6):
    mu = np.nanmean(arr, axis=0).astype(np.float32)
    sd = np.nanstd(arr, axis=0).astype(np.float32)
    sd = np.where(sd < eps, 1.0, sd).astype(np.float32)
    z = (arr - mu[None, :, :]) / sd[None, :, :]
    return np.where(np.isfinite(z), z, 0.0).astype(np.float32)

def compute_kbdi_simplified(t2m_C, rain_mm):
    T, H, W = t2m_C.shape
    kbdi = np.zeros((T, H, W), dtype=np.float32)
    prev = np.zeros((H, W), dtype=np.float32)
    for t in range(T):
        T_eff = np.maximum(t2m_C[t], 0.0).astype(np.float32)
        R = np.maximum(rain_mm[t], 0.0).astype(np.float32)
        wet = R > 5.0
        prev_wet = np.maximum(prev - 0.5 * (R - 5.0), 0.0)
        prev_dry = np.minimum(prev + 0.1 * (T_eff / 5.0), 800.0)
        prev = np.where(wet, prev_wet, prev_dry).astype(np.float32)
        kbdi[t] = prev
    return kbdi

def reliability_curve(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(np.float32).reshape(-1)
    y_prob = np.asarray(y_prob).astype(np.float32).reshape(-1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    confs, accs = [], []
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        confs.append(float(y_prob[mask].mean()))
        accs.append(float(y_true[mask].mean()))
    return np.array(confs), np.array(accs)

def plot_reliability(y_true, y_prob, out_png):
    c, a = reliability_curve(y_true, y_prob, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.plot(c, a, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title("Reliability diagram (MiniRocket)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def threshold_table(y, prob, thrs=(0.001,0.002,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.5)):
    print("\nthr |  Prec   Rec    F1   |   TP     FP    FN     TN | AlertRate")
    for thr in thrs:
        pred = (prob >= thr).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        prec = precision_score(y, pred, zero_division=0)
        rec  = recall_score(y, pred, zero_division=0)
        f1   = f1_score(y, pred, zero_division=0)
        alert_rate = (tp + fp) / len(y)
        print(f"{thr:>5.3f} | {prec:6.4f} {rec:6.4f} {f1:6.4f} | {tp:5d} {fp:6d} {fn:5d} {tn:6d} | {alert_rate:8.4f}")

# -----------------------------
# LOAD SAVED MODEL + CONFIG
# -----------------------------
bundle = joblib.load(MODEL_PATH)
rocket = bundle["rocket"]
scaler = bundle["scaler"]
clf = bundle["clf"]
cal = bundle["calibrator"]
chosen_threshold = float(bundle.get("chosen_threshold", 0.5))

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

WINDOW = int(cfg["WINDOW"])
LAT_BIN_DEG = float(cfg["LAT_BIN_DEG"])
LON_BIN_DEG = float(cfg["LON_BIN_DEG"])
lat_min_train = float(cfg["lat_min_train"])
lon_min_train = float(cfg["lon_min_train"])
H_train = int(cfg["H"])
W_train = int(cfg["W"])

FEATURE_COLS = cfg["feature_cols"]
ADD_WIND_SPEED = bool(cfg.get("add_wind_speed", False))
ADD_NEIGHBOR_MEAN = bool(cfg.get("add_neighbor_mean", False))
NEIGHBOR_SIZE = int(cfg.get("neighbor_size", 3) or 3)
ADD_FIRE_NEIGH_LAG1 = bool(cfg.get("add_fire_neighbor_lag1", False))

print("Loaded saved MiniRocket config:")
print("WINDOW:", WINDOW)
print("Grid:", H_train, W_train)
print("Chosen threshold:", chosen_threshold)

# -----------------------------
# LOAD NC
# -----------------------------
ds = xr.open_dataset(NC_PATH)

lat_dim = "lat" if "lat" in ds.dims else ("y" if "y" in ds.dims else None)
lon_dim = "lon" if "lon" in ds.dims else ("x" if "x" in ds.dims else None)
if lat_dim is None or lon_dim is None:
    raise ValueError("Expected spatial dims (lat, lon) or (y, x).")

ds = ds.sel(time=slice(START_DATE, END_DATE))

if USE_MONTH_FILTER:
    months_all = np.array([month_from_npdt64(t) for t in ds["time"].values], dtype=np.int32)
    keep_mask = np.array([m in MONTHS_KEEP for m in months_all], dtype=bool)
    ds = ds.isel(time=np.where(keep_mask)[0])

print("Subset time steps kept:", ds.sizes["time"])

target_name = TARGET_PREF if TARGET_PREF in ds.data_vars else ("burned_areas" if "burned_areas" in ds.data_vars else None)
if target_name is None:
    raise ValueError("Could not find ignition_points or burned_areas.")

# -----------------------------
# REGRID TO TRAIN GRID
# -----------------------------
lat_vals = ds[lat_dim].values
lon_vals = ds[lon_dim].values

lat_bin_all = np.floor((lat_vals - lat_min_train) / LAT_BIN_DEG).astype(int)
lon_bin_all = np.floor((lon_vals - lon_min_train) / LON_BIN_DEG).astype(int)

valid_lat_mask = (lat_bin_all >= 0) & (lat_bin_all < H_train)
valid_lon_mask = (lon_bin_all >= 0) & (lon_bin_all < W_train)

lat_idx_keep = np.where(valid_lat_mask)[0]
lon_idx_keep = np.where(valid_lon_mask)[0]

lat_bins_keep = lat_bin_all[lat_idx_keep]
lon_bins_keep = lon_bin_all[lon_idx_keep]

cell_counts = np.zeros((H_train, W_train), dtype=np.float32)
for i, bi in enumerate(lat_bins_keep):
    for j, bj in enumerate(lon_bins_keep):
        cell_counts[bi, bj] += 1.0
cell_counts_safe = np.where(cell_counts == 0, 1.0, cell_counts)

def regrid_mean_chunked(var_name, batch_t=32):
    if var_name not in ds.data_vars:
        raise ValueError(f"Missing var in NetCDF: {var_name}")

    da = ds[var_name].isel({lat_dim: lat_idx_keep, lon_dim: lon_idx_keep}).astype("float32")
    T = da.sizes["time"]
    out = np.zeros((T, H_train, W_train), dtype=np.float32)

    for t0 in range(0, T, batch_t):
        t1 = min(T, t0 + batch_t)
        block = da.isel(time=slice(t0, t1)).values.astype(np.float32)
        tmp = np.zeros((t1 - t0, H_train, W_train), dtype=np.float32)

        for i, bi in enumerate(lat_bins_keep):
            for j, bj in enumerate(lon_bins_keep):
                vals = block[:, i, j]
                good = np.isfinite(vals)
                tmp[good, bi, bj] += vals[good]

        out[t0:t1] = tmp / cell_counts_safe[None, :, :]
        print(f"{var_name}: {t0}:{t1}/{T}")

    return out.astype(np.float32)

def regrid_bin_chunked(var_name, batch_t=32):
    da = ds[var_name].isel({lat_dim: lat_idx_keep, lon_dim: lon_idx_keep})
    T = da.sizes["time"]
    out = np.zeros((T, H_train, W_train), dtype=np.uint8)

    for t0 in range(0, T, batch_t):
        t1 = min(T, t0 + batch_t)
        block = (da.isel(time=slice(t0, t1)).values > 0).astype(np.uint8)
        tmp = np.zeros((t1 - t0, H_train, W_train), dtype=np.uint8)

        for i, bi in enumerate(lat_bins_keep):
            for j, bj in enumerate(lon_bins_keep):
                tmp[:, bi, bj] = np.maximum(tmp[:, bi, bj], block[:, i, j])

        out[t0:t1] = tmp
        print(f"{var_name}: {t0}:{t1}/{T}")

    return out

# -----------------------------
# BUILD FEATURES
# -----------------------------
print("\nBuilding harmonized FireCube features for MiniRocket...")

t2m_raw = time_fill(regrid_mean_chunked("avg_t2m", batch_t=TIME_BATCH))
d2m_raw = time_fill(regrid_mean_chunked("avg_d2m", batch_t=TIME_BATCH))
u10 = time_fill(regrid_mean_chunked("avg_u10", batch_t=TIME_BATCH))
v10 = time_fill(regrid_mean_chunked("avg_v10", batch_t=TIME_BATCH))
tp_m = time_fill(regrid_mean_chunked("avg_tp", batch_t=TIME_BATCH))
fwi = time_fill(regrid_mean_chunked("fwi", batch_t=TIME_BATCH)) if "fwi" in ds.data_vars else np.zeros_like(tp_m, dtype=np.float32)

t2m = (t2m_raw - 273.15).astype(np.float32)
d2m = (d2m_raw - 273.15).astype(np.float32)

tp_mm = (tp_m * 1000.0).astype(np.float32)
PET = (0.4 * np.maximum(t2m, 0.0)).astype(np.float32)
D = (tp_mm - PET).astype(np.float32)
KBDI = compute_kbdi_simplified(t2m, tp_mm).astype(np.float32)
SPEI_30 = zscore_per_pixel(rolling_mean_time(D, 30)).astype(np.float32)
SPEI_90 = zscore_per_pixel(rolling_mean_time(D, 90)).astype(np.float32)
SPEI_180 = zscore_per_pixel(rolling_mean_time(D, 180)).astype(np.float32)

feat_map = {
    "t2m": t2m,
    "d2m": d2m,
    "u10": u10,
    "v10": v10,
    "tp": tp_mm,
    "fwi": fwi,
    "KBDI": KBDI,
    "PET": PET,
    "D": D,
    "SPEI_30": SPEI_30,
    "SPEI_90": SPEI_90,
    "SPEI_180": SPEI_180,
}

base_cols = [c for c in FEATURE_COLS if c != "wind_speed"]
frames_base = np.stack([feat_map[c] for c in base_cols], axis=-1).astype(np.float32)

if ADD_WIND_SPEED:
    wind_speed = np.sqrt(u10.astype(np.float32)**2 + v10.astype(np.float32)**2).astype(np.float32)
    frames_base = np.concatenate([frames_base, wind_speed[..., np.newaxis]], axis=-1)

labels = regrid_bin_chunked(target_name, batch_t=TIME_BATCH).astype(np.uint8)

parts = [frames_base]

if ADD_NEIGHBOR_MEAN:
    neigh = np.empty_like(frames_base)
    for t in range(frames_base.shape[0]):
        neigh[t] = uniform_filter(frames_base[t], size=(NEIGHBOR_SIZE, NEIGHBOR_SIZE, 1), mode="nearest")
    parts.append(neigh)

if ADD_FIRE_NEIGH_LAG1:
    fire_neigh_lag1 = np.zeros((labels.shape[0], H_train, W_train), dtype=np.float32)
    for t in range(1, labels.shape[0]):
        fire_neigh_lag1[t] = uniform_filter(
            labels[t - 1].astype(np.float32),
            size=NEIGHBOR_SIZE,
            mode="constant",
            cval=0.0
        )
    parts.append(fire_neigh_lag1[..., np.newaxis])

frames_aug = np.concatenate(parts, axis=-1).astype(np.float32)

print("frames_aug:", frames_aug.shape)
print("labels:", labels.shape)
print("Positive rate:", float(labels.mean()))

# -----------------------------
# STREAMING PREDICTION
# -----------------------------
print("\nRunning streaming MiniRocket evaluation...")

T_total = frames_aug.shape[0]
n_cells = H_train * W_train
target_days = np.arange(WINDOW - 1, T_total)

y_parts = []
p_parts = []

for t in target_days:
    seq = frames_aug[t - WINDOW + 1 : t + 1]  # (WINDOW,H,W,C)
    seq = seq.reshape(WINDOW, n_cells, frames_aug.shape[-1])
    X_day = np.transpose(seq, (1, 0, 2))      # (cells, WINDOW, C)
    y_day = labels[t].reshape(n_cells).astype(np.uint8)

    X_day = panelize(X_day)

    day_probs = []
    for s in range(0, X_day.shape[0], PRED_BATCH):
        e = min(s + PRED_BATCH, X_day.shape[0])
        Xb = X_day[s:e]

        Xr_b = rocket.transform(Xb)
        if hasattr(Xr_b, "to_numpy"):
            Xr_b = Xr_b.to_numpy()
        else:
            Xr_b = np.asarray(Xr_b)

        Xs_b = scaler.transform(Xr_b)
        pb = cal.predict_proba(Xs_b)[:, 1].astype(np.float32)
        day_probs.append(pb)

    prob_day = np.concatenate(day_probs, axis=0)

    y_parts.append(y_day)
    p_parts.append(prob_day)

    if (t - target_days[0] + 1) % 20 == 0 or t == target_days[-1]:
        print(f"Processed day {t} ({t - target_days[0] + 1}/{len(target_days)})")

y_all = np.concatenate(y_parts, axis=0)
prob = np.concatenate(p_parts, axis=0)

# -----------------------------
# METRICS
# -----------------------------
pr_auc = safe_auc(y_all, prob, average_precision_score)
roc_auc = safe_auc(y_all, prob, roc_auc_score)

print("\n================ FULL EVAL (MiniRocket, FireCube) ================")
print("PR-AUC :", pr_auc)
print("ROC-AUC:", roc_auc)

print("\nProbability stats:")
print("min   :", float(prob.min()))
print("p50   :", float(np.quantile(prob, 0.50)))
print("p90   :", float(np.quantile(prob, 0.90)))
print("p95   :", float(np.quantile(prob, 0.95)))
print("p99   :", float(np.quantile(prob, 0.99)))
print("p999  :", float(np.quantile(prob, 0.999)))
print("max   :", float(prob.max()))
print("positive rate:", float(np.mean(y_all)))

pred = (prob >= chosen_threshold).astype(np.uint8)
cm = confusion_matrix(y_all, pred)
prec = precision_score(y_all, pred, zero_division=0)
rec = recall_score(y_all, pred, zero_division=0)
f1 = f1_score(y_all, pred, zero_division=0)

print(f"\nSaved threshold = {chosen_threshold:.4f}")
print("Confusion matrix:\n", cm)
print("Precision:", prec)
print("Recall   :", rec)
print("F1       :", f1)

threshold_table(y_all, prob)

k = max(1, int(0.01 * len(prob)))
idx = np.argsort(-prob)[:k]
top1_prec = float(np.mean(y_all[idx]))
print("\nTop 1% precision:", top1_prec)

print("mean prob positive:", float(np.mean(prob[y_all == 1])) if np.any(y_all == 1) else float("nan"))
print("mean prob negative:", float(np.mean(prob[y_all == 0])) if np.any(y_all == 0) else float("nan"))

# -----------------------------
# SAVE
# -----------------------------
np.save(os.path.join(OUT_DIR, "y_true.npy"), y_all.astype(np.uint8))
np.save(os.path.join(OUT_DIR, "y_prob.npy"), prob.astype(np.float32))

plot_reliability(y_all, prob, os.path.join(OUT_DIR, "reliability_diagram.png"))

summary = {
    "model_path": MODEL_PATH,
    "config_path": CONFIG_PATH,
    "target_name": target_name,
    "subset": {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "months_keep": sorted(list(MONTHS_KEEP)) if USE_MONTH_FILTER else None,
    },
    "metrics": {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "chosen_threshold": float(chosen_threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "top1_precision": float(top1_prec),
        "mean_prob_positive": float(np.mean(prob[y_all == 1])) if np.any(y_all == 1) else None,
        "mean_prob_negative": float(np.mean(prob[y_all == 0])) if np.any(y_all == 0) else None,
        "cm": cm.tolist(),
    },
    "files": {
        "y_true": os.path.join(OUT_DIR, "y_true.npy"),
        "y_prob": os.path.join(OUT_DIR, "y_prob.npy"),
        "reliability_diagram": os.path.join(OUT_DIR, "reliability_diagram.png"),
    },
}

with open(os.path.join(OUT_DIR, "eval_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n✅ Saved:")
print(" -", os.path.join(OUT_DIR, "eval_summary.json"))
print(" -", os.path.join(OUT_DIR, "reliability_diagram.png"))
print(" -", os.path.join(OUT_DIR, "y_true.npy"))
print(" -", os.path.join(OUT_DIR, "y_prob.npy"))

ds.close()
print("\n✅ MiniRocket FireCube evaluation finished.")
# %%
