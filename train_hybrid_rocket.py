#%% ============================================================
# TRAIN v3: Hybrid MiniRocket (streaming) for CELL-FIRE (CSV)
# - Time split on target days (no leakage)
# - Train-only median impute with NaN fallback for SPEI
# - PRECOMPUTE spatial features for speed:
#     * neighbor mean for meteorology/features
#     * fire_neighbor_lag1 (from labels[t-1])
# - REMOVE per-sample zscore to preserve absolute intensity
# - Streaming: MiniRocket -> StandardScaler(partial_fit) -> SGDClassifier(partial_fit)
# - Epoch training + early stopping on sampled VAL PR-AUC
# - Calibration (sigmoid) on VAL subset
# - Threshold selection on VAL (Recall>=target, max precision), evaluate TEST
# ============================================================

import json
import copy
import numpy as np
import pandas as pd
import joblib
from scipy.ndimage import uniform_filter

from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

from sktime.transformations.panel.rocket import MiniRocket

SEED = 42
rng = np.random.default_rng(SEED)

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "training_dataset_final_prepared.csv"

WINDOW = 9
LAT_BIN_DEG = 0.25
LON_BIN_DEG = 0.25

FEATURE_COLS_BASE = [
    "t2m", "d2m", "u10", "v10", "tp", "fwi",
    "KBDI", "PET", "D", "SPEI_30", "SPEI_90", "SPEI_180"
]

# Recommended settings
ADD_WIND_SPEED = True
ADD_NEIGHBOR_MEAN = True
NEIGHBOR_SIZE = 3

ADD_FIRE_NEIGH_LAG1 = True   # assumes y[t-1] available at inference

# Sampling
NEG_RATIO_TRAIN = 50
MAX_NEG_PER_DAY_TRAIN = None  # will default to n_cells
NEG_RATIO_VALSAMP = 50
MAX_NEG_PER_DAY_VALSAMP = 2000
VALSAMP_MAX_INSTANCES = 200_000

# Rocket / streaming
NUM_KERNELS = 2000
BATCH_SIZE = 25000
ROCKET_FIT_INSTANCES = 50000
TRAIN_MAX_INSTANCES_PER_EPOCH = 300_000

# Split on target days
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15

# Epoch training
MAX_EPOCHS = 20
PATIENCE = 3
SGD_ALPHA = 1e-4

# Calibration
CAL_MAX_INSTANCES = 200_000

# Threshold objective
TARGET_RECALL_ON_VAL = 0.95

# Output
MODEL_PATH  = "minirocket_cell_model_v3.joblib"
CONFIG_PATH = "minirocket_cell_config_v3.json"

# -----------------------------
# Helpers
# -----------------------------
def safe_auc(y_true, y_score, fn):
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(fn(y_true, y_score))

def panelize(X_w_c):
    """(N, WINDOW, C) -> (N, C, WINDOW)"""
    return np.transpose(X_w_c, (0, 2, 1))

def day_range(name, days, dates):
    print(
        name,
        "from", pd.to_datetime(dates[days[0]]).date(),
        "to",   pd.to_datetime(dates[days[-1]]).date(),
        "| n_days=", len(days)
    )

def threshold_table(y, prob, thrs=(0.001,0.002,0.005,0.01,0.02,0.03,0.05,0.1,0.2,0.5,0.7,0.9)):
    print("\nthr |  Prec   Rec    F1   |   TP     FP    FN     TN | AlertRate")
    for thr in thrs:
        pred = (prob >= thr).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        prec = precision_score(y, pred, zero_division=0)
        rec  = recall_score(y, pred, zero_division=0)
        f1   = f1_score(y, pred, zero_division=0)
        alert_rate = (tp + fp) / len(y)
        print(f"{thr:>5.3f} | {prec:6.3f} {rec:6.3f} {f1:6.3f} | {tp:5d} {fp:6d} {fn:5d} {tn:6d} | {alert_rate:8.4f}")

def pick_threshold_for_recall(y, prob, target_recall=0.95):
    thrs = np.linspace(0.001, 0.999, 400)
    best = None
    for thr in thrs:
        pred = (prob >= thr).astype(np.uint8)
        rec = recall_score(y, pred, zero_division=0)
        if rec >= target_recall:
            prec = precision_score(y, pred, zero_division=0)
            f1 = f1_score(y, pred, zero_division=0)
            cand = (float(thr), float(prec), float(rec), float(f1))
            if best is None or cand[1] > best[1]:
                best = cand
    return best

def neighbor_mean_hw_c(frame_hwc, size=3):
    """Mean filter over HxW for each channel, channel axis untouched."""
    return uniform_filter(frame_hwc, size=(size, size, 1), mode="nearest")

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])

for c in FEATURE_COLS_BASE + ["lat", "lon", "y_fire", "date"]:
    if c not in df.columns:
        raise ValueError(f"Missing column in CSV: {c}")

FEATURE_COLS = FEATURE_COLS_BASE.copy()
if ADD_WIND_SPEED:
    df["wind_speed"] = np.sqrt(df["u10"].astype(float)**2 + df["v10"].astype(float)**2)
    FEATURE_COLS.append("wind_speed")

# -----------------------------
# Spatial binning (FLOOR for stability)
# -----------------------------
lat_min = float(df["lat"].min())
lon_min = float(df["lon"].min())

df["lat_bin"] = np.floor((df["lat"] - lat_min) / LAT_BIN_DEG).astype(int)
df["lon_bin"] = np.floor((df["lon"] - lon_min) / LON_BIN_DEG).astype(int)

lat_bins_sorted = np.sort(df["lat_bin"].unique())
lon_bins_sorted = np.sort(df["lon_bin"].unique())
latbin_to_i = {b: i for i, b in enumerate(lat_bins_sorted)}
lonbin_to_j = {b: j for j, b in enumerate(lon_bins_sorted)}

df["i"] = df["lat_bin"].map(latbin_to_i).astype(int)
df["j"] = df["lon_bin"].map(lonbin_to_j).astype(int)

H = len(lat_bins_sorted)
W = len(lon_bins_sorted)
n_cells = H * W
print(f"Grid HxW = {H} x {W}  (cells={n_cells})")

if MAX_NEG_PER_DAY_TRAIN is None:
    MAX_NEG_PER_DAY_TRAIN = n_cells

# -----------------------------
# Build day index
# -----------------------------
dates = np.sort(df["date"].unique())
T_total = len(dates)
date_to_t = {d: idx for idx, d in enumerate(dates)}
df["t"] = df["date"].map(date_to_t).astype(int)

# -----------------------------
# Split on TARGET days
# -----------------------------
target_days = np.arange(WINDOW - 1, T_total)
N_days = len(target_days)

n_train_days = int(TRAIN_RATIO * N_days)
n_val_days   = int((TRAIN_RATIO + VAL_RATIO) * N_days)

train_days = target_days[:n_train_days]
val_days   = target_days[n_train_days:n_val_days]
test_days  = target_days[n_val_days:]

print("Days:", "train", len(train_days), "val", len(val_days), "test", len(test_days))
day_range("TRAIN", train_days, dates)
day_range("VAL",   val_days,   dates)
day_range("TEST",  test_days,  dates)

# -----------------------------
# Train-only median impute + robust NaN fallback
# -----------------------------
train_last_t = int(train_days[-1])
train_medians = df[df["t"] <= train_last_t][FEATURE_COLS].median(numeric_only=True)
train_medians = train_medians.fillna(0.0)  # SPEI fallback if train has all-NaN early on

df[FEATURE_COLS] = df[FEATURE_COLS].fillna(train_medians)
df["y_fire"] = df["y_fire"].fillna(0).astype(np.uint8)

# -----------------------------
# Build base tensors
# -----------------------------
C0 = len(FEATURE_COLS)
frames_base = np.zeros((T_total, H, W, C0), dtype=np.float32)
labels = np.zeros((T_total, H, W), dtype=np.uint8)

t_idx = df["t"].to_numpy()
i_idx = df["i"].to_numpy()
j_idx = df["j"].to_numpy()

frames_base[t_idx, i_idx, j_idx, :] = df[FEATURE_COLS].to_numpy(dtype=np.float32)
labels[t_idx, i_idx, j_idx] = df["y_fire"].to_numpy(dtype=np.uint8)

print("frames_base:", frames_base.shape, "labels:", labels.shape)
print("Pixel pos rate overall:", float(labels.mean()))

# -----------------------------
# PRECOMPUTE augmented frames for speed
# frames_aug[t,h,w,channels] where channels include:
#   - base features
#   - (optional) neighbor mean of base features
#   - (optional) fire_neighbor_lag1
# -----------------------------
parts = [frames_base]

if ADD_NEIGHBOR_MEAN:
    neigh = np.empty_like(frames_base)
    for t in range(T_total):
        neigh[t] = neighbor_mean_hw_c(frames_base[t], size=NEIGHBOR_SIZE)
    parts.append(neigh)

fire_neigh_lag1 = None
if ADD_FIRE_NEIGH_LAG1:
    fire_neigh_lag1 = np.zeros((T_total, H, W), dtype=np.float32)
    for t in range(1, T_total):
        fire_neigh_lag1[t] = uniform_filter(
            labels[t-1].astype(np.float32),
            size=NEIGHBOR_SIZE,
            mode="constant",
            cval=0.0
        )
    parts.append(fire_neigh_lag1[..., np.newaxis])

frames_aug = np.concatenate(parts, axis=-1).astype(np.float32)
C_aug = frames_aug.shape[-1]
print(f"frames_aug: {frames_aug.shape} (C_aug={C_aug})")

# -----------------------------
# Generators (NO per-sample zscore; preserve absolute)
# -----------------------------
def iter_day_samples_sampled(days, neg_ratio, max_neg_per_day, batch_size=25000, shuffle_days=False, max_total_instances=None):
    days = np.array(days, copy=True)
    if shuffle_days:
        rng.shuffle(days)

    X_buf, y_buf = [], []
    buf_n = 0
    total = 0

    for t in days:
        seq = frames_aug[int(t) - WINDOW + 1 : int(t) + 1]   # (WINDOW,H,W,C_aug)
        seq = seq.reshape(WINDOW, n_cells, C_aug)
        X_day = np.transpose(seq, (1, 0, 2))                 # (cells,WINDOW,C_aug)
        y_day = labels[int(t)].reshape(n_cells).astype(np.uint8)

        pos_idx = np.where(y_day == 1)[0]
        neg_idx = np.where(y_day == 0)[0]

        if len(pos_idx) == 0:
            keep_neg = min(len(neg_idx), max_neg_per_day or 2000)
            if keep_neg == 0:
                continue
            idx = rng.choice(neg_idx, size=keep_neg, replace=False)
        else:
            keep_neg = min(len(neg_idx), int(neg_ratio * len(pos_idx)))
            if max_neg_per_day is not None:
                keep_neg = min(keep_neg, max_neg_per_day)
            neg_keep = rng.choice(neg_idx, size=keep_neg, replace=False) if keep_neg > 0 else np.array([], dtype=int)
            idx = np.concatenate([pos_idx, neg_keep]) if neg_keep.size else pos_idx

        rng.shuffle(idx)

        X_sel = X_day[idx]
        y_sel = y_day[idx]

        X_sel = panelize(X_sel)

        X_buf.append(X_sel)
        y_buf.append(y_sel)
        buf_n += X_sel.shape[0]
        total += X_sel.shape[0]

        if buf_n >= batch_size:
            yield np.concatenate(X_buf, axis=0), np.concatenate(y_buf, axis=0)
            X_buf, y_buf, buf_n = [], [], 0

        if max_total_instances is not None and total >= max_total_instances:
            break

    if buf_n > 0:
        yield np.concatenate(X_buf, axis=0), np.concatenate(y_buf, axis=0)

def iter_day_samples_all(days, batch_size=25000):
    days = np.array(days, copy=False)
    X_buf, y_buf = [], []
    buf_n = 0

    for t in days:
        seq = frames_aug[int(t) - WINDOW + 1 : int(t) + 1]
        seq = seq.reshape(WINDOW, n_cells, C_aug)
        X_day = np.transpose(seq, (1, 0, 2))
        y_day = labels[int(t)].reshape(n_cells).astype(np.uint8)

        X_day = panelize(X_day)

        X_buf.append(X_day)
        y_buf.append(y_day)
        buf_n += X_day.shape[0]

        if buf_n >= batch_size:
            yield np.concatenate(X_buf, axis=0), np.concatenate(y_buf, axis=0)
            X_buf, y_buf, buf_n = [], [], 0

    if buf_n > 0:
        yield np.concatenate(X_buf, axis=0), np.concatenate(y_buf, axis=0)

# -----------------------------
# 1) Fit MiniRocket on subset
# -----------------------------
print("\nFitting MiniRocket on a subset...")
rocket = MiniRocket(num_kernels=NUM_KERNELS, random_state=SEED)

X_fit_list, y_fit_list = [], []
fit_n = 0
for Xb, yb in iter_day_samples_sampled(
        train_days,
        neg_ratio=5, max_neg_per_day=MAX_NEG_PER_DAY_TRAIN,
        batch_size=BATCH_SIZE, shuffle_days=True):
    X_fit_list.append(Xb); y_fit_list.append(yb)
    fit_n += Xb.shape[0]
    if fit_n >= ROCKET_FIT_INSTANCES:
        break

X_fit = np.concatenate(X_fit_list, axis=0)[:ROCKET_FIT_INSTANCES]
y_fit = np.concatenate(y_fit_list, axis=0)[:ROCKET_FIT_INSTANCES]

print("Rocket fit instances:", X_fit.shape[0], "panel shape:", X_fit.shape)
rocket.fit(X_fit, y_fit)

# -----------------------------
# 2) Train SGD with epochs + early stopping (sampled VAL PR-AUC)
# -----------------------------
scaler = StandardScaler(with_mean=False)
clf = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=SGD_ALPHA,
    learning_rate="optimal",
    average=True,
    max_iter=1,
    tol=None,
    class_weight=None,   # sampling + thresholding used instead
    random_state=SEED,
)
classes = np.array([0, 1], dtype=np.uint8)

def sampled_val_pr_auc():
    y_all, p_all = [], []
    for Xb, yb in iter_day_samples_sampled(
            val_days,
            neg_ratio=NEG_RATIO_VALSAMP, max_neg_per_day=MAX_NEG_PER_DAY_VALSAMP,
            batch_size=BATCH_SIZE, shuffle_days=False,
            max_total_instances=VALSAMP_MAX_INSTANCES):
        Xr = rocket.transform(Xb)
        Xs = scaler.transform(Xr)
        prob = clf.predict_proba(Xs)[:, 1]
        y_all.append(yb); p_all.append(prob)
    y = np.concatenate(y_all); prob = np.concatenate(p_all)
    return float(average_precision_score(y, prob))

print("\nTraining with epochs + early stopping...")
best_pr = -1.0
pat = 0
best_state = None

for epoch in range(MAX_EPOCHS):
    seen = 0
    for Xb, yb in iter_day_samples_sampled(
            train_days,
            neg_ratio=NEG_RATIO_TRAIN, max_neg_per_day=MAX_NEG_PER_DAY_TRAIN,
            batch_size=BATCH_SIZE, shuffle_days=True,
            max_total_instances=TRAIN_MAX_INSTANCES_PER_EPOCH):  # <-- ADD THIS
        Xr = rocket.transform(Xb)
        scaler.partial_fit(Xr)
        Xs = scaler.transform(Xr)
        clf.partial_fit(Xs, yb, classes=classes)
        seen += Xb.shape[0]

    pr = sampled_val_pr_auc()
    print(f"Epoch {epoch+1:02d}/{MAX_EPOCHS}: trained={seen} | sampled VAL PR-AUC={pr:.4f}")

    if pr > best_pr + 1e-4:
        best_pr = pr
        pat = 0
        best_state = {"clf": copy.deepcopy(clf), "scaler": copy.deepcopy(scaler)}
    else:
        pat += 1
        if pat >= PATIENCE:
            print(f"Early stop: no improvement for {PATIENCE} epochs. Best sampled VAL PR-AUC={best_pr:.4f}")
            break

if best_state is not None:
    clf = best_state["clf"]
    scaler = best_state["scaler"]

# -----------------------------
# 3) Calibrate on VAL subset (sigmoid)
# -----------------------------
print("\nCollecting VAL features for calibration...")

def collect_features_for_calibration(days, max_instances=None):
    X_list, y_list = [], []
    total = 0
    for Xb, yb in iter_day_samples_all(days, batch_size=BATCH_SIZE):
        Xr = rocket.transform(Xb)
        Xs = scaler.transform(Xr)
        X_list.append(Xs); y_list.append(yb)
        total += Xs.shape[0]
        if max_instances is not None and total >= max_instances:
            break
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    if max_instances is not None and X.shape[0] > max_instances:
        X = X[:max_instances]
        y = y[:max_instances]
    return X, y

X_val_cal, y_val_cal = collect_features_for_calibration(val_days, max_instances=CAL_MAX_INSTANCES)
print("Calibration instances:", X_val_cal.shape[0], "features:", X_val_cal.shape[1])

cal = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
cal.fit(X_val_cal, y_val_cal)
print("✅ Calibrator fitted (sigmoid).")

# -----------------------------
# 4) FULL eval + threshold selection on VAL (calibrated)
# -----------------------------
def eval_full(name, days, thr=0.5):
    y_all, p_all = [], []
    for Xb, yb in iter_day_samples_all(days, batch_size=BATCH_SIZE):
        Xr = rocket.transform(Xb)
        Xs = scaler.transform(Xr)
        prob = cal.predict_proba(Xs)[:, 1]
        y_all.append(yb); p_all.append(prob)

    y = np.concatenate(y_all)
    prob = np.concatenate(p_all)
    pred = (prob >= thr).astype(np.uint8)

    pr  = safe_auc(y, prob, average_precision_score)
    roc = safe_auc(y, prob, roc_auc_score)

    cm = confusion_matrix(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec  = recall_score(y, pred, zero_division=0)
    f1   = f1_score(y, pred, zero_division=0)

    print(f"\n--- {name} (FULL eval, calibrated) @ thr={thr:.3f} ---")
    print("PR-AUC :", pr)
    print("ROC-AUC:", roc)
    print("Confusion [[TN FP],[FN TP]]:\n", cm)
    print("Precision:", prec, "Recall:", rec, "F1:", f1)

    return y, prob

y_val, prob_val = eval_full("VAL", val_days, thr=0.5)

print("\nVAL calibrated prob stats:",
      "min", float(prob_val.min()),
      "p1",  float(np.percentile(prob_val, 1)),
      "p50", float(np.percentile(prob_val, 50)),
      "p99", float(np.percentile(prob_val, 99)),
      "max", float(prob_val.max()))

threshold_table(y_val, prob_val)

best = pick_threshold_for_recall(y_val, prob_val, target_recall=TARGET_RECALL_ON_VAL)
if best is None:
    CHOSEN_THR = 0.5
    print(f"\nNo threshold achieved Recall>={TARGET_RECALL_ON_VAL:.2f} on VAL. Using thr=0.5.")
else:
    thr_r, prec_r, rec_r, f1_r = best
    CHOSEN_THR = thr_r
    print(f"\nChosen threshold on VAL (Recall>={TARGET_RECALL_ON_VAL:.2f}, max precision): "
          f"thr={thr_r:.3f} (P={prec_r:.4f}, R={rec_r:.4f}, F1={f1_r:.4f})")

_ = eval_full("TEST", test_days, thr=CHOSEN_THR)

# -----------------------------
# Save model + config
# -----------------------------
bundle = {
    "rocket": rocket,
    "scaler": scaler,
    "clf": clf,
    "calibrator": cal,
    "chosen_threshold": float(CHOSEN_THR),
}

joblib.dump(bundle, MODEL_PATH)

cfg = {
    "type": "MiniRocket(streaming)+StandardScaler+SGD+Calibrated(sigmoid) | absolute intensity preserved",
    "SEED": SEED,
    "WINDOW": int(WINDOW),
    "LAT_BIN_DEG": float(LAT_BIN_DEG),
    "LON_BIN_DEG": float(LON_BIN_DEG),
    "lat_min_train": float(lat_min),
    "lon_min_train": float(lon_min),
    "H": int(H),
    "W": int(W),
    "feature_cols": FEATURE_COLS,
    "add_wind_speed": bool(ADD_WIND_SPEED),
    "add_neighbor_mean": bool(ADD_NEIGHBOR_MEAN),
    "neighbor_size": int(NEIGHBOR_SIZE) if ADD_NEIGHBOR_MEAN else None,
    "add_fire_neighbor_lag1": bool(ADD_FIRE_NEIGH_LAG1),
    "num_kernels": int(NUM_KERNELS),
    "batch_size": int(BATCH_SIZE),
    "rocket_fit_instances": int(ROCKET_FIT_INSTANCES),
    "train_sampling": {"neg_ratio": int(NEG_RATIO_TRAIN), "max_neg_per_day": int(MAX_NEG_PER_DAY_TRAIN)},
    "max_epochs": int(MAX_EPOCHS),
    "patience": int(PATIENCE),
    "sgd_alpha": float(SGD_ALPHA),
    "calibration": {"method": "sigmoid", "max_instances": int(CAL_MAX_INSTANCES)},
    "target_recall_on_val": float(TARGET_RECALL_ON_VAL),
    "chosen_threshold": float(CHOSEN_THR),
    "split_days": {
        "train_start": str(pd.to_datetime(dates[train_days[0]]).date()),
        "train_end":   str(pd.to_datetime(dates[train_days[-1]]).date()),
        "val_start":   str(pd.to_datetime(dates[val_days[0]]).date()),
        "val_end":     str(pd.to_datetime(dates[val_days[-1]]).date()),
        "test_start":  str(pd.to_datetime(dates[test_days[0]]).date()),
        "test_end":    str(pd.to_datetime(dates[test_days[-1]]).date()),
    },
}

with open(CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)

print("\n✅ Saved:")
print(" -", MODEL_PATH)
print(" -", CONFIG_PATH)
# %%
