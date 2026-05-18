#%%
# ============================================================
# WILDFIRE PREDICTION — IMPROVED UNET PIPELINE
# ============================================================
#
# Key improvements over previous versions:
#
#   1. DATA AUDIT block — prints label semantics diagnostics
#      so you can verify y_fire means what you think it means
#      before spending GPU hours training on it.
#
#   2. GAP-AWARE WINDOWING — detects holes in the date
#      timeline (279 gaps in this dataset, up to 20 days long)
#      and skips any 7-day window that crosses a gap.
#      Previously day-1 and day-22 were treated as consecutive.
#
#   3. SPEI MISSINGNESS INDICATORS — instead of filling 41-74%
#      NaN SPEI values with the column median (which invents
#      data), we add a binary "was this observed?" flag for
#      each SPEI column, then fill with 0. The model can now
#      learn "this feature is absent" as a signal.
#
#   4. CYCLIC DAY-OF-YEAR — sin/cos encoding of day-of-year
#      gives the model an explicit seasonality signal without
#      any ordinal assumptions.
#
#   5. GLOBAL BOUNDING BOX CROP — crops the full 33x44 grid
#      to the tightest rectangle containing all valid cells.
#      Reduces wasted capacity on structural zeros.
#
#   6. MASKED FOCAL LOSS + MASKED EVALUATION — loss and ALL
#      metrics (PR-AUC, ROC-AUC, F1, confusion matrix) are
#      computed only over valid (observed) pixels. Background
#      zeros are excluded from everything.
#
#   7. CORRECT NORMALIZATION — mean/std computed from valid
#      train pixels only. Invalid cells zeroed after norm.
#
#   8. SHAP + LIME — pixel-level explainability with corrected
#      LIME sample size (500) and valid-pixel-only background.
#
# ============================================================

import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model, Input
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# CONFIG
# ============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CSV_PATH = "training_dataset_final_prepared.csv"

WINDOW       = 7
LAT_BIN_DEG  = 0.25
LON_BIN_DEG  = 0.25

TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15

LR           = 1e-3
EPOCHS       = 30
BATCH_SIZE   = 4

FOCAL_ALPHA  = 0.85
FOCAL_GAMMA  = 2.0

MODEL_PATH   = "unet_fire_improved.keras"
WEIGHTS_PATH = "unet_fire_improved.weights.h5"
META_PATH    = "unet_fire_improved_meta.json"

# SHAP / LIME
EXPLAIN_SAMPLE_INDEX = 0
EXPLAIN_TARGET_I     = 10
EXPLAIN_TARGET_J     = 15
SHAP_BG_SIZE         = 16
LIME_NUM_SAMPLES     = 500
LIME_NUM_FEATURES    = 20

# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])

base_feature_cols = [
    "t2m", "d2m", "u10", "v10", "tp", "fwi",
    "KBDI", "PET", "D", "SPEI_30", "SPEI_90", "SPEI_180",
]
base_feature_cols = [c for c in base_feature_cols if c in df.columns]

if "y_fire" not in df.columns:
    raise ValueError("Column y_fire not found.")

# ============================================================
# STAGE 1 — DATA AUDIT
# Print diagnostics to help you verify label semantics.
# Read this output before trusting any model results.
# ============================================================

print("\n" + "="*60)
print("STAGE 1 — DATA AUDIT")
print("="*60)
print(f"Rows: {len(df):,}   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"Unique dates: {df['date'].nunique():,}")
print(f"Unique (lat,lon) locations: {df.groupby(['lat','lon']).ngroups:,}")

fire_rate_overall = df["y_fire"].mean()
print(f"\ny_fire overall rate: {fire_rate_overall:.3f}  "
      f"({'WARNING: very high for active-fire labels' if fire_rate_overall > 0.15 else 'ok'})")

# Persistence: P(y_fire[t] == y_fire[t-1] | same location)
multi = df.groupby(["lat", "lon"]).filter(lambda x: len(x) > 5).sort_values(["lat", "lon", "date"])
persistence = multi.groupby(["lat", "lon"])["y_fire"].apply(
    lambda x: (x.values[1:] == x.values[:-1]).mean()
).mean()
print(f"Label persistence (same as previous day, per location): {persistence:.3f}  "
      f"({'WARNING: high — may be a static zone label, not dynamic fire' if persistence > 0.75 else 'ok'})")

print("\nFire rate by month:")
df["_month"] = df["date"].dt.month
monthly = df.groupby("_month")["y_fire"].mean()
for m, r in monthly.items():
    bar = "█" * int(r * 40)
    print(f"  {m:02d}: {r:.3f} {bar}")
df.drop(columns=["_month"], inplace=True)

print("\nKBDI stats by fire label (expected: fire=1 should have higher KBDI if it is drought-related):")
print(df.groupby("y_fire")["KBDI"].describe()[["mean","std","50%"]].round(4))

print("\nNaN rates:")
for col in base_feature_cols:
    rate = df[col].isnull().mean()
    flag = " ← HIGH" if rate > 0.3 else ""
    print(f"  {col}: {rate:.3f}{flag}")

print("\nLocations that EVER have fire:",
      df.groupby(["lat","lon"])["y_fire"].max().sum(),
      "/", df.groupby(["lat","lon"]).ngroups)
print("Locations that ALWAYS have fire:",
      df.groupby(["lat","lon"])["y_fire"].min().sum())
print("="*60 + "\n")

# ============================================================
# STAGE 2 — FEATURE ENGINEERING
# ============================================================

# 2a. SPEI missingness indicators (before filling)
spei_cols = [c for c in ["SPEI_30", "SPEI_90", "SPEI_180"] if c in df.columns]
for col in spei_cols:
    df[f"{col}_obs"] = (~df[col].isnull()).astype(np.float32)
    df[col] = df[col].fillna(0.0)  # fill with 0 AFTER flagging

# 2b. Fill remaining NaNs with median (only for non-SPEI features)
df[base_feature_cols] = df[base_feature_cols].fillna(
    df[base_feature_cols].median(numeric_only=True)
)

# 2c. Cyclic day-of-year (gives the model an explicit seasonality signal)
df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365).astype(np.float32)
df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365).astype(np.float32)

# Final feature list (order matters — saved to meta for inference)
spei_obs_cols = [f"{c}_obs" for c in spei_cols]
feature_cols  = base_feature_cols + spei_obs_cols + ["doy_sin", "doy_cos"]

df["y_fire"] = df["y_fire"].fillna(0).astype(np.float32)
C = len(feature_cols)

print(f"Features ({C}):", feature_cols)

# ============================================================
# STAGE 3 — SPATIAL GRID
# ============================================================

lat_min = df["lat"].min()
lon_min = df["lon"].min()

df["lat_bin"] = ((df["lat"] - lat_min) / LAT_BIN_DEG).round().astype(int)
df["lon_bin"] = ((df["lon"] - lon_min) / LON_BIN_DEG).round().astype(int)

lat_bins_sorted = np.sort(df["lat_bin"].unique())
lon_bins_sorted = np.sort(df["lon_bin"].unique())
latbin_to_i = {b: i for i, b in enumerate(lat_bins_sorted)}
lonbin_to_j = {b: j for j, b in enumerate(lon_bins_sorted)}  # note: j not i

H_full = len(lat_bins_sorted)
W_full = len(lon_bins_sorted)

df["i"] = df["lat_bin"].map(latbin_to_i).astype(int)
df["j"] = df["lon_bin"].map(lonbin_to_j).astype(int)

# Global bounding box crop
i_min_global = int(df["i"].min())
i_max_global = int(df["i"].max())
j_min_global = int(df["j"].min())
j_max_global = int(df["j"].max())

H = i_max_global - i_min_global + 1
W = j_max_global - j_min_global + 1

df["i_crop"] = df["i"] - i_min_global
df["j_crop"] = df["j"] - j_min_global

print(f"Full grid: {H_full}×{W_full}  →  Cropped grid: {H}×{W}")

# ============================================================
# STAGE 4 — DAILY TENSORS + VALID MASK (on cropped grid)
# ============================================================

dates_sorted = np.sort(df["date"].unique())
T_total      = len(dates_sorted)
date_to_t    = {d: idx for idx, d in enumerate(dates_sorted)}
df["t"]      = df["date"].map(date_to_t).astype(int)

frames     = np.zeros((T_total, H, W, C), dtype=np.float32)
labels     = np.zeros((T_total, H, W, 1), dtype=np.float32)
valid_mask = np.zeros((T_total, H, W, 1), dtype=np.float32)

t_arr = df["t"].to_numpy()
i_arr = df["i_crop"].to_numpy()
j_arr = df["j_crop"].to_numpy()

frames    [t_arr, i_arr, j_arr, :]  = df[feature_cols].to_numpy(dtype=np.float32)
labels    [t_arr, i_arr, j_arr, 0]  = df["y_fire"].to_numpy(dtype=np.float32)
valid_mask[t_arr, i_arr, j_arr, 0]  = 1.0

print(f"frames: {frames.shape}  valid pixel rate: {valid_mask.mean():.4f}")
print(f"Fire rate over valid pixels: {labels[valid_mask==1].mean():.4f}")

# ============================================================
# STAGE 5 — GAP-AWARE WINDOWING
# Only build windows from fully contiguous date runs.
# ============================================================

# gaps[k] = number of days between dates_sorted[k] and dates_sorted[k+1]
day_gaps = np.diff([pd.Timestamp(d).toordinal() for d in dates_sorted])

valid_window_ends = []
for t in range(WINDOW - 1, T_total):
    # The window uses time steps [t-WINDOW+1 .. t].
    # Transitions within this window: gaps[t-WINDOW+1 .. t-1] (WINDOW-1 values).
    window_gaps = day_gaps[t - WINDOW + 1 : t]
    if np.all(window_gaps == 1):
        valid_window_ends.append(t)

N_dropped = (T_total - WINDOW + 1) - len(valid_window_ends)
print(f"\nGap-aware windowing: {len(valid_window_ends)} valid windows "
      f"({N_dropped} dropped due to date gaps)")

X_seq = np.stack(
    [frames[t - WINDOW + 1 : t + 1] for t in valid_window_ends], axis=0
)  # (N, WINDOW, H, W, C)
Y_seq = np.stack(
    [labels[t]     for t in valid_window_ends], axis=0
)  # (N, H, W, 1)
M_seq = np.stack(
    [valid_mask[t] for t in valid_window_ends], axis=0
)  # (N, H, W, 1)

print(f"X_seq: {X_seq.shape}  Y_seq: {Y_seq.shape}  M_seq: {M_seq.shape}")

# ============================================================
# STAGE 6 — TIME SPLIT
# ============================================================

N      = X_seq.shape[0]
n_train = int(TRAIN_RATIO * N)
n_val   = int((TRAIN_RATIO + VAL_RATIO) * N)

X_train, Y_train, M_train = X_seq[:n_train],       Y_seq[:n_train],       M_seq[:n_train]
X_val,   Y_val,   M_val   = X_seq[n_train:n_val],  Y_seq[n_train:n_val],  M_seq[n_train:n_val]
X_test,  Y_test,  M_test  = X_seq[n_val:],         Y_seq[n_val:],         M_seq[n_val:]

print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

# ============================================================
# STAGE 7 — NORMALIZATION (valid train pixels only)
# ============================================================

# Collect the valid_mask for all time steps in each train window
valid_ends_train = valid_window_ends[:n_train]
M_train_input = np.stack(
    [valid_mask[t - WINDOW + 1 : t + 1] for t in valid_ends_train], axis=0
)  # (n_train, WINDOW, H, W, 1)

# Only use pixels that actually have data
valid_values_train = X_train[M_train_input[..., 0] == 1]  # (n_valid_pixels, C)

mean = valid_values_train.reshape(-1, C).mean(axis=0).astype(np.float32)
std  = valid_values_train.reshape(-1, C).std(axis=0).astype(np.float32)
std[std == 0] = 1.0

def normalize(X):
    return (X - mean[None, None, None, None, :]) / std[None, None, None, None, :]

X_train = normalize(X_train).astype(np.float32)
X_val   = normalize(X_val  ).astype(np.float32)
X_test  = normalize(X_test ).astype(np.float32)

def zero_invalid_inputs(X, window_ends_subset):
    """Set empty grid cells back to 0 after normalization."""
    X_out = X.copy()
    for n, t in enumerate(window_ends_subset):
        input_mask = valid_mask[t - WINDOW + 1 : t + 1]  # (WINDOW, H, W, 1)
        X_out[n][input_mask[..., 0] == 0] = 0.0
    return X_out

valid_ends_val  = valid_window_ends[n_train:n_val]
valid_ends_test = valid_window_ends[n_val:]

X_train = zero_invalid_inputs(X_train, valid_ends_train)
X_val   = zero_invalid_inputs(X_val,   valid_ends_val)
X_test  = zero_invalid_inputs(X_test,  valid_ends_test)

print("Normalization done.")

# ============================================================
# STAGE 8 — COLLAPSE TIME → CHANNELS  (N, T, H, W, C) → (N, H, W, T*C)
# ============================================================

def collapse_time_to_channels(X):
    Nn, Tt, Hh, Ww, Cc = X.shape
    return X.transpose(0, 2, 3, 1, 4).reshape(Nn, Hh, Ww, Tt * Cc)

X_train_u = collapse_time_to_channels(X_train)
X_val_u   = collapse_time_to_channels(X_val)
X_test_u  = collapse_time_to_channels(X_test)

Cin = X_train_u.shape[-1]
print(f"UNet input: (N, {H}, {W}, {Cin})  =  {WINDOW} days × {C} features")

# Concatenate label + mask for masked loss
Y_train_m = np.concatenate([Y_train, M_train], axis=-1)
Y_val_m   = np.concatenate([Y_val,   M_val  ], axis=-1)
Y_test_m  = np.concatenate([Y_test,  M_test ], axis=-1)

# ============================================================
# STAGE 9 — MASKED FOCAL LOSS
# Loss is computed only over valid (observed) grid cells.
# ============================================================

def masked_focal_loss(alpha=0.85, gamma=2.0):
    def loss(y_true_with_mask, y_pred):
        y_true = y_true_with_mask[..., 0:1]
        mask   = y_true_with_mask[..., 1:2]

        y_true = tf.cast(y_true, tf.float32)
        mask   = tf.cast(mask,   tf.float32)

        eps    = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        pt     = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        w      = tf.where(tf.equal(y_true, 1.0), alpha,  1.0 - alpha)
        focal  = -w * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        focal  = focal * mask

        return tf.reduce_sum(focal) / (tf.reduce_sum(mask) + eps)
    return loss

# ============================================================
# STAGE 10 — UNET MODEL
# Pads to next multiple of 16 internally, crops back to (H, W).
# ============================================================

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
        return tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])

    x = layers.Lambda(pad_hw, name="pad_hw")(inp)

    # Encoder
    c1 = conv_block(x,  base,     dropout, name="enc1"); p1 = layers.MaxPooling2D(name="pool1")(c1)
    c2 = conv_block(p1, base*2,   dropout, name="enc2"); p2 = layers.MaxPooling2D(name="pool2")(c2)
    c3 = conv_block(p2, base*4,   dropout, name="enc3"); p3 = layers.MaxPooling2D(name="pool3")(c3)
    c4 = conv_block(p3, base*8,   dropout, name="enc4"); p4 = layers.MaxPooling2D(name="pool4")(c4)

    # Bottleneck
    bn = conv_block(p4, base*16, dropout, name="bottleneck")

    # Decoder
    u4 = layers.UpSampling2D(name="up4")(bn);  u4 = layers.Concatenate(name="cat4")([u4, c4]); d4 = conv_block(u4, base*8,  dropout, name="dec4")
    u3 = layers.UpSampling2D(name="up3")(d4);  u3 = layers.Concatenate(name="cat3")([u3, c3]); d3 = conv_block(u3, base*4,  dropout, name="dec3")
    u2 = layers.UpSampling2D(name="up2")(d3);  u2 = layers.Concatenate(name="cat2")([u2, c2]); d2 = conv_block(u2, base*2,  dropout, name="dec2")
    u1 = layers.UpSampling2D(name="up1")(d2);  u1 = layers.Concatenate(name="cat1")([u1, c1]); d1 = conv_block(u1, base,    dropout, name="dec1")

    out = layers.Conv2D(1, 1, activation="sigmoid", name="fire_prob")(d1)

    if pad_h != 0 or pad_w != 0:
        out = layers.Cropping2D(cropping=((0, pad_h), (0, pad_w)), name="crop_back")(out)

    return Model(inp, out, name="ImprovedUNet")


unet = build_unet(H=H, W=W, Cin=Cin, base=32, dropout=0.1)
unet.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=masked_focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA),
)
unet.summary()

# ============================================================
# STAGE 11 — TRAIN
# Monitor val_loss (masked), NOT val_pr_auc.
# val_pr_auc computed on the full grid is inflated by background zeros.
# ============================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=6,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    ),
]

history = unet.fit(
    X_train_u, Y_train_m,
    validation_data=(X_val_u, Y_val_m),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)

# ============================================================
# STAGE 12 — MASKED EVALUATION
# ALL metrics computed on valid pixels only.
# ============================================================

def masked_flatten(y_true, y_prob, mask):
    """Return 1D arrays of true labels and probs for valid pixels only."""
    m = mask.reshape(-1).astype(bool)
    return y_true.reshape(-1)[m], y_prob.reshape(-1)[m]


def find_best_threshold(y_true, y_prob, mask, n_steps=300):
    yt, yp = masked_flatten(y_true, y_prob, mask)
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.001, 0.999, n_steps):
        f1 = f1_score(yt, (yp >= thr).astype(np.uint8), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


def evaluate_masked(name, y_true, y_prob, mask, threshold):
    yt, yp = masked_flatten(y_true, y_prob, mask)

    pr_auc  = average_precision_score(yt, yp)
    roc_auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else float("nan")

    ypred = (yp >= threshold).astype(np.uint8)
    cm    = confusion_matrix(yt, ypred, labels=[0, 1])
    prec  = precision_score(yt, ypred, zero_division=0)
    rec   = recall_score(yt,   ypred, zero_division=0)
    f1    = f1_score(yt,       ypred, zero_division=0)

    print(f"\n{'='*50}")
    print(f"{name}  (valid pixels: {len(yt):,}  positives: {int(yt.sum()):,})")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Precision: {prec:.4f}   Recall: {rec:.4f}   F1: {f1:.4f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")

    return {
        "pr_auc":    float(pr_auc),
        "roc_auc":   float(roc_auc) if not np.isnan(roc_auc) else None,
        "threshold": float(threshold),
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "cm":        cm.tolist(),
        "n_valid":   int(len(yt)),
        "n_pos":     int(yt.sum()),
    }


val_prob  = unet.predict(X_val_u,  batch_size=1, verbose=1)
test_prob = unet.predict(X_test_u, batch_size=1, verbose=1)

best_thr, best_val_f1 = find_best_threshold(Y_val, val_prob, M_val)
print(f"\nBest threshold on VAL (masked): {best_thr:.4f}   F1: {best_val_f1:.4f}")

val_metrics  = evaluate_masked("VAL",  Y_val,  val_prob,  M_val,  best_thr)
test_metrics = evaluate_masked("TEST", Y_test, test_prob, M_test, best_thr)

# ============================================================
# STAGE 13 — SAVE
# ============================================================

unet.save(MODEL_PATH)
unet.save_weights(WEIGHTS_PATH)

meta = {
    "model_type":          "ImprovedUNet",
    "feature_cols":        feature_cols,
    "base_feature_cols":   base_feature_cols,
    "spei_obs_cols":       spei_obs_cols,
    "mean":                mean.tolist(),
    "std":                 std.tolist(),
    "WINDOW":              int(WINDOW),
    "LAT_BIN_DEG":         float(LAT_BIN_DEG),
    "LON_BIN_DEG":         float(LON_BIN_DEG),
    "lat_min":             float(lat_min),
    "lon_min":             float(lon_min),
    "H_full":              int(H_full),
    "W_full":              int(W_full),
    "H":                   int(H),
    "W":                   int(W),
    "i_min_global":        int(i_min_global),
    "j_min_global":        int(j_min_global),
    "C":                   int(C),
    "Cin":                 int(Cin),
    "UNET_INPUT_MODE":     "collapse_time_to_channels",
    "gap_aware_windowing": True,
    "masked_loss":         True,
    "masked_evaluation":   True,
    "best_threshold":      float(best_thr),
    "best_val_f1":         float(best_val_f1),
    "val_metrics":         val_metrics,
    "test_metrics":        test_metrics,
    "inference_note": (
        "At inference: fill SPEI_*_obs=0/1, doy_sin/cos from date. "
        "Normalize with saved mean/std. "
        "Feed cropped grid [i_min_global:i_min_global+H, j_min_global:j_min_global+W]. "
        "Output is fire probability for that crop — place back into full grid at same offsets."
    ),
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved: {MODEL_PATH}  {WEIGHTS_PATH}  {META_PATH}")


# ============================================================
# STAGE 14 — SHAP + LIME EXPLAINABILITY
# ============================================================
# Both methods use a pixel-wrapper model that extracts a single
# scalar output (probability at target pixel), making SHAP and
# LIME tractable.
#
# Improvement over previous version:
#   - LIME background restricted to valid pixels only
#   - LIME num_samples raised to 500 (was 200 — too low for 84+ features)
#   - SHAP background sampled from valid-pixel rows
# ============================================================

try:
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    import matplotlib.pyplot as plt

    print("\n" + "="*60)
    print("STAGE 14 — SHAP + LIME")
    print("="*60)

    si = EXPLAIN_SAMPLE_INDEX
    ti = EXPLAIN_TARGET_I
    tj = EXPLAIN_TARGET_J

    assert 0 <= si < X_test_u.shape[0], "EXPLAIN_SAMPLE_INDEX out of range"
    assert 0 <= ti < H, "EXPLAIN_TARGET_I out of range"
    assert 0 <= tj < W, "EXPLAIN_TARGET_J out of range"

    # Check the target pixel is actually valid in this sample
    target_valid = M_test[si, ti, tj, 0]
    true_label   = Y_test[si, ti, tj, 0]
    base_prob    = unet.predict(X_test_u[si:si+1], verbose=0)[0, ti, tj, 0]

    print(f"Target: sample={si}, pixel=({ti},{tj})")
    print(f"Valid cell: {bool(target_valid)}  True label: {int(true_label)}  Predicted prob: {base_prob:.4f}")

    if not target_valid:
        print("WARNING: target pixel is not a valid (observed) cell in this sample.")
        print("Consider changing EXPLAIN_TARGET_I / EXPLAIN_TARGET_J to a valid cell.")

    # ---- Pixel-wrapper model ----
    pixel_out = layers.Lambda(
        lambda z: tf.expand_dims(z[:, ti, tj, 0], axis=-1),
        name="pixel_scalar"
    )(unet.output)
    pixel_model = tf.keras.Model(inputs=unet.input, outputs=pixel_out)

    print(f"Pixel model output shape: {pixel_model.output_shape}")

    # ---- Feature names for the collapsed (H, W, WINDOW*C) input ----
    def make_channel_names(feature_cols, window):
        return [f"t{t}_{f}" for t in range(window) for f in feature_cols]

    channel_names = make_channel_names(feature_cols, WINDOW)

    # ---- SHAP ----
    print("\n--- SHAP (GradientExplainer) ---")

    # Use only valid-cell samples from test set as background
    valid_test_mask = M_test[:, ti, tj, 0].astype(bool)
    valid_bg_pool   = X_test_u[valid_test_mask]

    if len(valid_bg_pool) < SHAP_BG_SIZE:
        print(f"  Only {len(valid_bg_pool)} valid samples at ({ti},{tj}) — using all for background")
        shap_bg = valid_bg_pool.astype(np.float32)
    else:
        idx = np.random.choice(len(valid_bg_pool), SHAP_BG_SIZE, replace=False)
        shap_bg = valid_bg_pool[idx].astype(np.float32)

    shap_explain = X_test_u[si:si+1].astype(np.float32)

    try:
        explainer        = shap.GradientExplainer(pixel_model, shap_bg)
        shap_values      = explainer.shap_values(shap_explain)
        sv               = np.array(shap_values).squeeze()  # (H, W, Cin)

        # Importance at target pixel only, broken down by day × feature
        pixel_sv         = sv[ti, tj, :]           # (Cin,)
        feat_time_mat    = pixel_sv.reshape(WINDOW, C)  # (WINDOW, C)
        total_importance = np.abs(feat_time_mat).sum(axis=0)

        print("\nSHAP feature importance at target pixel (summed across time steps):")
        order = np.argsort(-total_importance)
        for idx_f in order:
            print(f"  {feature_cols[idx_f]:20s}: {total_importance[idx_f]:.6f}")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(C), total_importance[np.arange(C)])
        ax.set_xticks(range(C))
        ax.set_xticklabels(feature_cols, rotation=45, ha="right")
        ax.set_title(f"SHAP feature importance at pixel ({ti},{tj})")
        plt.tight_layout()
        plt.savefig("shap_feature_importance.png", dpi=150)
        plt.show()
        print("Saved: shap_feature_importance.png")

        # Heatmap: day × feature importance
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(np.abs(feat_time_mat), aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(WINDOW))
        ax.set_yticklabels([f"t-{WINDOW-1-t}" for t in range(WINDOW)])
        ax.set_xticks(range(C))
        ax.set_xticklabels(feature_cols, rotation=45, ha="right")
        ax.set_title(f"SHAP |values| by day × feature at pixel ({ti},{tj})")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig("shap_time_feature_heatmap.png", dpi=150)
        plt.show()
        print("Saved: shap_time_feature_heatmap.png")

    except Exception as e:
        print(f"SHAP failed: {e!r}")

    # ---- LIME ----
    print("\n--- LIME (pixel-local tabular) ---")
    # We explain the channel vector at the target pixel only.
    # For each perturbed vector, replace that pixel's channels and predict.

    x0   = X_test_u[si].copy()         # (H, W, Cin)
    vec0 = x0[ti, tj, :].copy()        # (Cin,)

    # Training data for LIME: valid-pixel channel vectors at (ti, tj) across test set
    lime_train = X_test_u[valid_test_mask][:, ti, tj, :]  # (n_valid, Cin)
    if len(lime_train) < 10:
        lime_train = X_test_u[:, ti, tj, :]  # fallback: all test samples

    def lime_predict_fn(vectors):
        preds = []
        for v in vectors:
            x_copy = x0.copy()
            x_copy[ti, tj, :] = v.astype(np.float32)
            prob = pixel_model.predict(x_copy[None, ...], verbose=0)[0, 0]
            preds.append([1.0 - float(prob), float(prob)])
        return np.array(preds, dtype=np.float64)

    try:
        lime_explainer = LimeTabularExplainer(
            training_data=lime_train,
            feature_names=channel_names,
            class_names=["no_fire", "fire"],
            mode="classification",
            discretize_continuous=True,
            random_state=SEED,
        )

        exp = lime_explainer.explain_instance(
            data_row=vec0,
            predict_fn=lime_predict_fn,
            num_features=min(LIME_NUM_FEATURES, Cin),
            num_samples=LIME_NUM_SAMPLES,
        )

        print(f"\nLIME top {LIME_NUM_FEATURES} features at pixel ({ti},{tj}):")
        for feat, weight in exp.as_list():
            print(f"  {weight:+.6f}  {feat}")

        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig("lime_explanation.png", dpi=150)
        plt.show()
        print("Saved: lime_explanation.png")

    except Exception as e:
        print(f"LIME failed: {e!r}")

except ImportError as e:
    print(f"\nSHAP/LIME not installed — skipping explainability. ({e})")
    print("Install with: pip install shap lime")


# ============================================================
# STAGE 15 — FIRE PROBABILITY MAP (single date, for inspection)
# ============================================================

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    print("\n--- Generating probability map for last test sample ---")

    sample_idx = -1  # last test sample
    prob_map   = unet.predict(X_test_u[sample_idx:sample_idx+1], verbose=0)[0, :, :, 0]
    mask_map   = M_test[sample_idx, :, :, 0]
    label_map  = Y_test[sample_idx, :, :, 0]

    # Mask out invalid cells
    prob_display  = np.where(mask_map == 1, prob_map,  np.nan)
    label_display = np.where(mask_map == 1, label_map, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    cmap_prob = plt.cm.YlOrRd
    cmap_prob.set_bad(color="lightgrey")

    axes[0].imshow(prob_display, vmin=0, vmax=1, cmap=cmap_prob, origin="lower")
    axes[0].set_title("Predicted fire probability")
    axes[0].set_xlabel("lon grid"); axes[0].set_ylabel("lat grid")

    cmap_label = mcolors.ListedColormap(["#2d6a4f", "#d62828"])
    axes[1].imshow(label_display, vmin=0, vmax=1, cmap=cmap_label, origin="lower")
    axes[1].set_title("True label (0=no fire, 1=fire)")
    axes[1].set_xlabel("lon grid")

    axes[2].imshow(mask_map, cmap="Greys", origin="lower")
    axes[2].set_title("Valid mask (white=observed)")
    axes[2].set_xlabel("lon grid")

    plt.suptitle("Test set — last sample", fontsize=13)
    plt.tight_layout()
    plt.savefig("fire_prob_map.png", dpi=150)
    plt.show()
    print("Saved: fire_prob_map.png")

except Exception as e:
    print(f"Map plot failed: {e!r}")

print("\nDone.")
# %%
# ============================================================
# HEATMAPS FOR PRESENTATION
# ------------------------------------------------------------
# Produces:
# 1. predicted fire probability heatmap
# 2. ground truth fire map
# 3. valid mask map
# Invalid/background cells are shown as grey.
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_fire_heatmaps(sample_idx, save_prefix="unet_heatmap"):
    prob_map = unet.predict(X_test_u[sample_idx:sample_idx+1], verbose=0)[0, :, :, 0]
    label_map = Y_test[sample_idx, :, :, 0]
    mask_map = M_test[sample_idx, :, :, 0]

    # Apply explicit valid mask for visualization
    prob_display = np.where(mask_map == 1, prob_map, np.nan)
    label_display = np.where(mask_map == 1, label_map, np.nan)

    cmap_prob = plt.cm.YlOrRd.copy()
    cmap_prob.set_bad(color="lightgrey")

    cmap_label = mcolors.ListedColormap(["#f7f7f7", "#d62828"])
    cmap_label.set_bad(color="lightgrey")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(prob_display, vmin=0, vmax=1, cmap=cmap_prob, origin="lower")
    axes[0].set_title("Predicted fire probability")
    axes[0].set_xlabel("Longitude grid")
    axes[0].set_ylabel("Latitude grid")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(label_display, vmin=0, vmax=1, cmap=cmap_label, origin="lower")
    axes[1].set_title("Ground truth fire map")
    axes[1].set_xlabel("Longitude grid")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(mask_map, vmin=0, vmax=1, cmap="Greys", origin="lower")
    axes[2].set_title("Valid mask")
    axes[2].set_xlabel("Longitude grid")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle(f"UNet masked prediction — test sample {sample_idx}")
    plt.tight_layout()

    out_path = f"{save_prefix}_{sample_idx}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()

    print("Saved:", out_path)


# Example: expose two heatmaps
plot_fire_heatmaps(sample_idx=0, save_prefix="unet_masked_heatmap")
plot_fire_heatmaps(sample_idx=1, save_prefix="unet_masked_heatmap")
# %%
