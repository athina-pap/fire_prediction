#%%
# ============================================================
# TRAIN: CSV ViT ONLY
# - Keras 3 safe
# - Saves:
#   vit_fire_model_csv.keras
#   vit_fire_model_csv_legacy.h5
#   vit_fire_model_csv.weights.h5
#   vit_fire_model_csv_meta.json
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
print("TensorFlow:", tf.__version__)

# -----------------------------
# CONFIG
# -----------------------------
CSV_PATH = "training_dataset_final_prepared.csv"

WINDOW = 7
LAT_BIN_DEG = 0.25
LON_BIN_DEG = 0.25

PATCH = 8
D_MODEL = 96
N_HEADS = 4
N_LAYERS = 4
MLP_RATIO = 4
DROPOUT = 0.1

LR = 1e-3
EPOCHS = 30
BATCH_SIZE = 4

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

FOCAL_ALPHA = 0.85
FOCAL_GAMMA = 2.0

# -----------------------------
# LOAD + PREP DATA
# -----------------------------
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])

feature_cols = [
    "t2m", "d2m", "u10", "v10", "tp", "fwi",
    "KBDI", "PET", "D", "SPEI_30", "SPEI_90", "SPEI_180"
]
feature_cols = [c for c in feature_cols if c in df.columns]
C = len(feature_cols)

print("Features:", feature_cols)
print("C =", C)

if C == 0:
    raise ValueError("No valid feature columns found in CSV.")

df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

if "y_fire" not in df.columns:
    raise ValueError("Column y_fire not found in CSV.")

df["y_fire"] = df["y_fire"].fillna(0).astype(float)

required_cols = ["lat", "lon", "date"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")

# spatial binning
lat_min = df["lat"].min()
lon_min = df["lon"].min()

df["lat_bin"] = ((df["lat"] - lat_min) / LAT_BIN_DEG).round().astype(int)
df["lon_bin"] = ((df["lon"] - lon_min) / LON_BIN_DEG).round().astype(int)

lat_bins_sorted = np.sort(df["lat_bin"].unique())
lon_bins_sorted = np.sort(df["lon_bin"].unique())

latbin_to_i = {b: i for i, b in enumerate(lat_bins_sorted)}
lonbin_to_j = {b: j for j, b in enumerate(lon_bins_sorted)}


H = len(lat_bins_sorted)
W = len(lon_bins_sorted)

print("Grid HxW:", H, "x", W, "cells:", H * W)

# build daily tensors
dates = np.sort(df["date"].unique())
T_total = len(dates)
date_to_t = {d: idx for idx, d in enumerate(dates)}

print("Total dates T =", T_total)

df["i"] = df["lat_bin"].map(latbin_to_i).astype(int)
df["j"] = df["lon_bin"].map(lonbin_to_j).astype(int)
df["t"] = df["date"].map(date_to_t).astype(int)

frames = np.zeros((T_total, H, W, C), dtype=np.float32)
labels = np.zeros((T_total, H, W, 1), dtype=np.float32)

t_idx = df["t"].to_numpy()
i_idx = df["i"].to_numpy()
j_idx = df["j"].to_numpy()

frames[t_idx, i_idx, j_idx, :] = df[feature_cols].to_numpy(dtype=np.float32)
labels[t_idx, i_idx, j_idx, 0] = df["y_fire"].to_numpy(dtype=np.float32)

print("frames:", frames.shape, "labels:", labels.shape)
print("Fire positive rate:", float(labels.mean()))

if T_total < WINDOW:
    raise ValueError(f"Not enough timesteps ({T_total}) for WINDOW={WINDOW}.")

# windowing
X_seq = np.stack(
    [frames[t - WINDOW + 1:t + 1] for t in range(WINDOW - 1, T_total)],
    axis=0
)
Y_seq = np.stack(
    [labels[t] for t in range(WINDOW - 1, T_total)],
    axis=0
)

N = X_seq.shape[0]
print("X_seq:", X_seq.shape, "Y_seq:", Y_seq.shape, "N =", N)

# time split
n_train = int(TRAIN_RATIO * N)
n_val = int((TRAIN_RATIO + VAL_RATIO) * N)

X_train, Y_train = X_seq[:n_train], Y_seq[:n_train]
X_val,   Y_val   = X_seq[n_train:n_val], Y_seq[n_train:n_val]
X_test,  Y_test  = X_seq[n_val:], Y_seq[n_val:]

print("Train:", X_train.shape, Y_train.shape)
print("Val:  ", X_val.shape, Y_val.shape)
print("Test: ", X_test.shape, Y_test.shape)

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise ValueError("Train/Val/Test split produced an empty split. Check dataset size and ratios.")

# normalize (train stats only)
train_flat = X_train.reshape(-1, C)
mean = train_flat.mean(axis=0)
std = train_flat.std(axis=0)
std[std == 0] = 1.0

def normalize(X):
    return (X - mean[None, None, None, None, :]) / std[None, None, None, None, :]

X_train = normalize(X_train).astype(np.float32)
X_val   = normalize(X_val).astype(np.float32)
X_test  = normalize(X_test).astype(np.float32)

print("Normalization done.")

# -----------------------------
# LOSSES / METRICS
# -----------------------------
def focal_loss(alpha=0.85, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        w = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return tf.reduce_mean(-w * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss

pr_auc_metric = tf.keras.metrics.AUC(curve="PR", name="pr_auc")
roc_auc_metric = tf.keras.metrics.AUC(curve="ROC", name="roc_auc")

# -----------------------------
# VIT (stable names)
# -----------------------------
class AddPositionalEmbeddings(layers.Layer):
    def __init__(self, window, num_patches, d_model, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.window = int(window)
        self.num_patches = int(num_patches)
        self.d_model = int(d_model)
        self.dropout_rate = float(dropout)
        self.dropout = layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        self.pos_spatial = self.add_weight(
            name="pos_spatial",
            shape=(1, 1, self.num_patches, self.d_model),
            initializer="random_normal",
            trainable=True,
        )
        self.pos_temporal = self.add_weight(
            name="pos_temporal",
            shape=(1, self.window, 1, self.d_model),
            initializer="random_normal",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, training=False):
        x = x + self.pos_spatial + self.pos_temporal
        return self.dropout(x, training=training)

def transformer_block(x, d_model, n_heads, mlp_ratio=4, dropout=0.1, name="block"):
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    y = layers.MultiHeadAttention(
        num_heads=n_heads,
        key_dim=d_model // n_heads,
        dropout=dropout,
        name=f"{name}_mha",
    )(y, y)
    y = layers.Dropout(dropout, name=f"{name}_drop1")(y)
    x = layers.Add(name=f"{name}_add1")([x, y])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    y = layers.Dense(d_model * mlp_ratio, activation="gelu", name=f"{name}_fc1")(y)
    y = layers.Dropout(dropout, name=f"{name}_drop2")(y)
    y = layers.Dense(d_model, name=f"{name}_fc2")(y)
    y = layers.Dropout(dropout, name=f"{name}_drop3")(y)
    x = layers.Add(name=f"{name}_add2")([x, y])

    return x

def build_spatiotemporal_vit(window, H, W, C, patch, d_model, n_heads, n_layers, mlp_ratio, dropout):
    inp = Input(shape=(window, H, W, C), name="vit_input")

    pad_h = (patch - (H % patch)) % patch
    pad_w = (patch - (W % patch)) % patch

    def pad_hw(x):
        if pad_h == 0 and pad_w == 0:
            return x
        return tf.pad(x, [[0, 0], [0, 0], [0, pad_h], [0, pad_w], [0, 0]])

    x = layers.Lambda(pad_hw, name="pad_hw")(inp)

    H2, W2 = H + pad_h, W + pad_w
    Hp, Wp = H2 // patch, W2 // patch
    Np = Hp * Wp

    patch_embed = layers.TimeDistributed(
        layers.Conv2D(
            filters=d_model,
            kernel_size=patch,
            strides=patch,
            padding="valid"
        ),
        name="patch_embed"
    )(x)  # (B,T,Hp,Wp,D)

    tokens = layers.Reshape((window, Np, d_model), name="tokens_reshaped")(patch_embed)
    tokens = AddPositionalEmbeddings(window, Np, d_model, dropout, name="pos_emb")(tokens)

    tokens = layers.Reshape((window * Np, d_model), name="tokens_flat")(tokens)

    for i in range(n_layers):
        tokens = transformer_block(
            tokens,
            d_model=d_model,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            name=f"block_{i}"
        )

    tokens = layers.Reshape((window, Np, d_model), name="tokens_unflat")(tokens)
    last_tokens = layers.Lambda(lambda z: z[:, -1, :, :], name="last_tokens")(tokens)
    feat_map = layers.Reshape((Hp, Wp, d_model), name="feat_map")(last_tokens)

    x = layers.Conv2DTranspose(
        64,
        kernel_size=patch,
        strides=patch,
        padding="valid",
        activation="relu",
        name="up"
    )(feat_map)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="dec_conv1")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="dec_conv2")(x)
    out = layers.Conv2D(1, 1, activation="sigmoid", name="out")(x)

    if pad_h != 0 or pad_w != 0:
        out = layers.Cropping2D(cropping=((0, pad_h), (0, pad_w)), name="crop_back")(out)

    return Model(inp, out, name="ViT")

vit = build_spatiotemporal_vit(
    window=WINDOW,
    H=H,
    W=W,
    C=C,
    patch=PATCH,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    mlp_ratio=MLP_RATIO,
    dropout=DROPOUT
)

vit.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA),
    metrics=["accuracy", pr_auc_metric, roc_auc_metric],
)

vit.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc",
        mode="max",
        patience=6,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_pr_auc",
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-5
    ),
]

# -----------------------------
# TRAIN VIT
# -----------------------------
print("\n=== Training ViT ===")
hist_vit = vit.fit(
    X_train,
    Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# -----------------------------
# EVALUATE VIT
# -----------------------------
print("\n=== Eval ViT thr=0.5 ===")
pred_test = vit.predict(X_test, batch_size=1, verbose=0)

y_true = (Y_test > 0.5).astype(np.uint8).ravel()
y_prob = pred_test.ravel()
y_pred = (y_prob > 0.5).astype(np.uint8)

cm = confusion_matrix(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

loss, acc, pr, roc = vit.evaluate(X_test, Y_test, verbose=0)

print("\n--- ViT Test ---")
print("loss:", loss)
print("acc:", acc)
print("pr_auc:", pr)
print("roc_auc:", roc)
print("Confusion [[TN FP],[FN TP]]:\n", cm)
print("Precision:", prec)
print("Recall:", rec)
print("F1:", f1)

# -----------------------------
# SAVING VIT ONLY
# -----------------------------
def force_build_and_save(model_obj, input_shape, keras_path, legacy_path, weights_path):
    _ = model_obj(tf.zeros(input_shape, dtype=tf.float32))
    model_obj.save(keras_path)
    model_obj.save(legacy_path)
    model_obj.save_weights(weights_path)

VIT_MODEL_PATH   = "TrainHeat/vit_model.keras"
VIT_LEGACY_PATH  = "TrainHeat/vit_model.h5"
VIT_WEIGHTS_PATH = "TrainHeat/vit_model.weights.h5"
VIT_META_PATH    = "TrainHeat/vit_model_csv_meta.json"

force_build_and_save(
    vit,
    input_shape=(1, WINDOW, H, W, C),
    keras_path=VIT_MODEL_PATH,
    legacy_path=VIT_LEGACY_PATH,
    weights_path=VIT_WEIGHTS_PATH
)

vit_meta = {
    "feature_cols": feature_cols,
    "mean": mean.tolist(),
    "std": std.tolist(),
    "WINDOW": int(WINDOW),
    "PATCH": int(PATCH),
    "D_MODEL": int(D_MODEL),
    "N_HEADS": int(N_HEADS),
    "N_LAYERS": int(N_LAYERS),
    "MLP_RATIO": int(MLP_RATIO),
    "DROPOUT": float(DROPOUT),
    "LAT_BIN_DEG": float(LAT_BIN_DEG),
    "LON_BIN_DEG": float(LON_BIN_DEG),
    "H": int(H),
    "W": int(W),
    "C": int(C),
    "model_name": "ViT",
    "csv_path": CSV_PATH
}

with open(VIT_META_PATH, "w", encoding="utf-8") as f:
    json.dump(vit_meta, f, indent=2)

print("\n✅ ViT saved:")
print(" -", VIT_MODEL_PATH)
print(" -", VIT_LEGACY_PATH)
print(" -", VIT_WEIGHTS_PATH)
print(" -", VIT_META_PATH)

# -----------------------------
# DEBUG: stable layer names
# -----------------------------
print("\n[DEBUG] vit layer name sanity check:")
print("Total vit.layers =", len(vit.layers))

block_like = [l.name for l in vit.layers if l.name.startswith("block_")]
print("block_* layers (sample):", block_like[:40])

mha_like = [l.name for l in vit.layers if "_mha" in l.name]
ln_like = [l.name for l in vit.layers if "_ln" in l.name]
print("mha layers:", mha_like)
print("ln layers:", ln_like)

if len(mha_like) == 0 and len(ln_like) == 0:
    print("❌ Δεν βλέπω *_mha ή *_ln layers στο vit.")
else:
    print("✅ Stable transformer names υπάρχουν μέσα στο vit object.")

print("\n✅ DONE.")

#%%