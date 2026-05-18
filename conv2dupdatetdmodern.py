# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =========================
# 0) CONFIG
# =========================
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

# =========================
# 1) LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])

feature_cols = [
    "t2m", "d2m", "u10", "v10", "tp", "fwi",
    "KBDI", "PET", "D", "SPEI_30", "SPEI_90", "SPEI_180"
]
feature_cols = [c for c in feature_cols if c in df.columns]
C = len(feature_cols)

df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))

# =========================
# 2) SPATIAL BINNING -> GRID (H,W)
# =========================
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

print("H:", H, "W:", W, "C:", C, "cells:", H * W)

dates = np.sort(df["date"].unique())
T_total = len(dates)
print("Total dates:", T_total)

date_to_t = {d: idx for idx, d in enumerate(dates)}

# Map bins to grid indices (vectorized)
df["i"] = df["lat_bin"].map(latbin_to_i).astype(int)
df["j"] = df["lon_bin"].map(lonbin_to_j).astype(int)
df["t"] = df["date"].map(date_to_t).astype(int)

# =========================
# 3) BUILD FRAMES (FAST)
# =========================
frames = np.zeros((T_total, H, W, C), dtype=np.float32)
labels = np.zeros((T_total, H, W, 1), dtype=np.float32)

# Fill frames/labels using numpy indexing
t_idx = df["t"].to_numpy()
i_idx = df["i"].to_numpy()
j_idx = df["j"].to_numpy()

frames[t_idx, i_idx, j_idx, :] = df[feature_cols].to_numpy(dtype=np.float32)
labels[t_idx, i_idx, j_idx, 0] = df["y_fire"].to_numpy(dtype=np.float32)

print("Frames:", frames.shape, "Labels:", labels.shape)

# =========================
# 4) MAKE SEQUENCES
# =========================
X_seq = np.stack([frames[t - WINDOW + 1: t + 1] for t in range(WINDOW - 1, T_total)], axis=0)
Y_seq = np.stack([labels[t] for t in range(WINDOW - 1, T_total)], axis=0)

N = X_seq.shape[0]
print("Sequences X:", X_seq.shape, "Y:", Y_seq.shape)

# =========================
# 5) TIME SPLIT
# =========================
n_train = int(TRAIN_RATIO * N)
n_val = int((TRAIN_RATIO + VAL_RATIO) * N)

X_train, Y_train = X_seq[:n_train], Y_seq[:n_train]
X_val,   Y_val   = X_seq[n_train:n_val], Y_seq[n_train:n_val]
X_test,  Y_test  = X_seq[n_val:], Y_seq[n_val:]

print("Train:", X_train.shape, Y_train.shape)
print("Val:  ", X_val.shape,   Y_val.shape)
print("Test: ", X_test.shape,  Y_test.shape)

# =========================
# 6) NORMALIZE (fit on train only)
# =========================
train_flat = X_train.reshape(-1, C)
mean = train_flat.mean(axis=0)
std = train_flat.std(axis=0)
std[std == 0] = 1.0

def normalize(X):
    return (X - mean[None, None, None, None, :]) / std[None, None, None, None, :]

X_train = normalize(X_train).astype(np.float32)
X_val   = normalize(X_val).astype(np.float32)
X_test  = normalize(X_test).astype(np.float32)

# =========================
# 7) FOCAL LOSS
# =========================
def focal_loss(alpha=0.85, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        w = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        return tf.reduce_mean(-w * tf.pow(1.0 - pt, gamma) * tf.math.log(pt))
    return loss

# =========================
# 8) MODEL (Keras-friendly pos embeddings)
# =========================
class AddPositionalEmbeddings(layers.Layer):
    def __init__(self, window, num_patches, d_model, dropout=0.1):
        super().__init__()
        self.window = window
        self.num_patches = num_patches
        self.d_model = d_model
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape):
        # (1, 1, Np, D) and (1, T, 1, D)
        self.pos_spatial = self.add_weight(
            name="pos_spatial",
            shape=(1, 1, self.num_patches, self.d_model),
            initializer="random_normal",
            trainable=True
        )
        self.pos_temporal = self.add_weight(
            name="pos_temporal",
            shape=(1, self.window, 1, self.d_model),
            initializer="random_normal",
            trainable=True
        )

    def call(self, x, training=False):
        x = x + self.pos_spatial + self.pos_temporal
        return self.dropout(x, training=training)

def transformer_block(x, d_model, n_heads, mlp_ratio=4, dropout=0.1):
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads, dropout=dropout)(y, y)
    x = layers.Add()([x, y])

    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(d_model * mlp_ratio, activation="gelu")(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(d_model)(y)
    x = layers.Add()([x, y])
    return x

def build_spatiotemporal_vit(window, H, W, C, patch, d_model, n_heads, n_layers, mlp_ratio, dropout):
    inp = Input(shape=(window, H, W, C))

    pad_h = (patch - (H % patch)) % patch
    pad_w = (patch - (W % patch)) % patch

    def pad_hw(x):
        if pad_h == 0 and pad_w == 0:
            return x
        return tf.pad(x, [[0,0],[0,0],[0,pad_h],[0,pad_w],[0,0]])

    x = layers.Lambda(pad_hw)(inp)
    H2, W2 = H + pad_h, W + pad_w
    Hp, Wp = H2 // patch, W2 // patch
    Np = Hp * Wp

    patch_embed = layers.TimeDistributed(
        layers.Conv2D(filters=d_model, kernel_size=patch, strides=patch, padding="valid")
    )(x)  # (B,T,Hp,Wp,D)

    tokens = layers.Reshape((window, Np, d_model))(patch_embed)  # (B,T,Np,D)
    tokens = AddPositionalEmbeddings(window, Np, d_model, dropout)(tokens)

    tokens = layers.Reshape((window * Np, d_model))(tokens)  # (B, T*Np, D)

    for _ in range(n_layers):
        tokens = transformer_block(tokens, d_model, n_heads, mlp_ratio, dropout)

    tokens = layers.Reshape((window, Np, d_model))(tokens)
    last_tokens = layers.Lambda(lambda z: z[:, -1, :, :])(tokens)       # (B,Np,D)
    feat_map = layers.Reshape((Hp, Wp, d_model))(last_tokens)          # (B,Hp,Wp,D)

    # Stronger decoder
    x = layers.Conv2DTranspose(64, kernel_size=patch, strides=patch, padding="valid", activation="relu")(feat_map)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)

    if pad_h != 0 or pad_w != 0:
        out = layers.Cropping2D(cropping=((0, pad_h), (0, pad_w)))(out)

    return Model(inp, out)

model = build_spatiotemporal_vit(
    window=WINDOW, H=H, W=W, C=C,
    patch=PATCH, d_model=D_MODEL, n_heads=N_HEADS,
    n_layers=N_LAYERS, mlp_ratio=MLP_RATIO, dropout=DROPOUT
)

pr_auc = tf.keras.metrics.AUC(curve="PR", name="pr_auc")
roc_auc = tf.keras.metrics.AUC(curve="ROC", name="roc_auc")

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=focal_loss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA),
    metrics=["accuracy", pr_auc, roc_auc],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=2, min_lr=1e-5),
]

# =========================
# 9) TRAIN
# =========================
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# =========================
# 10) EVALUATE (default threshold 0.5)
# =========================
pred_test = model.predict(X_test, batch_size=1)

y_true = (Y_test > 0.5).astype(np.uint8).ravel()
y_prob = pred_test.ravel()
y_pred = (y_prob > 0.5).astype(np.uint8)

cm = confusion_matrix(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)

loss, acc, pr, roc = model.evaluate(X_test, Y_test, verbose=0)

print("\n--- SpatioTemporal ViT Test (thr=0.5) ---")
print("Test loss:", loss)
print("Test accuracy:", acc)
print("Test PR-AUC:", pr)
print("Test ROC-AUC:", roc)
print("Confusion [[TN FP],[FN TP]]:\n", cm)
print("Precision:", prec)
print("Recall:   ", rec)
print("F1:       ", f1)

# =========================
# 11) THRESHOLD TUNING ON VAL
# =========================
def best_threshold_by_f1(y_true_bin, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (y_prob >= t).astype(np.uint8)
        f1v = f1_score(y_true_bin, pred, zero_division=0)
        if f1v > best_f1:
            best_f1 = f1v
            best_t = t
    return best_t, best_f1

pred_val = model.predict(X_val, batch_size=1)
y_val_true = (Y_val > 0.5).astype(np.uint8).ravel()
y_val_prob = pred_val.ravel()

t_best, f1_best = best_threshold_by_f1(y_val_true, y_val_prob)
print(f"\nBest val threshold: {t_best:.3f} | Best val F1: {f1_best:.4f}")

y_test_hat = (y_prob >= t_best).astype(np.uint8)
print("Test F1 @ best threshold:", f1_score(y_true, y_test_hat, zero_division=0))

# %%
