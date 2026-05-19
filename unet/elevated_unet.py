# %%
# ============================================================
# ELEVATED UNET - RESIDUAL CONDITIONAL VAE
# ============================================================
#
# This file is a standalone "elevated model" script.
#
# It does NOT retrain the base UNet from zero.
# It loads the Improved UNet that you already trained:
#
#   unet_fire_improved.keras
#   unet_fire_improved_meta.json
#
# Then it rebuilds the same test/train/validation tensors from:
#
#   training_dataset_final_prepared.csv
#
# and trains an additional AI component:
#
#   Residual Conditional VAE
#
# The purpose is to turn the deterministic UNet into an
# uncertainty-aware model.
#
# Output:
#   1. base UNet probability map
#   2. elevated mean risk map
#   3. uncertainty map
#   4. masked evaluation metrics
#   5. heatmaps for presentation
#
# ============================================================


# ============================================================
# 0. IMPORTS
# ============================================================

import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, Model, Input
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. CONFIG
# ============================================================
# These paths must match the files produced by your improved UNet script.

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CSV_PATH = "training_dataset_final_prepared.csv"

BASE_UNET_MODEL_PATH = "unet_fire_improved.keras"
BASE_UNET_META_PATH = "unet_fire_improved_meta.json"

CVAE_MODEL_PATH = "elevated_unet_residual_cvae.keras"
CVAE_META_PATH = "elevated_unet_residual_cvae_meta.json"
OUTPUT_NPZ_PATH = "elevated_unet_outputs.npz"

# Training parameters for the elevated residual-cVAE.
CVAE_EPOCHS = 40
CVAE_BATCH_SIZE = 4
CVAE_LR = 1e-4

# Latent dimension controls how many hidden uncertainty factors the cVAE can learn.
LATENT_DIM = 16

# KL regularization weight.
# Small value keeps the VAE stable without destroying reconstruction quality.
BETA_KL = 1e-4

# Number of Monte-Carlo samples used to estimate mean risk and uncertainty.
N_SAMPLES = 30


# ============================================================
# 2. LOAD BASE UNET METADATA
# ============================================================
# The metadata contains:
# - feature list
# - normalization mean/std
# - grid settings
# - train/val/test logic
# - best threshold
#
# This is important because the elevated model must use the exact
# same preprocessing as the trained UNet.

with open(BASE_UNET_META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

feature_cols = meta["feature_cols"]
base_feature_cols = meta.get("base_feature_cols", [
    "t2m", "d2m", "u10", "v10", "tp", "fwi",
    "KBDI", "PET", "D", "SPEI_30", "SPEI_90", "SPEI_180",
])
spei_obs_cols = meta.get("spei_obs_cols", [])

WINDOW = int(meta["WINDOW"])
LAT_BIN_DEG = float(meta["LAT_BIN_DEG"])
LON_BIN_DEG = float(meta["LON_BIN_DEG"])

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

H = int(meta["H"])
W = int(meta["W"])
C = int(meta["C"])
Cin = int(meta["Cin"])

lat_min_saved = float(meta["lat_min"])
lon_min_saved = float(meta["lon_min"])

i_min_global = int(meta["i_min_global"])
j_min_global = int(meta["j_min_global"])

mean = np.array(meta["mean"], dtype=np.float32)
std = np.array(meta["std"], dtype=np.float32)

BASE_THRESHOLD = float(meta.get("best_threshold", 0.5))

print("Loaded base UNet metadata.")
print("Features:", feature_cols)
print("Grid HxW:", H, "x", W)
print("Input channels:", Cin)
print("Base threshold:", BASE_THRESHOLD)


# ============================================================
# 3. LOAD TRAINED BASE UNET
# ============================================================
# This is the already trained deterministic model.
# It is frozen because the elevated stage should only learn
# uncertainty/residual correction, not retrain the base predictor.

@tf.keras.utils.register_keras_serializable()
def pad_hw(x):
    factor = 16

    H_in = x.shape[1]
    W_in = x.shape[2]

    pad_h = (factor - (H_in % factor)) % factor
    pad_w = (factor - (W_in % factor)) % factor

    return tf.pad(
        x,
        [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
    )

unet = tf.keras.models.load_model(
    BASE_UNET_MODEL_PATH,
    compile=False,
    custom_objects={"pad_hw": pad_hw}
)

print("Loaded base Improved UNet.")
print("UNet output shape:", unet.output_shape)


# ============================================================
# 4. REBUILD DATASET EXACTLY LIKE BASE IMPROVED UNET
# ============================================================
# This block repeats the same preprocessing used by the training script:
#
# 1. SPEI missingness indicators
# 2. day-of-year cyclic encoding
# 3. spatial grid
# 4. global bbox crop
# 5. daily tensors
# 6. valid mask
# 7. gap-aware windowing
# 8. train/val/test split
# 9. normalization from saved mean/std
# 10. collapse time to channels for UNet

df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"])

if "y_fire" not in df.columns:
    raise ValueError("Column y_fire not found.")

# -----------------------------
# 4.1 Feature engineering
# -----------------------------
# SPEI columns get missingness flags.
# This must match the improved UNet training.

spei_cols = [c for c in ["SPEI_30", "SPEI_90", "SPEI_180"] if c in df.columns]

for col in spei_cols:
    obs_col = f"{col}_obs"
    df[obs_col] = (~df[col].isnull()).astype(np.float32)
    df[col] = df[col].fillna(0.0)

# Fill remaining base feature NaNs with median.
available_base = [c for c in base_feature_cols if c in df.columns]
df[available_base] = df[available_base].fillna(
    df[available_base].median(numeric_only=True)
)

# Cyclic day-of-year encoding.
df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365).astype(np.float32)
df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365).astype(np.float32)

df["y_fire"] = df["y_fire"].fillna(0).astype(np.float32)

# Check that all expected features exist.
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing features required by base UNet metadata: {missing}")


# -----------------------------
# 4.2 Spatial grid and crop
# -----------------------------
# Important: use saved lat_min/lon_min from the base model metadata,
# not recomputed values, to keep grid alignment identical.

df["lat_bin"] = ((df["lat"] - lat_min_saved) / LAT_BIN_DEG).round().astype(int)
df["lon_bin"] = ((df["lon"] - lon_min_saved) / LON_BIN_DEG).round().astype(int)

lat_bins_sorted = np.sort(df["lat_bin"].unique())
lon_bins_sorted = np.sort(df["lon_bin"].unique())

latbin_to_i = {b: i for i, b in enumerate(lat_bins_sorted)}
lonbin_to_j = {b: j for j, b in enumerate(lon_bins_sorted)}

df["i"] = df["lat_bin"].map(latbin_to_i).astype(int)
df["j"] = df["lon_bin"].map(lonbin_to_j).astype(int)

df["i_crop"] = df["i"] - i_min_global
df["j_crop"] = df["j"] - j_min_global

# Keep only points inside the trained crop.
inside = (
    (df["i_crop"] >= 0) & (df["i_crop"] < H) &
    (df["j_crop"] >= 0) & (df["j_crop"] < W)
)

dropped = len(df) - int(inside.sum())
if dropped > 0:
    print(f"Warning: dropped {dropped} rows outside saved crop.")

df = df[inside].copy()


# -----------------------------
# 4.3 Daily tensors and valid mask
# -----------------------------

dates_sorted = np.sort(df["date"].unique())
T_total = len(dates_sorted)

date_to_t = {d: idx for idx, d in enumerate(dates_sorted)}
df["t"] = df["date"].map(date_to_t).astype(int)

frames = np.zeros((T_total, H, W, C), dtype=np.float32)
labels = np.zeros((T_total, H, W, 1), dtype=np.float32)
valid_mask = np.zeros((T_total, H, W, 1), dtype=np.float32)

t_arr = df["t"].to_numpy()
i_arr = df["i_crop"].to_numpy()
j_arr = df["j_crop"].to_numpy()

frames[t_arr, i_arr, j_arr, :] = df[feature_cols].to_numpy(dtype=np.float32)
labels[t_arr, i_arr, j_arr, 0] = df["y_fire"].to_numpy(dtype=np.float32)
valid_mask[t_arr, i_arr, j_arr, 0] = 1.0

print("frames:", frames.shape)
print("labels:", labels.shape)
print("valid_mask:", valid_mask.shape)
print("Valid pixel rate:", float(valid_mask.mean()))
print("Fire rate over valid pixels:", float(labels[valid_mask == 1].mean()))


# -----------------------------
# 4.4 Gap-aware windowing
# -----------------------------
# We only use windows where the 7 days are truly consecutive.

day_gaps = np.diff([pd.Timestamp(d).toordinal() for d in dates_sorted])

valid_window_ends = []

for t in range(WINDOW - 1, T_total):
    window_gaps = day_gaps[t - WINDOW + 1:t]
    if np.all(window_gaps == 1):
        valid_window_ends.append(t)

if len(valid_window_ends) == 0:
    raise ValueError("No valid 7-day contiguous windows found.")

print("Valid windows:", len(valid_window_ends))

X_seq = np.stack(
    [frames[t - WINDOW + 1:t + 1] for t in valid_window_ends],
    axis=0
)

Y_seq = np.stack(
    [labels[t] for t in valid_window_ends],
    axis=0
)

M_seq = np.stack(
    [valid_mask[t] for t in valid_window_ends],
    axis=0
)

print("X_seq:", X_seq.shape)
print("Y_seq:", Y_seq.shape)
print("M_seq:", M_seq.shape)


# -----------------------------
# 4.5 Train / val / test split
# -----------------------------

N = X_seq.shape[0]
n_train = int(TRAIN_RATIO * N)
n_val = int((TRAIN_RATIO + VAL_RATIO) * N)

X_train, Y_train, M_train = X_seq[:n_train], Y_seq[:n_train], M_seq[:n_train]
X_val, Y_val, M_val = X_seq[n_train:n_val], Y_seq[n_train:n_val], M_seq[n_train:n_val]
X_test, Y_test, M_test = X_seq[n_val:], Y_seq[n_val:], M_seq[n_val:]

valid_ends_train = valid_window_ends[:n_train]
valid_ends_val = valid_window_ends[n_train:n_val]
valid_ends_test = valid_window_ends[n_val:]

print("Train / Val / Test:", X_train.shape[0], X_val.shape[0], X_test.shape[0])


# -----------------------------
# 4.6 Normalize with saved training mean/std
# -----------------------------
# We use the exact mean/std learned by the base UNet training script.

def normalize(X):
    return (X - mean[None, None, None, None, :]) / std[None, None, None, None, :]

X_train = normalize(X_train).astype(np.float32)
X_val = normalize(X_val).astype(np.float32)
X_test = normalize(X_test).astype(np.float32)


def zero_invalid_inputs(X, window_ends_subset):
    """
    After normalization, invalid cells would no longer necessarily be zero.
    We reset them to zero to preserve the explicit background convention.
    """
    X_out = X.copy()

    for n, t in enumerate(window_ends_subset):
        input_mask = valid_mask[t - WINDOW + 1:t + 1]
        X_out[n][input_mask[..., 0] == 0] = 0.0

    return X_out


X_train = zero_invalid_inputs(X_train, valid_ends_train)
X_val = zero_invalid_inputs(X_val, valid_ends_val)
X_test = zero_invalid_inputs(X_test, valid_ends_test)


# -----------------------------
# 4.7 Collapse time into channels for UNet
# -----------------------------

def collapse_time_to_channels(X):
    """
    Convert:
        (N, WINDOW, H, W, C)
    into:
        (N, H, W, WINDOW*C)

    This matches the base Improved UNet input format.
    """
    Nn, Tt, Hh, Ww, Cc = X.shape
    return X.transpose(0, 2, 3, 1, 4).reshape(Nn, Hh, Ww, Tt * Cc)


X_train_u = collapse_time_to_channels(X_train)
X_val_u = collapse_time_to_channels(X_val)
X_test_u = collapse_time_to_channels(X_test)

if X_train_u.shape[-1] != Cin:
    raise ValueError(
        f"Input channel mismatch. Expected {Cin}, got {X_train_u.shape[-1]}"
    )

print("X_train_u:", X_train_u.shape)
print("X_val_u:", X_val_u.shape)
print("X_test_u:", X_test_u.shape)


# ============================================================
# 5. BASE UNET PREDICTIONS
# ============================================================
# These are deterministic risk maps from the trained Improved UNet.
# The elevated model learns residual uncertainty around these maps.

print("\nGenerating base UNet predictions...")

base_train_prob = unet.predict(X_train_u, batch_size=1, verbose=1)
base_val_prob = unet.predict(X_val_u, batch_size=1, verbose=1)
base_test_prob = unet.predict(X_test_u, batch_size=1, verbose=1)

print("base_train_prob:", base_train_prob.shape)
print("base_val_prob:", base_val_prob.shape)
print("base_test_prob:", base_test_prob.shape)


# ============================================================
# 6. RESIDUAL TARGETS
# ============================================================
# The residual is the error/correction map:
#
#   residual = y_true - base_probability
#
# If the UNet underpredicts a fire pixel, residual is positive.
# If the UNet overpredicts a non-fire pixel, residual is negative.

res_train = Y_train - base_train_prob
res_val = Y_val - base_val_prob
res_test = Y_test - base_test_prob

print("Residual train min/max:", float(res_train.min()), float(res_train.max()))


# ============================================================
# 7. BUILD CVAE INPUTS
# ============================================================
# Encoder sees:
#   base probability + residual + valid mask
#
# Decoder condition sees:
#   base probability
#
# Target for loss:
#   residual + valid mask

encoder_train_input = np.concatenate(
    [base_train_prob, res_train, M_train],
    axis=-1
).astype(np.float32)

encoder_val_input = np.concatenate(
    [base_val_prob, res_val, M_val],
    axis=-1
).astype(np.float32)

condition_train = base_train_prob.astype(np.float32)
condition_val = base_val_prob.astype(np.float32)

cvae_y_train = np.concatenate([res_train, M_train], axis=-1).astype(np.float32)
cvae_y_val = np.concatenate([res_val, M_val], axis=-1).astype(np.float32)

print("encoder_train_input:", encoder_train_input.shape)
print("condition_train:", condition_train.shape)
print("cvae_y_train:", cvae_y_train.shape)


# ============================================================
# 8. SAMPLING LAYER
# ============================================================

class Sampling(layers.Layer):
    """
    VAE reparameterization trick.

    Instead of sampling z directly from N(mean, variance),
    we sample epsilon from N(0,1) and compute:

        z = mean + exp(0.5 * log_var) * epsilon

    This keeps the model differentiable.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


# ============================================================
# 9. BUILD RESIDUAL CONDITIONAL VAE
# ============================================================

def cvae_conv_block(x, filters, name):
    """
    Small convolutional block used inside the residual-cVAE.
    """
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    return x


def build_residual_cvae(H, W, latent_dim=16):
    """
    Residual Conditional VAE.

    Encoder:
        input = [base_prob, residual, valid_mask]
        output = latent distribution z_mean, z_log_var

    Decoder:
        input = [base_prob, sampled_z]
        output = residual correction map

    The decoder learns possible residual scenarios conditioned
    on the deterministic UNet risk map.
    """

    # -----------------------------
    # Encoder
    # -----------------------------
    enc_in = Input(shape=(H, W, 3), name="encoder_input_base_residual_mask")

    x = cvae_conv_block(enc_in, 32, "enc1")
    x = layers.MaxPooling2D(name="enc_pool1")(x)

    x = cvae_conv_block(x, 64, "enc2")
    x = layers.MaxPooling2D(name="enc_pool2")(x)

    x = cvae_conv_block(x, 128, "enc3")
    x = layers.GlobalAveragePooling2D(name="enc_gap")(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    z = Sampling(name="latent_sampling")([z_mean, z_log_var])

    encoder = Model(
        enc_in,
        [z_mean, z_log_var, z],
        name="residual_cvae_encoder"
    )

    # -----------------------------
    # Decoder
    # -----------------------------
    cond_in = Input(shape=(H, W, 1), name="condition_base_probability")
    z_in = Input(shape=(latent_dim,), name="latent_z")

    z_dense = layers.Dense(H * W * 8, activation="relu", name="z_dense")(z_in)
    z_map = layers.Reshape((H, W, 8), name="z_map")(z_dense)

    d = layers.Concatenate(name="decoder_concat_condition_z")([cond_in, z_map])

    d = cvae_conv_block(d, 64, "dec1")
    d = cvae_conv_block(d, 32, "dec2")

    # Residual can be negative or positive.
    # tanh restricts residual output to [-1, 1].
    residual_out = layers.Conv2D(
        1,
        1,
        activation="tanh",
        name="residual_output"
    )(d)

    decoder = Model(
        [cond_in, z_in],
        residual_out,
        name="residual_cvae_decoder"
    )

    # -----------------------------
    # Full CVAE
    # -----------------------------
    z_mean_out, z_log_var_out, z_sample = encoder(enc_in)
    residual_pred = decoder([cond_in, z_sample])

    cvae = Model(
        [enc_in, cond_in],
        residual_pred,
        name="ResidualConditionalVAE"
    )

    return cvae, encoder, decoder


cvae, cvae_encoder, cvae_decoder = build_residual_cvae(
    H=H,
    W=W,
    latent_dim=LATENT_DIM
)

cvae.summary()


# ============================================================
# 10. MASKED CVAE MODEL WITH CUSTOM TRAIN_STEP
# ============================================================
# We use a subclassed model so the KL loss is handled correctly.
# This is safer than trying to access encoder outputs inside a plain
# Keras loss function.

class ResidualCVAETrainer(tf.keras.Model):
    """
    Wrapper model for training residual-cVAE with:
    - masked reconstruction loss
    - KL divergence
    """

    def __init__(self, cvae, encoder, decoder, beta_kl=1e-4):
        super().__init__()
        self.cvae = cvae
        self.encoder = encoder
        self.decoder = decoder
        self.beta_kl = beta_kl

        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.recon_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.recon_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=False):
        return self.cvae(inputs, training=training)

    def train_step(self, data):
        inputs, y_true_with_mask = data
        enc_input, cond_input = inputs

        true_residual = y_true_with_mask[..., 0:1]
        mask = y_true_with_mask[..., 1:2]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(enc_input, training=True)
            pred_residual = self.decoder([cond_input, z], training=True)

            recon = tf.square(true_residual - pred_residual) * mask
            recon_loss = tf.reduce_sum(recon) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())

            kl_loss = -0.5 * tf.reduce_mean(
                1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )

            total_loss = recon_loss + self.beta_kl * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        inputs, y_true_with_mask = data
        enc_input, cond_input = inputs

        true_residual = y_true_with_mask[..., 0:1]
        mask = y_true_with_mask[..., 1:2]

        z_mean, z_log_var, z = self.encoder(enc_input, training=False)
        pred_residual = self.decoder([cond_input, z], training=False)

        recon = tf.square(true_residual - pred_residual) * mask
        recon_loss = tf.reduce_sum(recon) / (tf.reduce_sum(mask) + tf.keras.backend.epsilon())

        kl_loss = -0.5 * tf.reduce_mean(
            1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )

        total_loss = recon_loss + self.beta_kl * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.recon_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.recon_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


trainer = ResidualCVAETrainer(
    cvae=cvae,
    encoder=cvae_encoder,
    decoder=cvae_decoder,
    beta_kl=BETA_KL
)

trainer.compile(
    optimizer=tf.keras.optimizers.Adam(CVAE_LR)
)


# ============================================================
# 11. TRAIN RESIDUAL CVAE
# ============================================================

callbacks_cvae = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=8,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
]

history_cvae = trainer.fit(
    [encoder_train_input, condition_train],
    cvae_y_train,
    validation_data=([encoder_val_input, condition_val], cvae_y_val),
    epochs=CVAE_EPOCHS,
    batch_size=CVAE_BATCH_SIZE,
    callbacks=callbacks_cvae,
    verbose=1
)


# ============================================================
# 12. MONTE-CARLO SAMPLING
# ============================================================
# This is where the elevated model becomes probabilistic.
#
# For each test sample:
#   1. sample random latent vector z
#   2. generate residual correction map
#   3. add residual to base UNet probability
#   4. clip to [0,1]
#
# Repeating this N times gives:
#   mean map = final risk map
#   std map  = uncertainty map

def sample_cvae_predictions(base_prob, decoder, n_samples=30, latent_dim=16, batch_size=1):
    all_preds = []
    n = base_prob.shape[0]

    for k in range(n_samples):
        print(f"Sampling scenario {k + 1}/{n_samples}")

        sampled_batches = []

        for start in range(0, n, batch_size):
            end = start + batch_size
            cond_batch = base_prob[start:end]

            z = np.random.normal(
                size=(cond_batch.shape[0], latent_dim)
            ).astype(np.float32)

            residual_sample = decoder.predict(
                [cond_batch, z],
                verbose=0
            )

            prob_sample = np.clip(
                cond_batch + residual_sample,
                0.0,
                1.0
            )

            sampled_batches.append(prob_sample)

        all_preds.append(np.concatenate(sampled_batches, axis=0))

    all_preds = np.stack(all_preds, axis=0)

    mean_prob = all_preds.mean(axis=0)
    std_prob = all_preds.std(axis=0)

    return mean_prob, std_prob, all_preds


cvae_test_mean, cvae_test_std, cvae_test_samples = sample_cvae_predictions(
    base_prob=base_test_prob,
    decoder=cvae_decoder,
    n_samples=N_SAMPLES,
    latent_dim=LATENT_DIM,
    batch_size=1
)

print("cvae_test_mean:", cvae_test_mean.shape)
print("cvae_test_std:", cvae_test_std.shape)


# ============================================================
# 13. MASKED EVALUATION
# ============================================================
# Evaluation is done only over valid pixels, exactly like the improved UNet.

def masked_flatten(y_true, y_prob, mask):
    m = mask.reshape(-1).astype(bool)
    return y_true.reshape(-1)[m], y_prob.reshape(-1)[m]


def evaluate_masked_prob(name, y_true, y_prob, mask, threshold):
    yt, yp = masked_flatten(y_true, y_prob, mask)

    pr_auc = average_precision_score(yt, yp)
    roc_auc = roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else np.nan

    ypred = (yp >= threshold).astype(np.uint8)

    cm = confusion_matrix(yt, ypred, labels=[0, 1])
    precision = precision_score(yt, ypred, zero_division=0)
    recall = recall_score(yt, ypred, zero_division=0)
    f1 = f1_score(yt, ypred, zero_division=0)

    print("\n" + "=" * 50)
    print(name)
    print(f"Valid pixels: {len(yt):,}   positives: {int(yt.sum()):,}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Precision: {precision:.4f}   Recall: {recall:.4f}   F1: {f1:.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(cm)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": None if np.isnan(roc_auc) else float(roc_auc),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cm": cm.tolist(),
        "n_valid": int(len(yt)),
        "n_pos": int(yt.sum())
    }


# Base deterministic UNet evaluation.
base_test_metrics = evaluate_masked_prob(
    name="BASE IMPROVED UNET TEST",
    y_true=Y_test,
    y_prob=base_test_prob,
    mask=M_test,
    threshold=BASE_THRESHOLD
)

# Elevated model evaluation.
elevated_test_metrics = evaluate_masked_prob(
    name="ELEVATED UNET + RESIDUAL CVAE TEST",
    y_true=Y_test,
    y_prob=cvae_test_mean,
    mask=M_test,
    threshold=BASE_THRESHOLD
)


# ============================================================
# 14. UNCERTAINTY VS ERROR DIAGNOSTIC
# ============================================================
# This checks whether the uncertainty estimate is meaningful.
# If uncertainty is useful, high uncertainty should correlate
# with larger absolute prediction error.

yt_valid, yp_valid = masked_flatten(Y_test, cvae_test_mean, M_test)
_, std_valid = masked_flatten(Y_test, cvae_test_std, M_test)

abs_error = np.abs(yt_valid - yp_valid)

if len(std_valid) > 1:
    uncertainty_error_corr = np.corrcoef(std_valid, abs_error)[0, 1]
else:
    uncertainty_error_corr = np.nan

print("\nUncertainty vs absolute error correlation:", uncertainty_error_corr)

plt.figure(figsize=(6, 5))
plt.scatter(std_valid, abs_error, s=4, alpha=0.25)
plt.xlabel("Uncertainty std")
plt.ylabel("|y - p|")
plt.title(f"Uncertainty vs absolute error corr={uncertainty_error_corr:.3f}")
plt.tight_layout()
plt.savefig("elevated_unet_uncertainty_vs_error.png", dpi=150)
plt.show()


# ============================================================
# 15. ELEVATED HEATMAPS
# ============================================================
# These figures are for thesis/presentation.
#
# Each sample shows:
#   1. Ground truth fire map
#   2. Base UNet probability map
#   3. Elevated mean risk map
#   4. Elevated uncertainty map

def plot_elevated_maps(sample_idx, save_prefix="elevated_unet_maps"):
    gt = Y_test[sample_idx, :, :, 0]
    mask = M_test[sample_idx, :, :, 0]

    base_map = base_test_prob[sample_idx, :, :, 0]
    mean_map = cvae_test_mean[sample_idx, :, :, 0]
    std_map = cvae_test_std[sample_idx, :, :, 0]

    gt_display = np.where(mask == 1, gt, np.nan)
    base_display = np.where(mask == 1, base_map, np.nan)
    mean_display = np.where(mask == 1, mean_map, np.nan)
    std_display = np.where(mask == 1, std_map, np.nan)

    cmap_risk = plt.cm.YlOrRd.copy()
    cmap_risk.set_bad(color="lightgrey")

    cmap_unc = plt.cm.viridis.copy()
    cmap_unc.set_bad(color="lightgrey")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(gt_display, vmin=0, vmax=1, cmap=cmap_risk, origin="lower")
    axes[0].set_title("Ground truth")
    axes[0].set_xlabel("lon grid")
    axes[0].set_ylabel("lat grid")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(base_display, vmin=0, vmax=1, cmap=cmap_risk, origin="lower")
    axes[1].set_title("Base UNet probability")
    axes[1].set_xlabel("lon grid")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(mean_display, vmin=0, vmax=1, cmap=cmap_risk, origin="lower")
    axes[2].set_title("Elevated mean risk")
    axes[2].set_xlabel("lon grid")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(std_display, cmap=cmap_unc, origin="lower")
    axes[3].set_title("Uncertainty std")
    axes[3].set_xlabel("lon grid")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.suptitle(f"Elevated UNet - test sample {sample_idx}")
    plt.tight_layout()

    out_path = f"{save_prefix}_{sample_idx}.png"
    plt.savefig(out_path, dpi=150)
    plt.show()

    print("Saved:", out_path)


fire_counts = (Y_test[:, :, :, 0] * M_test[:, :, :, 0]).sum(axis=(1, 2))
candidate_samples = np.where(fire_counts > 0)[0]

print("Candidate fire samples:", candidate_samples[:10])

if len(candidate_samples) >= 2:
    plot_elevated_maps(int(candidate_samples[0]))
    plot_elevated_maps(int(candidate_samples[1]))
elif len(candidate_samples) == 1:
    plot_elevated_maps(int(candidate_samples[0]))
else:
    print("No fire samples found in test set. Plotting sample 0.")
    plot_elevated_maps(0)


# ============================================================
# 16. SAVE CVAE, OUTPUTS, AND METADATA
# ============================================================

cvae.save(CVAE_MODEL_PATH)

np.savez_compressed(
    OUTPUT_NPZ_PATH,
    base_test_prob=base_test_prob,
    cvae_test_mean=cvae_test_mean,
    cvae_test_std=cvae_test_std,
    Y_test=Y_test,
    M_test=M_test,
    cvae_test_samples=cvae_test_samples
)

cvae_meta = {
    "model_type": "Elevated ImprovedUNet with residual conditional VAE",
    "base_unet_model": BASE_UNET_MODEL_PATH,
    "base_unet_meta": BASE_UNET_META_PATH,
    "cvae_model": CVAE_MODEL_PATH,
    "latent_dim": int(LATENT_DIM),
    "beta_kl": float(BETA_KL),
    "n_samples": int(N_SAMPLES),
    "threshold_used": float(BASE_THRESHOLD),
    "base_test_metrics": base_test_metrics,
    "elevated_test_metrics": elevated_test_metrics,
    "uncertainty_error_corr": None if np.isnan(uncertainty_error_corr) else float(uncertainty_error_corr),
    "outputs_npz": OUTPUT_NPZ_PATH,
    "description": (
        "The base ImprovedUNet is frozen and used as a deterministic risk estimator. "
        "A residual conditional VAE is trained on residual maps y_true - p_unet. "
        "At inference, multiple residual maps are sampled and added to the base UNet "
        "probability map. The average gives the elevated risk map and the standard "
        "deviation gives the uncertainty map."
    )
}

with open(CVAE_META_PATH, "w", encoding="utf-8") as f:
    json.dump(cvae_meta, f, indent=2)

print("\nSaved:")
print(CVAE_MODEL_PATH)
print(CVAE_META_PATH)
print(OUTPUT_NPZ_PATH)

print("\nDone.")

# %%
