# %%
# ============================================================
# ELEVATED UNET OCCLUSION SENSITIVITY
# ============================================================
#
# Standalone script.
#
# Loads:
#   - unet_fire_improved.keras
#   - unet_fire_improved_meta.json
#   - elevated_unet_outputs.npz
#
# Produces:
#   - elevated_unet_occlusion_sensitivity.png
#
# What it explains:
#   1. Base UNet probability sensitivity
#   2. Elevated mean risk sensitivity
#   3. Uncertainty sensitivity
#
# ============================================================

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_UNET_MODEL_PATH = "unet_fire_improved.keras"
BASE_UNET_META_PATH = "unet_fire_improved_meta.json"
ELEVATED_OUTPUTS_PATH = "elevated_unet_outputs.npz"

SAMPLE_INDEX = 0

PATCH_SIZE = 3
STRIDE = 1
BASELINE_VALUE = 0.0


# ============================================================
# 2. LOAD META
# ============================================================

with open(BASE_UNET_META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

H = int(meta["H"])
W = int(meta["W"])
Cin = int(meta["Cin"])

print("Loaded metadata.")
print("Grid:", H, "x", W)
print("Input channels:", Cin)


# ============================================================
# 3. CUSTOM PAD FUNCTION FOR SAVED UNET
# ============================================================
# Your UNet was saved with a Lambda layer called pad_hw.
# This function is needed so Keras can load it.

@tf.keras.utils.register_keras_serializable()
def pad_hw(x):
    factor = 16

    h = x.shape[1]
    w = x.shape[2]

    pad_h = (factor - (h % factor)) % factor
    pad_w = (factor - (w % factor)) % factor

    return tf.pad(
        x,
        [[0, 0], [0, pad_h], [0, pad_w], [0, 0]]
    )


# ============================================================
# 4. LOAD BASE UNET
# ============================================================

unet = tf.keras.models.load_model(
    BASE_UNET_MODEL_PATH,
    compile=False,
    custom_objects={"pad_hw": pad_hw}
)

unet.trainable = False

print("Loaded base Improved UNet.")


# ============================================================
# 5. LOAD ELEVATED OUTPUTS
# ============================================================
# This file was saved by the elevated residual-cVAE script.
#
# Expected arrays:
#   base_test_prob
#   cvae_test_mean
#   cvae_test_std
#   Y_test
#   M_test
#
# IMPORTANT:
# This script uses saved elevated outputs for visualization.
# For true full recomputation-based occlusion on elevated model,
# you also need the cVAE decoder model loaded separately.

data = np.load(ELEVATED_OUTPUTS_PATH)

base_test_prob = data["base_test_prob"]
cvae_test_mean = data["cvae_test_mean"]
cvae_test_std = data["cvae_test_std"]
Y_test = data["Y_test"]
M_test = data["M_test"]

print("Loaded elevated outputs.")
print("base_test_prob:", base_test_prob.shape)
print("cvae_test_mean:", cvae_test_mean.shape)
print("cvae_test_std:", cvae_test_std.shape)
print("Y_test:", Y_test.shape)
print("M_test:", M_test.shape)


# ============================================================
# 6. LOAD OR CREATE X_test_u
# ============================================================
# If you already have X_test_u in memory, comment this block.
#
# If not, this script cannot recompute true occlusion from inputs
# because elevated_unet_outputs.npz does not contain X_test_u.
#
# The elevated script can be modified to save X_test_u too.
# For now, we check if it exists in the npz.

if "X_test_u" in data.files:
    X_test_u = data["X_test_u"]
    print("Loaded X_test_u from npz:", X_test_u.shape)
else:
    raise ValueError(
        "X_test_u was not found in elevated_unet_outputs.npz.\n"
        "To run occlusion, save X_test_u in the elevated script:\n\n"
        "np.savez_compressed(\n"
        "    'elevated_unet_outputs.npz',\n"
        "    base_test_prob=base_test_prob,\n"
        "    cvae_test_mean=cvae_test_mean,\n"
        "    cvae_test_std=cvae_test_std,\n"
        "    Y_test=Y_test,\n"
        "    M_test=M_test,\n"
        "    X_test_u=X_test_u\n"
        ")\n"
    )


# ============================================================
# 7. BASE UNET OCCLUSION
# ============================================================
# This measures how much the deterministic UNet prediction changes
# when a local spatial patch is hidden.

x0 = X_test_u[SAMPLE_INDEX:SAMPLE_INDEX + 1].copy()

y0 = Y_test[SAMPLE_INDEX, :, :, 0]
mask0 = M_test[SAMPLE_INDEX, :, :, 0]

base_original = unet.predict(x0, verbose=0)[0, :, :, 0]

base_occ_map = np.zeros((H, W), dtype=np.float32)
count_map = np.zeros((H, W), dtype=np.float32)

print("Running base UNet occlusion...")

for i in range(0, H - PATCH_SIZE + 1, STRIDE):
    for j in range(0, W - PATCH_SIZE + 1, STRIDE):

        x_occ = x0.copy()

        x_occ[
            :,
            i:i + PATCH_SIZE,
            j:j + PATCH_SIZE,
            :
        ] = BASELINE_VALUE

        base_occ = unet.predict(x_occ, verbose=0)[0, :, :, 0]

        diff = np.abs(base_original - base_occ)

        base_occ_map[
            i:i + PATCH_SIZE,
            j:j + PATCH_SIZE
        ] += diff[
            i:i + PATCH_SIZE,
            j:j + PATCH_SIZE
        ]

        count_map[
            i:i + PATCH_SIZE,
            j:j + PATCH_SIZE
        ] += 1

count_map[count_map == 0] = 1
base_occ_map = base_occ_map / count_map

print("Base occlusion done.")


# ============================================================
# 8. ELEVATED OUTPUT DIFFERENCE MAPS
# ============================================================
# Since cVAE decoder is not loaded here, we cannot recompute
# elevated mean/std under each occlusion.
#
# Instead, we produce comparison maps:
#   - difference between elevated mean risk and base UNet
#   - uncertainty std
#
# These are useful for presentation, but they are not true
# elevated occlusion. For true elevated occlusion, use the second
# full version below.

base_saved = base_test_prob[SAMPLE_INDEX, :, :, 0]
mean_saved = cvae_test_mean[SAMPLE_INDEX, :, :, 0]
std_saved = cvae_test_std[SAMPLE_INDEX, :, :, 0]

elevated_delta = np.abs(mean_saved - base_saved)


# ============================================================
# 9. APPLY VALID MASK
# ============================================================

gt_display = np.where(mask0 == 1, y0, np.nan)
base_display = np.where(mask0 == 1, base_saved, np.nan)
mean_display = np.where(mask0 == 1, mean_saved, np.nan)
std_display = np.where(mask0 == 1, std_saved, np.nan)

base_occ_display = np.where(mask0 == 1, base_occ_map, np.nan)
delta_display = np.where(mask0 == 1, elevated_delta, np.nan)


# ============================================================
# 10. PLOTS
# ============================================================

cmap_risk = plt.cm.YlOrRd.copy()
cmap_risk.set_bad(color="lightgrey")

cmap_hot = plt.cm.hot.copy()
cmap_hot.set_bad(color="lightgrey")

cmap_unc = plt.cm.viridis.copy()
cmap_unc.set_bad(color="lightgrey")

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

im0 = axes[0, 0].imshow(gt_display, vmin=0, vmax=1, cmap=cmap_risk, origin="lower")
axes[0, 0].set_title("Ground truth")
plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

im1 = axes[0, 1].imshow(base_display, vmin=0, vmax=1, cmap=cmap_risk, origin="lower")
axes[0, 1].set_title("Base UNet probability")
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

im2 = axes[0, 2].imshow(mean_display, vmin=0, vmax=1, cmap=cmap_risk, origin="lower")
axes[0, 2].set_title("Elevated mean risk")
plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

im3 = axes[1, 0].imshow(base_occ_display, cmap=cmap_hot, origin="lower")
axes[1, 0].set_title("Occlusion sensitivity: Base UNet")
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

im4 = axes[1, 1].imshow(delta_display, cmap=cmap_hot, origin="lower")
axes[1, 1].set_title("|Elevated mean - Base probability|")
plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

im5 = axes[1, 2].imshow(std_display, cmap=cmap_unc, origin="lower")
axes[1, 2].set_title("Elevated uncertainty std")
plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

for ax in axes.ravel():
    ax.set_xlabel("lon grid")
    ax.set_ylabel("lat grid")

plt.suptitle(f"Elevated UNet explanation maps - sample {SAMPLE_INDEX}")
plt.tight_layout()
plt.savefig("elevated_unet_occlusion_sensitivity.png", dpi=150)
plt.show()

print("Saved: elevated_unet_occlusion_sensitivity.png")
# %%
