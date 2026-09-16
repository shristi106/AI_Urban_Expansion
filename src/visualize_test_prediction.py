import os
import io
import numpy as np
import polars as pl
import torch
import matplotlib.pyplot as plt

from PIL import Image
from model_hybrid import HybridSiameseCNNViT


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

MODEL_PATH = "hybrid_siamese_vit.pth"

PATCH_SIZE = 128
STRIDE = 128

THRESHOLD = 0.5

HF_DATA_DIR = (
    r"C:\Users\SHRISTI\.cache\huggingface\hub"
    r"\datasets--blanchon--OSCD_MSI"
    r"\snapshots"
    r"\16651ca137f0b4577801cf4413c77453d829ec5a"
    r"\data"
)

TEST_FILE = os.path.join(
    HF_DATA_DIR,
    "test-00000-of-00001.parquet"
)

OUTPUT_FILE = "best_test_scene_result.png"


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained Hybrid CNN + ViT model...")

model = HybridSiameseCNNViT(
    in_channels=13
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading untouched OSCD test scenes...")

df = pl.read_parquet(TEST_FILE)

print(f"Test scenes: {len(df)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize(image):

    image = image.astype(np.float32)

    image = image / 10000.0

    image = np.clip(
        image,
        0.0,
        1.0
    )

    return image


def decode_mask(mask_struct):

    mask_bytes = mask_struct["bytes"]

    image = Image.open(
        io.BytesIO(mask_bytes)
    )

    mask = np.array(image)

    return (
        mask > 0
    ).astype(np.uint8)


def make_rgb(image):

    # Sentinel-2:
    # B4 = Red
    # B3 = Green
    # B2 = Blue

    rgb = np.stack(
        [
            image[3],
            image[2],
            image[1]
        ],
        axis=-1
    )

    low = np.percentile(
        rgb,
        2
    )

    high = np.percentile(
        rgb,
        98
    )

    rgb = (
        rgb - low
    ) / (
        high - low + 1e-8
    )

    return np.clip(
        rgb,
        0,
        1
    )


# ============================================================
# PREDICT COMPLETE SCENE
# ============================================================

def predict_scene(image1, image2):

    channels, height, width = image1.shape

    prediction_sum = np.zeros(
        (height, width),
        dtype=np.float32
    )

    prediction_count = np.zeros(
        (height, width),
        dtype=np.float32
    )

    with torch.no_grad():

        for y in range(
            0,
            height - PATCH_SIZE + 1,
            STRIDE
        ):

            for x in range(
                0,
                width - PATCH_SIZE + 1,
                STRIDE
            ):

                patch1 = image1[
                    :,
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                patch2 = image2[
                    :,
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                tensor1 = torch.from_numpy(
                    patch1
                ).unsqueeze(0).float()

                tensor2 = torch.from_numpy(
                    patch2
                ).unsqueeze(0).float()

                output = model(
                    tensor1,
                    tensor2
                )

                probability = torch.sigmoid(
                    output
                )[0, 0].numpy()

                prediction_sum[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ] += probability

                prediction_count[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ] += 1

    prediction_count[
        prediction_count == 0
    ] = 1

    probability = (
        prediction_sum
        /
        prediction_count
    )

    prediction = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    return probability, prediction


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(
        prediction,
        target
    ).sum()

    tn = np.logical_and(
        ~prediction,
        ~target
    ).sum()

    fp = np.logical_and(
        prediction,
        ~target
    ).sum()

    fn = np.logical_and(
        ~prediction,
        target
    ).sum()

    epsilon = 1e-8

    accuracy = (
        tp + tn
    ) / (
        tp + tn + fp + fn + epsilon
    )

    precision = (
        tp
    ) / (
        tp + fp + epsilon
    )

    recall = (
        tp
    ) / (
        tp + fn + epsilon
    )

    f1 = (
        2 * precision * recall
    ) / (
        precision + recall + epsilon
    )

    iou = (
        tp
    ) / (
        tp + fp + fn + epsilon
    )

    dice = (
        2 * tp
    ) / (
        2 * tp + fp + fn + epsilon
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        iou,
        dice
    )


# ============================================================
# FIND BEST TEST SCENE
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING ALL TEST SCENES")
print("=" * 60)

scene_data = []

best_scene = None
best_f1 = -1


for scene_index in range(len(df)):

    print(
        f"\nProcessing scene "
        f"{scene_index + 1}/{len(df)}..."
    )

    row = df.row(
        scene_index,
        named=True
    )

    image1 = np.array(
        row["image1"],
        dtype=np.uint16
    )

    image2 = np.array(
        row["image2"],
        dtype=np.uint16
    )

    mask = decode_mask(
        row["mask"]
    )

    image1 = normalize(image1)
    image2 = normalize(image2)

    probability, prediction = predict_scene(
        image1,
        image2
    )

    # Match dimensions
    h = min(
        prediction.shape[0],
        mask.shape[0]
    )

    w = min(
        prediction.shape[1],
        mask.shape[1]
    )

    prediction = prediction[
        :h,
        :w
    ]

    mask = mask[
        :h,
        :w
    ]

    image1 = image1[
        :,
        :h,
        :w
    ]

    image2 = image2[
        :,
        :h,
        :w
    ]

    metrics = calculate_metrics(
        prediction,
        mask
    )

    (
        accuracy,
        precision,
        recall,
        f1,
        iou,
        dice
    ) = metrics

    print(
        f"F1: {f1:.4f} | "
        f"IoU: {iou:.4f}"
    )

    current_data = {
        "scene": scene_index + 1,
        "image1": image1,
        "image2": image2,
        "mask": mask,
        "prediction": prediction,
        "probability": probability[
            :h,
            :w
        ],
        "metrics": metrics
    }

    scene_data.append(
        current_data
    )

    if f1 > best_f1:

        best_f1 = f1
        best_scene = current_data


# ============================================================
# BEST SCENE INFORMATION
# ============================================================

(
    accuracy,
    precision,
    recall,
    f1,
    iou,
    dice
) = best_scene["metrics"]


print("\n" + "=" * 60)
print("BEST TEST SCENE")
print("=" * 60)

print(
    f"Scene       : {best_scene['scene']}"
)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Recall      : {recall:.4f}"
)

print(
    f"F1 Score    : {f1:.4f}"
)

print(
    f"IoU         : {iou:.4f}"
)

print(
    f"Dice        : {dice:.4f}"
)

print("=" * 60)


# ============================================================
# PREPARE VISUALS
# ============================================================

image1 = best_scene["image1"]
image2 = best_scene["image2"]

mask = best_scene["mask"]
prediction = best_scene["prediction"]


rgb1 = make_rgb(image1)
rgb2 = make_rgb(image2)


# ============================================================
# CREATE OVERLAY
# ============================================================

overlay = rgb2.copy()

# Highlight predicted changes
change_pixels = prediction == 1

overlay[change_pixels] = [
    1.0,
    0.0,
    0.0
]


# ============================================================
# CREATE FINAL FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    5,
    figsize=(20, 4.5)
)


axes[0].imshow(rgb1)

axes[0].set_title(
    "T1 Satellite Image",
    fontsize=13
)

axes[0].axis("off")


axes[1].imshow(rgb2)

axes[1].set_title(
    "T2 Satellite Image",
    fontsize=13
)

axes[1].axis("off")


axes[2].imshow(mask)

axes[2].set_title(
    "Ground Truth Change",
    fontsize=13
)

axes[2].axis("off")


axes[3].imshow(prediction)

axes[3].set_title(
    "AI Predicted Change",
    fontsize=13
)

axes[3].axis("off")


axes[4].imshow(overlay)

axes[4].set_title(
    "Change Overlay",
    fontsize=13
)

axes[4].axis("off")


fig.suptitle(
    "AI-Based Satellite Urban Expansion and Land-Use Change Detection",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print(
    f"\nFinal visualization saved as:"
)

print(
    OUTPUT_FILE
)