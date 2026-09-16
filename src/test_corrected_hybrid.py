import os
import sys
import io
import numpy as np
import polars as pl
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_hybrid import HybridSiameseCNNViT


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "hybrid_siamese_vit_corrected.pth"
)

CACHE_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache",
    "huggingface",
    "hub",
    "datasets--blanchon--OSCD_MSI",
    "snapshots",
    "16651ca137f0b4577801cf4413c77453d829ec5a",
    "data"
)

TEST_PARQUET = os.path.join(
    CACHE_DIR,
    "test-00000-of-00001.parquet"
)

PATCH_SIZE = 128
STRIDE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(pred, target):

    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)

    tp = np.logical_and(
        pred == 1,
        target == 1
    ).sum()

    tn = np.logical_and(
        pred == 0,
        target == 0
    ).sum()

    fp = np.logical_and(
        pred == 1,
        target == 0
    ).sum()

    fn = np.logical_and(
        pred == 0,
        target == 1
    ).sum()

    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn + 1e-8)
    )

    precision = (
        tp
        /
        (tp + fp + 1e-8)
    )

    recall = (
        tp
        /
        (tp + fn + 1e-8)
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall + 1e-8)
    )

    iou = (
        tp
        /
        (tp + fp + fn + 1e-8)
    )

    dice = (
        2 * tp
        /
        (
            2 * tp
            + fp
            + fn
            + 1e-8
        )
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        iou,
        dice,
        tp,
        tn,
        fp,
        fn
    )


# ============================================================
# LOAD OSCD SCENE
# ============================================================

def load_scene(row):

    image1 = np.asarray(
        row["image1"],
        dtype=np.float32
    )

    image2 = np.asarray(
        row["image2"],
        dtype=np.float32
    )

    mask_struct = row["mask"]

    mask_bytes = mask_struct["bytes"]

    mask = np.array(
        Image.open(
            io.BytesIO(mask_bytes)
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # SAME NORMALIZATION AS TRAINING
    # --------------------------------------------------------

    image1 = image1 / 10000.0
    image2 = image2 / 10000.0

    image1 = np.clip(
        image1,
        0.0,
        1.0
    )

    image2 = np.clip(
        image2,
        0.0,
        1.0
    )

    mask = (
        mask > 0
    ).astype(np.uint8)

    return image1, image2, mask


# ============================================================
# PAD SCENE
# ============================================================

def pad_scene(image1, image2, mask):

    channels, height, width = image1.shape

    padded_height = (
        int(np.ceil(height / PATCH_SIZE))
        * PATCH_SIZE
    )

    padded_width = (
        int(np.ceil(width / PATCH_SIZE))
        * PATCH_SIZE
    )

    padded_image1 = np.zeros(
        (
            channels,
            padded_height,
            padded_width
        ),
        dtype=np.float32
    )

    padded_image2 = np.zeros(
        (
            channels,
            padded_height,
            padded_width
        ),
        dtype=np.float32
    )

    padded_mask = np.zeros(
        (
            padded_height,
            padded_width
        ),
        dtype=np.uint8
    )

    padded_image1[
        :,
        :height,
        :width
    ] = image1

    padded_image2[
        :,
        :height,
        :width
    ] = image2

    padded_mask[
        :height,
        :width
    ] = mask

    return (
        padded_image1,
        padded_image2,
        padded_mask,
        height,
        width
    )


# ============================================================
# PREDICT ONE SCENE
# ============================================================

def predict_scene(
    model,
    image1,
    image2
):

    (
        image1,
        image2,
        _,
        original_height,
        original_width
    ) = pad_scene(
        image1,
        image2,
        np.zeros(
            image1.shape[1:],
            dtype=np.uint8
        )
    )

    padded_height = image1.shape[1]
    padded_width = image1.shape[2]

    # --------------------------------------------------------
    # Prediction probability map
    # --------------------------------------------------------

    probability_map = np.zeros(
        (
            padded_height,
            padded_width
        ),
        dtype=np.float32
    )

    count_map = np.zeros(
        (
            padded_height,
            padded_width
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Number of patches
    # --------------------------------------------------------

    total_rows = (
        padded_height // PATCH_SIZE
    )

    total_cols = (
        padded_width // PATCH_SIZE
    )

    total_patches = (
        total_rows * total_cols
    )

    processed = 0

    # --------------------------------------------------------
    # Patch-by-patch inference
    # --------------------------------------------------------

    with torch.no_grad():

        for y in range(
            0,
            padded_height,
            STRIDE
        ):

            for x in range(
                0,
                padded_width,
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

                # Both are guaranteed to be 128x128
                tensor1 = torch.from_numpy(
                    patch1
                ).unsqueeze(0).to(DEVICE)

                tensor2 = torch.from_numpy(
                    patch2
                ).unsqueeze(0).to(DEVICE)

                output = model(
                    tensor1,
                    tensor2
                )

                probability = torch.sigmoid(
                    output
                )

                probability = (
                    probability
                    .squeeze()
                    .cpu()
                    .numpy()
                )

                probability_map[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ] += probability

                count_map[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ] += 1.0

                processed += 1

                print(
                    f"\r      Patches: "
                    f"{processed}/{total_patches}",
                    end=""
                )

    print()

    # --------------------------------------------------------
    # Average overlapping predictions
    # --------------------------------------------------------

    probability_map = (
        probability_map
        /
        np.maximum(
            count_map,
            1.0
        )
    )

    # Remove padding
    probability_map = probability_map[
        :original_height,
        :original_width
    ]

    return probability_map


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CORRECTED HYBRID MODEL")
    print("UNTOUCHED OSCD TEST-SCENE EVALUATION")
    print("=" * 70)

    print(f"Device       : {DEVICE}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Test data    : {TEST_PARQUET}")
    print(f"Patch size   : {PATCH_SIZE} x {PATCH_SIZE}")
    print(f"Stride       : {STRIDE}")

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(
        TEST_PARQUET
    ):

        raise FileNotFoundError(
            f"\nTest Parquet not found:\n{TEST_PARQUET}"
        )

    # ========================================================
    # LOAD TEST DATA
    # ========================================================

    print(
        "\nLoading cached OSCD test scenes..."
    )

    df = pl.read_parquet(
        TEST_PARQUET
    )

    print(
        f"Number of untouched test scenes: "
        f"{df.height}"
    )

    if df.height != 10:

        print(
            "\nWARNING: Expected 10 test scenes."
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading corrected model..."
    )

    model = HybridSiameseCNNViT(
        in_channels=13
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)
    model.eval()

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # OVERALL COUNTS
    # ========================================================

    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    scene_results = []

    # ========================================================
    # PROCESS EACH TEST SCENE
    # ========================================================

    for scene_idx in range(
        df.height
    ):

        print()
        print("-" * 70)

        print(
            f"TEST SCENE {scene_idx + 1}/{df.height}"
        )

        print("-" * 70)

        row = df.row(
            scene_idx,
            named=True
        )

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        image1, image2, mask = load_scene(
            row
        )

        print(
            f"T1 shape : {image1.shape}"
        )

        print(
            f"T2 shape : {image2.shape}"
        )

        print(
            f"Mask     : {mask.shape}"
        )

        # ----------------------------------------------------
        # Predict using 128x128 tiles
        # ----------------------------------------------------

        print(
            "\n      Running tiled inference..."
        )

        probability_map = predict_scene(
            model,
            image1,
            image2
        )

        # ----------------------------------------------------
        # Convert probability to binary mask
        # ----------------------------------------------------

        prediction = (
            probability_map >= 0.5
        ).astype(np.uint8)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        (
            accuracy,
            precision,
            recall,
            f1,
            iou,
            dice,
            tp,
            tn,
            fp,
            fn
        ) = calculate_metrics(
            prediction,
            mask
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        scene_results.append(
            (
                accuracy,
                precision,
                recall,
                f1,
                iou,
                dice
            )
        )

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        print(
            f"\nScene {scene_idx + 1:02d} Results:"
        )

        print(
            f"Accuracy  : {accuracy:.4f} "
            f"({accuracy * 100:.2f}%)"
        )

        print(
            f"Precision : {precision:.4f} "
            f"({precision * 100:.2f}%)"
        )

        print(
            f"Recall    : {recall:.4f} "
            f"({recall * 100:.2f}%)"
        )

        print(
            f"F1 Score  : {f1:.4f} "
            f"({f1 * 100:.2f}%)"
        )

        print(
            f"IoU       : {iou:.4f} "
            f"({iou * 100:.2f}%)"
        )

        print(
            f"Dice      : {dice:.4f} "
            f"({dice * 100:.2f}%)"
        )

    # ========================================================
    # OVERALL PIXEL-LEVEL METRICS
    # ========================================================

    accuracy = (
        (total_tp + total_tn)
        /
        (
            total_tp
            + total_tn
            + total_fp
            + total_fn
            + 1e-8
        )
    )

    precision = (
        total_tp
        /
        (
            total_tp
            + total_fp
            + 1e-8
        )
    )

    recall = (
        total_tp
        /
        (
            total_tp
            + total_fn
            + 1e-8
        )
    )

    f1 = (
        2 * precision * recall
        /
        (
            precision
            + recall
            + 1e-8
        )
    )

    iou = (
        total_tp
        /
        (
            total_tp
            + total_fp
            + total_fn
            + 1e-8
        )
    )

    dice = (
        2 * total_tp
        /
        (
            2 * total_tp
            + total_fp
            + total_fn
            + 1e-8
        )
    )

    # ========================================================
    # SCENE-LEVEL MEANS
    # ========================================================

    scene_results = np.asarray(
        scene_results,
        dtype=np.float64
    )

    mean_accuracy = (
        scene_results[:, 0].mean()
    )

    mean_precision = (
        scene_results[:, 1].mean()
    )

    mean_recall = (
        scene_results[:, 2].mean()
    )

    mean_f1 = (
        scene_results[:, 3].mean()
    )

    mean_iou = (
        scene_results[:, 4].mean()
    )

    mean_dice = (
        scene_results[:, 5].mean()
    )

    std_f1 = (
        scene_results[:, 3].std()
    )

    std_iou = (
        scene_results[:, 4].std()
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n\n")

    print("=" * 70)
    print("CORRECTED HYBRID MODEL")
    print("FINAL UNTOUCHED OSCD TEST RESULTS")
    print("=" * 70)

    print("\nOverall pixel-level metrics:")

    print(
        f"Accuracy       : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision      : "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall         : "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score       : "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"IoU            : "
        f"{iou:.4f} "
        f"({iou * 100:.2f}%)"
    )

    print(
        f"Dice           : "
        f"{dice:.4f} "
        f"({dice * 100:.2f}%)"
    )

    print("\nMean scene-level metrics:")

    print(
        f"Mean Accuracy  : "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Mean Precision : "
        f"{mean_precision:.4f}"
    )

    print(
        f"Mean Recall    : "
        f"{mean_recall:.4f}"
    )

    print(
        f"Mean F1        : "
        f"{mean_f1:.4f} "
        f"± {std_f1:.4f}"
    )

    print(
        f"Mean IoU       : "
        f"{mean_iou:.4f} "
        f"± {std_iou:.4f}"
    )

    print(
        f"Mean Dice      : "
        f"{mean_dice:.4f}"
    )

    # ========================================================
    # CONFUSION COUNTS
    # ========================================================

    print("\nConfusion counts:")

    print(
        f"TP : {total_tp}"
    )

    print(
        f"TN : {total_tn}"
    )

    print(
        f"FP : {total_fp}"
    )

    print(
        f"FN : {total_fn}"
    )

    # ========================================================
    # COMPARISON
    # ========================================================

    print("\n")
    print("=" * 70)
    print("COMPARISON WITH EXISTING AUGMENTED HYBRID")
    print("=" * 70)

    existing_f1 = 0.4744
    existing_iou = 0.3110

    print(
        f"Existing Augmented Hybrid F1 : "
        f"{existing_f1 * 100:.2f}%"
    )

    print(
        f"Corrected Hybrid F1          : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Existing Augmented Hybrid IoU: "
        f"{existing_iou * 100:.2f}%"
    )

    print(
        f"Corrected Hybrid IoU         : "
        f"{iou * 100:.2f}%"
    )

    f1_difference = (
        f1 - existing_f1
    ) * 100

    iou_difference = (
        iou - existing_iou
    ) * 100

    print(
        f"\nF1 difference                : "
        f"{f1_difference:+.2f} percentage points"
    )

    print(
        f"IoU difference               : "
        f"{iou_difference:+.2f} percentage points"
    )

    print("=" * 70)
    print("Evaluation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()