import sys
from pathlib import Path
import io
import random

import numpy as np
import polars as pl
import torch
import torch.nn as nn

from PIL import Image
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model_hybrid import HybridSiameseCNNViT


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = Path(
    r"C:\Users\SHRISTI\.cache\huggingface\hub"
    r"\datasets--blanchon--OSCD_MSI\snapshots"
    r"\16651ca137f0b4577801cf4413c77453d829ec5a\data"
)

TRAIN_PARQUET = (
    CACHE_DIR / "train-00000-of-00001.parquet"
)

PATCH_SIZE = 128
STRIDE = 128

N_FOLDS = 10

BATCH_SIZE = 2
EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cpu")

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CV_RESULTS_FILE = (
    RESULTS_DIR / "10_fold_cv_results.txt"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()

        self.smooth = smooth


    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = (
            probabilities.contiguous()
        )

        targets = (
            targets.contiguous()
        )

        intersection = (
            probabilities * targets
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probabilities.sum(
                dim=(1, 2, 3)
            )
            +
            targets.sum(
                dim=(1, 2, 3)
            )
        )

        dice = (
            (
                2.0 * intersection
                + self.smooth
            )
            /
            (
                denominator
                + self.smooth
            )
        )

        return 1.0 - dice.mean()


# ============================================================
# BCE + DICE
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

        self.dice = DiceLoss()


    def forward(
        self,
        logits,
        targets
    ):

        return (
            self.bce(
                logits,
                targets
            )
            +
            self.dice(
                logits,
                targets
            )
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= 0.5
    ).float()

    targets = targets.float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    tp = (
        (predictions == 1)
        &
        (targets == 1)
    ).sum().item()

    tn = (
        (predictions == 0)
        &
        (targets == 0)
    ).sum().item()

    fp = (
        (predictions == 1)
        &
        (targets == 0)
    ).sum().item()

    fn = (
        (predictions == 0)
        &
        (targets == 1)
    ).sum().item()

    precision = (
        tp / (tp + fp + 1e-8)
    )

    recall = (
        tp / (tp + fn + 1e-8)
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
        tp
        /
        (
            tp
            + fp
            + fn
            + 1e-8
        )
    )

    accuracy = (
        (tp + tn)
        /
        (
            tp
            + tn
            + fp
            + fn
            + 1e-8
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }


# ============================================================
# PATCH CREATION
# ============================================================

def create_patches(
    image1,
    image2,
    mask
):

    patches = []

    _, height, width = image1.shape

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

            patch_mask = mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # Skip completely empty patches
            # to reduce extreme imbalance.

            if (
                patch_mask.sum()
                == 0
            ):

                continue

            patches.append(
                (
                    patch1,
                    patch2,
                    patch_mask
                )
            )

    return patches


# ============================================================
# DATASET
# ============================================================

class ScenePatchDataset(Dataset):

    def __init__(
        self,
        scenes
    ):

        self.samples = []

        for scene_index in scenes:

            image1 = scene_index[
                "image1"
            ]

            image2 = scene_index[
                "image2"
            ]

            mask = scene_index[
                "mask"
            ]

            patches = create_patches(
                image1,
                image2,
                mask
            )

            self.samples.extend(
                patches
            )


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        image1, image2, mask = (
            self.samples[index]
        )

        image1 = (
            image1.astype(
                np.float32
            )
            / 10000.0
        )

        image2 = (
            image2.astype(
                np.float32
            )
            / 10000.0
        )

        mask = mask.astype(
            np.float32
        )

        image1 = torch.from_numpy(
            image1
        )

        image2 = torch.from_numpy(
            image2
        )

        mask = torch.from_numpy(
            mask
        ).unsqueeze(0)

        return (
            image1,
            image2,
            mask
        )


# ============================================================
# LOAD ORIGINAL OSCD SCENES
# ============================================================

def load_scenes():

    print(
        "\nLoading original OSCD scenes..."
    )

    table = pl.read_parquet(
        TRAIN_PARQUET
    )

    scenes = []

    for row in table.iter_rows(
        named=True
    ):

        image1 = np.array(
            row["image1"],
            dtype=np.uint16
        )

        image2 = np.array(
            row["image2"],
            dtype=np.uint16
        )

        mask_bytes = (
            row["mask"]["bytes"]
        )

        mask_image = (
            Image.open(
                io.BytesIO(
                    mask_bytes
                )
            )
        )

        mask = np.array(
            mask_image,
            dtype=np.uint8
        )

        scenes.append({
            "image1": image1,
            "image2": image2,
            "mask": mask
        })

    print(
        f"Loaded {len(scenes)} scenes."
    )

    return scenes


# ============================================================
# TRAIN ONE FOLD
# ============================================================

def train_fold(
    train_scenes,
    val_scenes,
    fold_number
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"FOLD {fold_number}/{N_FOLDS}"
    )

    print(
        "=" * 70
    )

    print(
        f"Training scenes: "
        f"{len(train_scenes)}"
    )

    print(
        f"Validation scenes: "
        f"{len(val_scenes)}"
    )


    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = (
        ScenePatchDataset(
            train_scenes
        )
    )

    val_dataset = (
        ScenePatchDataset(
            val_scenes
        )
    )

    print(
        f"Training patches: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation patches: "
        f"{len(val_dataset)}"
    )


    if len(train_dataset) == 0:
        raise RuntimeError(
            "No training patches generated."
        )

    if len(val_dataset) == 0:
        raise RuntimeError(
            "No validation patches generated."
        )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = (
        HybridSiameseCNNViT(
            in_channels=13
        )
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = BCEDiceLoss()


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    best_f1 = 0.0
    best_iou = 0.0


    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=(
                f"Fold {fold_number} "
                f"Epoch {epoch}"
            ),
            leave=False
        )

        for image1, image2, mask in progress:

            image1 = image1.to(
                DEVICE
            )

            image2 = image2.to(
                DEVICE
            )

            mask = mask.to(
                DEVICE
            )

            optimizer.zero_grad()

            output = model(
                image1,
                image2
            )

            loss = criterion(
                output,
                mask
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        totals = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "iou": 0.0
        }

        with torch.no_grad():

            for image1, image2, mask in val_loader:

                image1 = image1.to(
                    DEVICE
                )

                image2 = image2.to(
                    DEVICE
                )

                mask = mask.to(
                    DEVICE
                )

                output = model(
                    image1,
                    image2
                )

                metrics = (
                    calculate_metrics(
                        output,
                        mask
                    )
                )

                for key in totals:

                    totals[key] += (
                        metrics[key]
                    )


        num_batches = len(
            val_loader
        )

        for key in totals:

            totals[key] /= num_batches


        print(
            f"Epoch {epoch}: "
            f"F1={totals['f1']:.4f}, "
            f"IoU={totals['iou']:.4f}"
        )


        if totals["f1"] > best_f1:

            best_f1 = totals["f1"]
            best_iou = totals["iou"]


    print(
        f"\nFold {fold_number} Best:"
    )

    print(
        f"F1  : {best_f1:.4f}"
    )

    print(
        f"IoU : {best_iou:.4f}"
    )


    return (
        best_f1,
        best_iou
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "10-FOLD SCENE-LEVEL CROSS VALIDATION"
    )

    print(
        "SIAMESE CNN + VISION TRANSFORMER"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Number of folds: {N_FOLDS}"
    )

    print(
        f"Epochs per fold: {EPOCHS}"
    )


    # --------------------------------------------------------
    # LOAD SCENES
    # --------------------------------------------------------

    scenes = load_scenes()


    # --------------------------------------------------------
    # CREATE 10 SCENE-LEVEL FOLDS
    # --------------------------------------------------------

    scene_indices = list(
        range(len(scenes))
    )

    random.Random(
        SEED
    ).shuffle(
        scene_indices
    )

    folds = np.array_split(
        scene_indices,
        N_FOLDS
    )


    results = []


    # --------------------------------------------------------
    # RUN FOLDS
    # --------------------------------------------------------

    for fold_index in range(
        N_FOLDS
    ):

        validation_indices = (
            set(
                folds[fold_index]
                .tolist()
            )
        )

        train_indices = [
            index
            for index in scene_indices
            if index
            not in validation_indices
        ]


        train_scenes = [
            scenes[index]
            for index in train_indices
        ]

        val_scenes = [
            scenes[index]
            for index in validation_indices
        ]


        f1, iou = train_fold(
            train_scenes,
            val_scenes,
            fold_index + 1
        )


        results.append({
            "fold":
                fold_index + 1,
            "f1":
                f1,
            "iou":
                iou
        })


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    f1_scores = np.array([
        result["f1"]
        for result in results
    ])

    iou_scores = np.array([
        result["iou"]
        for result in results
    ])


    mean_f1 = (
        f1_scores.mean()
    )

    std_f1 = (
        f1_scores.std(
            ddof=1
        )
    )

    mean_iou = (
        iou_scores.mean()
    )

    std_iou = (
        iou_scores.std(
            ddof=1
        )
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "10-FOLD CROSS-VALIDATION RESULTS"
    )

    print(
        "=" * 70
    )


    for result in results:

        print(
            f"Fold {result['fold']:2d} | "
            f"F1 = {result['f1']:.4f} | "
            f"IoU = {result['iou']:.4f}"
        )


    print(
        "\n" + "-" * 70
    )

    print(
        f"Mean F1  : "
        f"{mean_f1:.4f} "
        f"+/- {std_f1:.4f}"
    )

    print(
        f"Mean IoU : "
        f"{mean_iou:.4f} "
        f"+/- {std_iou:.4f}"
    )

    print(
        "-" * 70
    )


    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        CV_RESULTS_FILE,
        "w"
    ) as file:

        file.write(
            "10-FOLD SCENE-LEVEL CROSS VALIDATION\n"
        )

        file.write(
            "Siamese CNN + Vision Transformer\n\n"
        )

        for result in results:

            file.write(
                f"Fold {result['fold']}: "
                f"F1={result['f1']:.4f}, "
                f"IoU={result['iou']:.4f}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            f"Mean F1: "
            f"{mean_f1:.4f} "
            f"+/- {std_f1:.4f}\n"
        )

        file.write(
            f"Mean IoU: "
            f"{mean_iou:.4f} "
            f"+/- {std_iou:.4f}\n"
        )


    print(
        f"\nResults saved to:"
    )

    print(
        CV_RESULTS_FILE
    )


    print(
        "\n10-fold cross-validation complete."
    )


if __name__ == "__main__":
    main()