import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Allow imports from src/
sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(__file__))
)

from model_hybrid import HybridSiameseCNNViT


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

REAL_TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train"
)

GAN_TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "gan_synthetic_corrected"
)

VAL_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "val"
)

OUTPUT_MODEL = os.path.join(
    PROJECT_ROOT,
    "models",
    "hybrid_siamese_vit_gan_corrected_trial.pth"
)

BATCH_SIZE = 2

EPOCHS = 5

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DATASET
# ============================================================

class CombinedGANDataset(Dataset):

    def __init__(
        self,
        real_files,
        gan_files
    ):

        self.samples = []

        # ----------------------------------------------------
        # Real samples
        # ----------------------------------------------------

        for path in real_files:

            self.samples.append(
                (
                    path,
                    "real"
                )
            )

        # ----------------------------------------------------
        # GAN samples
        # ----------------------------------------------------

        for path in gan_files:

            self.samples.append(
                (
                    path,
                    "gan"
                )
            )

        # Shuffle only training sample order
        random.shuffle(
            self.samples
        )

        print(
            f"Real samples : {len(real_files)}"
        )

        print(
            f"GAN samples  : {len(gan_files)}"
        )

        print(
            f"Total samples: {len(self.samples)}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(
        self,
        index
    ):

        path, sample_type = (
            self.samples[index]
        )

        data = np.load(path)

        image1 = data[
            "image1"
        ].astype(
            np.float32
        )

        image2 = data[
            "image2"
        ].astype(
            np.float32
        )

        mask = data[
            "mask"
        ].astype(
            np.float32
        )

        # ----------------------------------------------------
        # REAL DATA
        #
        # Stored as uint16-like Sentinel-2 values.
        # Normalize here.
        # ----------------------------------------------------

        if sample_type == "real":

            image1 = (
                image1 / 10000.0
            )

            image2 = (
                image2 / 10000.0
            )

        # ----------------------------------------------------
        # GAN DATA
        #
        # Already normalized to [0,1].
        #
        # DO NOT divide by 10000 again.
        # ----------------------------------------------------

        else:

            # Safety clipping
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

        # ----------------------------------------------------
        # Final safety clipping
        # ----------------------------------------------------

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

        mask = np.clip(
            mask,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image1 = torch.from_numpy(
            image1
        ).float()

        image2 = torch.from_numpy(
            image2
        ).float()

        mask = torch.from_numpy(
            mask
        ).float()

        # ----------------------------------------------------
        # Mask:
        # [128,128] → [1,128,128]
        # ----------------------------------------------------

        mask = mask.unsqueeze(
            0
        )

        return (
            image1,
            image2,
            mask
        )


# ============================================================
# VALIDATION DATASET
# ============================================================

class ValidationDataset(Dataset):

    def __init__(
        self,
        directory
    ):

        self.files = sorted(
            [
                os.path.join(
                    directory,
                    f
                )
                for f in os.listdir(
                    directory
                )
                if f.endswith(".npz")
            ]
        )

        print(
            f"Validation samples: "
            f"{len(self.files)}"
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(
        self,
        index
    ):

        path = self.files[index]

        data = np.load(path)

        image1 = data[
            "image1"
        ].astype(
            np.float32
        )

        image2 = data[
            "image2"
        ].astype(
            np.float32
        )

        mask = data[
            "mask"
        ].astype(
            np.float32
        )

        # ----------------------------------------------------
        # Validation patches are real Sentinel-2 data.
        # Normalize exactly once.
        # ----------------------------------------------------

        image1 = (
            image1 / 10000.0
        )

        image2 = (
            image2 / 10000.0
        )

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

        mask = np.clip(
            mask,
            0.0,
            1.0
        )

        image1 = torch.from_numpy(
            image1
        ).float()

        image2 = torch.from_numpy(
            image2
        ).float()

        mask = torch.from_numpy(
            mask
        ).float().unsqueeze(
            0
        )

        return (
            image1,
            image2,
            mask
        )


# ============================================================
# GET FILES
# ============================================================

def get_npz_files(directory):

    if not os.path.exists(directory):

        raise FileNotFoundError(
            f"Directory not found:\n{directory}"
        )

    return sorted(
        [
            os.path.join(
                directory,
                f
            )
            for f in os.listdir(
                directory
            )
            if f.endswith(".npz")
        ]
    )


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    predictions,
    targets,
    smooth=1.0
):

    predictions = torch.sigmoid(
        predictions
    )

    predictions = predictions.reshape(
        -1
    )

    targets = targets.reshape(
        -1
    )

    intersection = (
        predictions
        *
        targets
    ).sum()

    dice = (
        2.0
        *
        intersection
        +
        smooth
    ) / (
        predictions.sum()
        +
        targets.sum()
        +
        smooth
    )

    return 1.0 - dice


# ============================================================
# COMBINED BCE + DICE LOSS
# ============================================================

def combined_loss(
    predictions,
    targets
):

    bce = nn.functional.binary_cross_entropy_with_logits(
        predictions,
        targets
    )

    dice = dice_loss(
        predictions,
        targets
    )

    return bce + dice


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    targets
):

    probabilities = torch.sigmoid(
        predictions
    )

    predicted = (
        probabilities >= 0.5
    ).float()

    targets = (
        targets >= 0.5
    ).float()

    predicted = predicted.reshape(
        -1
    )

    targets = targets.reshape(
        -1
    )

    tp = (
        predicted
        *
        targets
    ).sum().item()

    fp = (
        predicted
        *
        (1.0 - targets)
    ).sum().item()

    fn = (
        (1.0 - predicted)
        *
        targets
    ).sum().item()

    tn = (
        (1.0 - predicted)
        *
        (1.0 - targets)
    ).sum().item()

    epsilon = 1e-8

    precision = (
        tp
        /
        (tp + fp + epsilon)
    )

    recall = (
        tp
        /
        (tp + fn + epsilon)
    )

    f1 = (
        2.0
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
            +
            epsilon
        )
    )

    iou = (
        tp
        /
        (
            tp
            +
            fp
            +
            fn
            +
            epsilon
        )
    )

    accuracy = (
        (tp + tn)
        /
        (
            tp
            +
            tn
            +
            fp
            +
            fn
            +
            epsilon
        )
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        iou
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    total_loss = 0.0

    for (
        image1,
        image2,
        mask
    ) in loader:

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

        loss = combined_loss(
            output,
            mask
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
        )

    return (
        total_loss
        /
        len(loader)
    )


# ============================================================
# VALIDATION
# ============================================================

def validate(
    model,
    loader
):

    model.eval()

    total_loss = 0.0

    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_iou = 0.0

    with torch.no_grad():

        for (
            image1,
            image2,
            mask
        ) in loader:

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

            loss = combined_loss(
                output,
                mask
            )

            total_loss += (
                loss.item()
            )

            (
                accuracy,
                precision,
                recall,
                f1,
                iou
            ) = calculate_metrics(
                output,
                mask
            )

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            total_iou += iou

    n = len(loader)

    return (
        total_loss / n,
        total_accuracy / n,
        total_precision / n,
        total_recall / n,
        total_f1 / n,
        total_iou / n
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "5-EPOCH GAN AUGMENTATION TRIAL"
    )
    print(
        "SIAMESE CNN + VISION TRANSFORMER"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print()

    # ========================================================
    # LOAD FILES
    # ========================================================

    real_files = get_npz_files(
        REAL_TRAIN_DIR
    )

    gan_files = get_npz_files(
        GAN_TRAIN_DIR
    )

    val_files = get_npz_files(
        VAL_DIR
    )

    print(
        f"Real training patches: "
        f"{len(real_files)}"
    )

    print(
        f"GAN synthetic patches: "
        f"{len(gan_files)}"
    )

    print(
        f"Validation patches: "
        f"{len(val_files)}"
    )

    # ========================================================
    # VERIFY GAN DATA
    # ========================================================

    print()
    print(
        "Checking GAN dataset..."
    )

    for i, path in enumerate(
        gan_files[:10]
    ):

        data = np.load(path)

        image1 = data[
            "image1"
        ]

        image2 = data[
            "image2"
        ]

        mask = data[
            "mask"
        ]

        assert image1.shape == (
            13,
            128,
            128
        )

        assert image2.shape == (
            13,
            128,
            128
        )

        assert mask.shape == (
            128,
            128
        )

        assert (
            image1.min() >= 0.0
            and
            image1.max() <= 1.0
        )

        assert (
            image2.min() >= 0.0
            and
            image2.max() <= 1.0
        )

        assert set(
            np.unique(mask).tolist()
        ).issubset({0, 1})

    print(
        "GAN dataset verification: PASS"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = CombinedGANDataset(
        real_files,
        gan_files
    )

    val_dataset = ValidationDataset(
        VAL_DIR
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    print()
    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(val_loader)}"
    )

    # ========================================================
    # TEST ONE BATCH BEFORE TRAINING
    # ========================================================

    print()
    print(
        "Testing data pipeline..."
    )

    sample_t1, sample_t2, sample_mask = next(
        iter(train_loader)
    )

    print(
        f"T1 shape   : {sample_t1.shape}"
    )

    print(
        f"T2 shape   : {sample_t2.shape}"
    )

    print(
        f"Mask shape : {sample_mask.shape}"
    )

    print(
        f"T1 range   : "
        f"{sample_t1.min().item():.4f} "
        f"to "
        f"{sample_t1.max().item():.4f}"
    )

    print(
        f"T2 range   : "
        f"{sample_t2.min().item():.4f} "
        f"to "
        f"{sample_t2.max().item():.4f}"
    )

    print(
        f"Mask values: "
        f"{torch.unique(sample_mask)}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "Creating model..."
    )

    model = HybridSiameseCNNViT(
        in_channels=13
    )

    model = model.to(
        DEVICE
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: "
        f"{total_params:,}"
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # TRAIN
    # ========================================================

    best_f1 = -1.0

    best_epoch = 0

    print()
    print("=" * 70)
    print(
        "STARTING 5-EPOCH TRIAL"
    )
    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        (
            val_loss,
            accuracy,
            precision,
            recall,
            f1,
            iou
        ) = validate(
            model,
            val_loader
        )

        print()
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Accuracy   : "
            f"{accuracy:.4f}"
        )

        print(
            f"Precision  : "
            f"{precision:.4f}"
        )

        print(
            f"Recall     : "
            f"{recall:.4f}"
        )

        print(
            f"F1         : "
            f"{f1:.4f}"
        )

        print(
            f"IoU        : "
            f"{iou:.4f}"
        )

        # ----------------------------------------------------
        # Save best TRIAL checkpoint only
        # ----------------------------------------------------

        if f1 > best_f1:

            best_f1 = f1

            best_epoch = epoch

            torch.save(
                model.state_dict(),
                OUTPUT_MODEL
            )

            print(
                "✓ New best trial model saved."
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "GAN TRIAL COMPLETE"
    )
    print("=" * 70)

    print(
        f"Best epoch : {best_epoch}"
    )

    print(
        f"Best Val F1: "
        f"{best_f1:.4f}"
    )

    print()
    print(
        "Trial model saved to:"
    )

    print(
        OUTPUT_MODEL
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This is ONLY a 5-epoch trial."
    )

    print(
        "It has NOT replaced the existing final model."
    )

    print(
        "Existing final model:"
    )

    print(
        "models/hybrid_siamese_vit_augmented.pth"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()