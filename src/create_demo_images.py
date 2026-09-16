import os
import numpy as np
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "train"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "demo_images"
)

PATCH_SIZE = 128


# ============================================================
# SENTINEL-2 RGB CONVERSION
# ============================================================

def make_rgb(image):

    # Sentinel-2 bands:
    # B2 = index 1
    # B3 = index 2
    # B4 = index 3

    blue = image[1].astype(np.float32)
    green = image[2].astype(np.float32)
    red = image[3].astype(np.float32)

    # Normalize reflectance
    red = np.clip(red / 10000.0, 0, 1)
    green = np.clip(green / 10000.0, 0, 1)
    blue = np.clip(blue / 10000.0, 0, 1)

    # Slight contrast enhancement for visualization
    rgb = np.stack(
        [red, green, blue],
        axis=-1
    )

    rgb = np.clip(
        rgb * 2.5,
        0,
        1
    )

    rgb = (
        rgb * 255
    ).astype(np.uint8)

    return rgb


# ============================================================
# FIND GOOD CHANGED PATCHES
# ============================================================

def find_good_patches():

    files = sorted(
        [
            os.path.join(
                TRAIN_DIR,
                f
            )
            for f in os.listdir(
                TRAIN_DIR
            )
            if f.endswith(".npz")
        ]
    )

    candidates = []

    for path in files:

        data = np.load(path)

        mask = data["mask"]

        change_ratio = (
            mask.sum()
            /
            mask.size
            *
            100
        )

        # Prefer patches with visible change
        if 1.0 <= change_ratio <= 10.0:

            candidates.append(
                (
                    change_ratio,
                    path
                )
            )

    # Largest change first
    candidates.sort(
        reverse=True
    )

    return candidates


# ============================================================
# SAVE DEMO PAIR
# ============================================================

def save_demo_pair(
    source_path,
    number
):

    data = np.load(
        source_path
    )

    image1 = data[
        "image1"
    ]

    image2 = data[
        "image2"
    ]

    mask = data[
        "mask"
    ]

    t1_rgb = make_rgb(
        image1
    )

    t2_rgb = make_rgb(
        image2
    )

    Image.fromarray(
        t1_rgb
    ).save(
        os.path.join(
            OUTPUT_DIR,
            f"sample_{number}_T1.png"
        )
    )

    Image.fromarray(
        t2_rgb
    ).save(
        os.path.join(
            OUTPUT_DIR,
            f"sample_{number}_T2.png"
        )
    )

    # Also save ground truth for your own reference.
    mask_image = (
        mask * 255
    ).astype(
        np.uint8
    )

    Image.fromarray(
        mask_image
    ).save(
        os.path.join(
            OUTPUT_DIR,
            f"sample_{number}_ground_truth.png"
        )
    )

    ratio = (
        mask.sum()
        /
        mask.size
        *
        100
    )

    print(
        f"Sample {number}: "
        f"{os.path.basename(source_path)} "
        f"| Ground-truth change: "
        f"{ratio:.2f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print(
        "CREATING DEMO SATELLITE IMAGE PAIRS"
    )
    print("=" * 65)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Remove previous demo images
    for filename in os.listdir(
        OUTPUT_DIR
    ):

        if filename.endswith(".png"):

            os.remove(
                os.path.join(
                    OUTPUT_DIR,
                    filename
                )
            )

    candidates = find_good_patches()

    print(
        f"\nFound {len(candidates)} "
        f"patches with visible change."
    )

    if len(candidates) < 3:

        raise RuntimeError(
            "Not enough suitable changed patches."
        )

    # Create three demo pairs
    for i in range(3):

        _, source_path = candidates[i]

        save_demo_pair(
            source_path,
            i + 1
        )

    print()
    print("=" * 65)
    print(
        "DEMO IMAGES CREATED"
    )
    print("=" * 65)

    print(
        f"Location:\n{OUTPUT_DIR}"
    )

    print()
    print(
        "Files:"
    )

    for filename in sorted(
        os.listdir(
            OUTPUT_DIR
        )
    ):

        print(
            f"  {filename}"
        )

    print()
    print(
        "Upload matching T1/T2 pairs to the website."
    )

    print(
        "Example:"
    )

    print(
        "  sample_1_T1.png"
    )

    print(
        "  sample_1_T2.png"
    )

    print("=" * 65)


if __name__ == "__main__":

    main()