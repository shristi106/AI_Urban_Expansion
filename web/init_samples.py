import numpy as np
from PIL import Image
from pathlib import Path
import json
import shutil

base_dir = Path("C:/Users/SHRISTI/OneDrive/Desktop/AI_Urban_Expansion")
val_dir = base_dir / "dataset/val"
samples_dir = base_dir / "web/static/samples"
samples_dir.mkdir(parents=True, exist_ok=True)

# Also create assets dir and copy result figures
assets_dir = base_dir / "web/static/assets"
assets_dir.mkdir(parents=True, exist_ok=True)

results_dir = base_dir / "results"
for f in results_dir.glob("*.png"):
    shutil.copyfile(f, assets_dir / f.name)

def make_rgb(im):
    rgb = im[[3, 2, 1]].astype(np.float32)
    rmin, rmax = rgb.min(), rgb.max()
    rgb = (rgb - rmin) / (rmax - rmin + 1e-8)
    rgb = np.transpose(rgb, (1, 2, 0))
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))

sample_info = [
    {
        "id": "sample_1",
        "file": "changed_0005.npz",
        "name": "Scene A: Rapid Urban Outskirts Expansion",
        "location": "Montpellier Suburbs (Sentinel-2)",
        "expected": "High Expansion (~21.5%)",
        "desc": "Intensive conversion of agricultural/fallow parcels to built-up residential structures and impervious road paving."
    },
    {
        "id": "sample_2",
        "file": "changed_0018.npz",
        "name": "Scene B: Commercial & Transport Corridor",
        "location": "Beirut Suburbs (Sentinel-2)",
        "expected": "Moderate Growth (~2.1%)",
        "desc": "Linear infrastructure extension and commercial warehouse site preparation along an arterial highway."
    },
    {
        "id": "sample_3",
        "file": "changed_0000.npz",
        "name": "Scene C: Localized Infill Development",
        "location": "Las Vegas Fringe (Sentinel-2)",
        "expected": "Localized Infill (~1.4%)",
        "desc": "Subdivision expansion and building construction on previously bare desert soil."
    },
    {
        "id": "sample_4",
        "file": "unchanged_0000.npz",
        "name": "Scene D: Dense Stable Urban Matrix (Control)",
        "location": "Madrid Metropolitan (Sentinel-2)",
        "expected": "No Change (0.0%)",
        "desc": "Established dense urban core serving as a negative control to test noise rejection and false-positive suppression."
    }
]

for s in sample_info:
    src_file = val_dir / s["file"]
    if src_file.exists():
        data = np.load(src_file)
        t1_rgb = make_rgb(data["image1"])
        t2_rgb = make_rgb(data["image2"])
        t1_rgb.save(samples_dir / f"{s['id']}_t1.png")
        t2_rgb.save(samples_dir / f"{s['id']}_t2.png")
        shutil.copyfile(src_file, samples_dir / s["file"])
        print(f"Generated sample previews and copied {s['file']}")

with open(samples_dir / "samples.json", "w") as f:
    json.dump(sample_info, f, indent=2)

print("All sample previews and assets initialized successfully!")
