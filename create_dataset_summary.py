from pathlib import Path
from collections import Counter
from PIL import Image
import json
import statistics

PROJECT_ROOT = Path(".")

TRAIN_IMAGES = Path(
    "data/raw/VisDrone/train/VisDrone2019-DET-train/images"
)

TRAIN_ANN = Path(
    "data/raw/VisDrone/train/VisDrone2019-DET-train/annotations"
)

VAL_IMAGES = Path(
    "data/raw/VisDrone/val/VisDrone2019-DET-val/images"
)

VAL_ANN = Path(
    "data/raw/VisDrone/val/VisDrone2019-DET-val/annotations"
)

REPORT_DIR = Path("reports")
REPORT_FILE = REPORT_DIR / "dataset_summary.json"

CLASS_NAMES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def count_files(directory, extension):
    return len(list(directory.glob(f"*.{extension}")))


def analyze_annotations(directory):
    class_counts = Counter()
    total_objects = 0

    for txt_file in directory.glob("*.txt"):
        with txt_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().rstrip(",")

                if not line:
                    continue

                values = [x.strip() for x in line.split(",")]

                if len(values) != 8:
                    continue

                category = int(values[5])

                if category in CLASS_NAMES:
                    class_counts[category] += 1
                    total_objects += 1

    return total_objects, class_counts


def analyze_resolutions(directory):
    widths = []
    heights = []

    for image_path in directory.glob("*.jpg"):
        try:
            with Image.open(image_path) as img:
                width, height = img.size

            widths.append(width)
            heights.append(height)

        except Exception:
            pass

    if not widths:
        return {}

    return {
        "images_analyzed": len(widths),
        "min_width": min(widths),
        "max_width": max(widths),
        "average_width": round(statistics.mean(widths), 2),
        "min_height": min(heights),
        "max_height": max(heights),
        "average_height": round(statistics.mean(heights), 2),
    }


train_objects, train_classes = analyze_annotations(TRAIN_ANN)
val_objects, val_classes = analyze_annotations(VAL_ANN)

train_class_distribution = {}

for class_id, class_name in CLASS_NAMES.items():
    count = train_classes[class_id]
    percentage = (
        count / train_objects * 100
        if train_objects
        else 0
    )

    train_class_distribution[class_name] = {
        "count": count,
        "percentage": round(percentage, 2),
    }


val_class_distribution = {}

for class_id, class_name in CLASS_NAMES.items():
    count = val_classes[class_id]
    percentage = (
        count / val_objects * 100
        if val_objects
        else 0
    )

    val_class_distribution[class_name] = {
        "count": count,
        "percentage": round(percentage, 2),
    }


summary = {
    "dataset": "VisDrone2019-DET",
    "purpose": "UAV Aerial Image Analysis System",

    "train": {
        "images": count_files(TRAIN_IMAGES, "jpg"),
        "annotation_files": count_files(TRAIN_ANN, "txt"),
        "valid_objects": train_objects,
        "class_distribution": train_class_distribution,
        "resolution": analyze_resolutions(TRAIN_IMAGES),
    },

    "validation": {
        "images": count_files(VAL_IMAGES, "jpg"),
        "annotation_files": count_files(VAL_ANN, "txt"),
        "valid_objects": val_objects,
        "class_distribution": val_class_distribution,
        "resolution": analyze_resolutions(VAL_IMAGES),
    },

    "classes": CLASS_NAMES,
}

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with REPORT_FILE.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("=" * 60)
print("UAV Dataset Master Summary")
print("=" * 60)

print(f"Train images       : {summary['train']['images']:,}")
print(f"Train annotations  : {summary['train']['annotation_files']:,}")
print(f"Train objects      : {train_objects:,}")

print()

print(f"Validation images  : {summary['validation']['images']:,}")
print(f"Validation ann.    : {summary['validation']['annotation_files']:,}")
print(f"Validation objects : {val_objects:,}")

print()

print(f"Report: {REPORT_FILE}")
print("STATUS: SUCCESS")
print("=" * 60)