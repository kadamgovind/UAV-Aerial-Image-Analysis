from pathlib import Path
from collections import Counter

ANNOTATION_DIR = Path(
    "data/raw/VisDrone/train/VisDrone2019-DET-train/annotations"
)

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

class_counts = Counter()
total_annotations = 0

for txt_file in ANNOTATION_DIR.glob("*.txt"):
    with txt_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")

            if not line:
                continue

            values = [x.strip() for x in line.split(",")]

            if len(values) != 8:
                continue

            category = int(values[5])

            # VisDrone category 0 = ignored region
            if category in CLASS_NAMES:
                class_counts[category] += 1
                total_annotations += 1

print("=" * 55)
print("VisDrone Training Dataset — Class Distribution")
print("=" * 55)

for class_id, class_name in CLASS_NAMES.items():
    count = class_counts[class_id]
    percentage = (count / total_annotations) * 100 if total_annotations else 0

    print(
        f"{class_id:2d}. {class_name:18s} "
        f"{count:8,d}  ({percentage:6.2f}%)"
    )

print("=" * 55)
print(f"Total valid object annotations: {total_annotations:,}")
print("=" * 55)