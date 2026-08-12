from pathlib import Path
from collections import Counter
import csv

ANNOTATION_DIR = Path(
    "data/raw/VisDrone/train/VisDrone2019-DET-train/annotations"
)

OUTPUT_DIR = Path("reports")
OUTPUT_FILE = OUTPUT_DIR / "class_distribution.csv"

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

            if category in CLASS_NAMES:
                class_counts[category] += 1

total_annotations = sum(class_counts.values())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "class_id",
        "class_name",
        "object_count",
        "percentage"
    ])

    for class_id, class_name in CLASS_NAMES.items():
        count = class_counts[class_id]
        percentage = (
            count / total_annotations * 100
            if total_annotations
            else 0
        )

        writer.writerow([
            class_id,
            class_name,
            count,
            f"{percentage:.2f}"
        ])

print("=" * 50)
print("Class distribution report created")
print("=" * 50)
print(f"Output: {OUTPUT_FILE}")
print(f"Total objects: {total_annotations:,}")
print("STATUS: SUCCESS")