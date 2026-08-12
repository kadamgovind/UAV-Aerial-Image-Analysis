from pathlib import Path
from collections import Counter
import csv
import statistics

ANNOTATION_DIR = Path(
    "data/raw/VisDrone/train/VisDrone2019-DET-train/annotations"
)

OUTPUT_DIR = Path("reports")
OUTPUT_FILE = OUTPUT_DIR / "object_size_analysis.csv"

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

# Object-size thresholds based on bounding-box area.
# These are analysis buckets, not dataset modifications.
def size_category(width, height):
    area = width * height

    if area < 32 * 32:
        return "small"
    elif area < 96 * 96:
        return "medium"
    else:
        return "large"


size_counts = Counter()
class_size_counts = Counter()

widths = []
heights = []
areas = []

total_objects = 0

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

            if category not in CLASS_NAMES:
                continue

            width = int(values[2])
            height = int(values[3])

            if width <= 0 or height <= 0:
                continue

            area = width * height
            category_name = size_category(width, height)

            size_counts[category_name] += 1
            class_size_counts[(category, category_name)] += 1

            widths.append(width)
            heights.append(height)
            areas.append(area)

            total_objects += 1


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "class_id",
        "class_name",
        "small",
        "medium",
        "large",
        "total"
    ])

    for class_id, class_name in CLASS_NAMES.items():

        small = class_size_counts[(class_id, "small")]
        medium = class_size_counts[(class_id, "medium")]
        large = class_size_counts[(class_id, "large")]

        writer.writerow([
            class_id,
            class_name,
            small,
            medium,
            large,
            small + medium + large
        ])


print("=" * 60)
print("VisDrone Object Size Analysis")
print("=" * 60)

print(f"Total valid objects : {total_objects:,}")

print()
print("Size distribution:")
print(f"Small  : {size_counts['small']:,}")
print(f"Medium : {size_counts['medium']:,}")
print(f"Large  : {size_counts['large']:,}")

print()
print("Bounding-box statistics:")
print(f"Average width  : {statistics.mean(widths):.2f} px")
print(f"Average height : {statistics.mean(heights):.2f} px")
print(f"Average area   : {statistics.mean(areas):.2f} px²")

print("=" * 60)
print(f"Report: {OUTPUT_FILE}")
print("STATUS: SUCCESS")