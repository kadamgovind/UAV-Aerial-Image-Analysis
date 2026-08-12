from pathlib import Path
import csv

DATASET_ROOT = Path("data/processed/yolo")

SPLITS = ["train", "val"]

MIN_CLASS_ID = 0
MAX_CLASS_ID = 9

total_images = 0
total_labels = 0
total_objects = 0

missing_labels = 0
orphan_labels = 0
invalid_rows = 0
invalid_classes = 0
invalid_coordinates = 0
empty_labels = 0

report_rows = []


def validate_label(label_path):
    global invalid_rows
    global invalid_classes
    global invalid_coordinates
    global total_objects
    global empty_labels

    content = label_path.read_text(encoding="utf-8").strip()

    if not content:
        empty_labels += 1
        return

    for line_number, line in enumerate(content.splitlines(), start=1):

        parts = line.strip().split()

        if len(parts) != 5:
            invalid_rows += 1
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError:
            invalid_rows += 1
            continue

        total_objects += 1

        if not (MIN_CLASS_ID <= class_id <= MAX_CLASS_ID):
            invalid_classes += 1

        values = [
            x_center,
            y_center,
            width,
            height
        ]

        if any(value < 0 or value > 1 for value in values):
            invalid_coordinates += 1


for split in SPLITS:

    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    image_files = sorted(image_dir.glob("*.jpg"))
    label_files = sorted(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    missing = image_stems - label_stems
    orphan = label_stems - image_stems

    missing_labels += len(missing)
    orphan_labels += len(orphan)

    total_images += len(image_files)
    total_labels += len(label_files)

    for label_path in label_files:
        before = (
            invalid_rows,
            invalid_classes,
            invalid_coordinates,
            total_objects
        )

        validate_label(label_path)

        after = (
            invalid_rows,
            invalid_classes,
            invalid_coordinates,
            total_objects
        )

        report_rows.append([
            split,
            label_path.name,
            after[3] - before[3],
            "checked"
        ])


REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = REPORT_DIR / "yolo_integrity_report.csv"

with REPORT_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "split",
        "label_file",
        "objects",
        "status"
    ])

    writer.writerows(report_rows)


print("=" * 60)
print("YOLO Dataset Integrity Validation")
print("=" * 60)

print(f"Images checked       : {total_images:,}")
print(f"Labels checked       : {total_labels:,}")
print(f"Objects checked      : {total_objects:,}")

print()
print(f"Missing labels       : {missing_labels}")
print(f"Orphan labels        : {orphan_labels}")
print(f"Invalid rows         : {invalid_rows}")
print(f"Invalid class IDs    : {invalid_classes}")
print(f"Invalid coordinates  : {invalid_coordinates}")
print(f"Empty label files    : {empty_labels}")

print()
print(f"Report: {REPORT_FILE}")

print("=" * 60)

if (
    missing_labels == 0
    and orphan_labels == 0
    and invalid_rows == 0
    and invalid_classes == 0
    and invalid_coordinates == 0
):
    print("STATUS: PASS")
else:
    print("STATUS: CHECK REQUIRED")