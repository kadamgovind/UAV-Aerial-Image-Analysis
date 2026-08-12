from pathlib import Path
from PIL import Image
import shutil
import csv

# ============================================================
# VisDrone → YOLO Dataset Converter
# ============================================================

PROJECT_ROOT = Path(".")

DATASETS = {
    "train": {
        "images": Path(
            "data/raw/VisDrone/train/"
            "VisDrone2019-DET-train/images"
        ),
        "annotations": Path(
            "data/raw/VisDrone/train/"
            "VisDrone2019-DET-train/annotations"
        ),
    },
    "val": {
        "images": Path(
            "data/raw/VisDrone/val/"
            "VisDrone2019-DET-val/images"
        ),
        "annotations": Path(
            "data/raw/VisDrone/val/"
            "VisDrone2019-DET-val/annotations"
        ),
    },
}

OUTPUT_ROOT = Path("data/processed/yolo")

CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

# VisDrone class IDs:
# 1–10 → YOLO class IDs 0–9
CLASS_ID_OFFSET = 1

total_images = 0
total_labels = 0
total_objects = 0
skipped_objects = 0
errors = 0

report_rows = []


def convert_annotation(annotation_path, image_width, image_height):
    """
    Convert one VisDrone annotation file to YOLO format.
    """

    global total_objects
    global skipped_objects

    yolo_lines = []

    with annotation_path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip().rstrip(",")

            if not line:
                continue

            values = [x.strip() for x in line.split(",")]

            if len(values) != 8:
                skipped_objects += 1
                continue

            try:
                x = float(values[0])
                y = float(values[1])
                width = float(values[2])
                height = float(values[3])

                score = int(values[4])
                category = int(values[5])
                truncation = int(values[6])
                occlusion = int(values[7])

            except ValueError:
                skipped_objects += 1
                continue

            # Ignore category 0 / unknown categories
            if category not in range(1, 11):
                skipped_objects += 1
                continue

            # Ignore invalid bounding boxes
            if width <= 0 or height <= 0:
                skipped_objects += 1
                continue

            # Convert VisDrone class → YOLO class
            yolo_class = category - CLASS_ID_OFFSET

            # Convert xywh → YOLO normalized xywh
            x_center = (x + width / 2) / image_width
            y_center = (y + height / 2) / image_height

            normalized_width = width / image_width
            normalized_height = height / image_height

            # Clamp values to valid YOLO range
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            normalized_width = min(max(normalized_width, 0.0), 1.0)
            normalized_height = min(max(normalized_height, 0.0), 1.0)

            yolo_lines.append(
                f"{yolo_class} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{normalized_width:.6f} "
                f"{normalized_height:.6f}"
            )

            total_objects += 1

    return yolo_lines


def process_split(split_name, paths):

    global total_images
    global total_labels
    global errors

    source_images = paths["images"]
    source_annotations = paths["annotations"]

    output_images = OUTPUT_ROOT / "images" / split_name
    output_labels = OUTPUT_ROOT / "labels" / split_name

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(source_images.glob("*.jpg"))

    print()
    print("=" * 60)
    print(f"Processing: {split_name.upper()}")
    print(f"Images found: {len(image_files):,}")
    print("=" * 60)

    for index, image_path in enumerate(image_files, start=1):

        annotation_path = source_annotations / (
            image_path.stem + ".txt"
        )

        try:

            with Image.open(image_path) as image:
                image_width, image_height = image.size

            yolo_lines = convert_annotation(
                annotation_path,
                image_width,
                image_height
            )

            # Copy image without modifying original
            shutil.copy2(
                image_path,
                output_images / image_path.name
            )

            # Write converted annotation
            output_label = output_labels / (
                image_path.stem + ".txt"
            )

            output_label.write_text(
                "\n".join(yolo_lines),
                encoding="utf-8"
            )

            total_images += 1
            total_labels += 1

            report_rows.append([
                split_name,
                image_path.name,
                image_width,
                image_height,
                len(yolo_lines),
                "success"
            ])

        except Exception as e:

            errors += 1

            report_rows.append([
                split_name,
                image_path.name,
                "",
                "",
                "",
                f"ERROR: {e}"
            ])

        # Progress every 500 images
        if index % 500 == 0 or index == len(image_files):
            print(
                f"Progress: {index:,}/{len(image_files):,}"
            )


def main():

    print()
    print("=" * 60)
    print("VisDrone → YOLO Dataset Conversion")
    print("=" * 60)
    print("Original dataset will remain untouched.")
    print()

    for split_name, paths in DATASETS.items():
        process_split(split_name, paths)

    # Create conversion report
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = report_dir / "yolo_conversion_report.csv"

    with report_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "split",
            "image",
            "width",
            "height",
            "objects",
            "status"
        ])

        writer.writerows(report_rows)

    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(f"Images processed : {total_images:,}")
    print(f"Labels created   : {total_labels:,}")
    print(f"Objects converted: {total_objects:,}")
    print(f"Objects skipped  : {skipped_objects:,}")
    print(f"Errors           : {errors:,}")

    print()
    print(f"Output dataset:")
    print(f"{OUTPUT_ROOT}")

    print()
    print(f"Conversion report:")
    print(f"{report_file}")

    print("=" * 60)

    if errors == 0:
        print("STATUS: SUCCESS")
    else:
        print("STATUS: COMPLETED WITH ERRORS")


if __name__ == "__main__":
    main()