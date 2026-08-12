from pathlib import Path
from PIL import Image
import csv
import statistics

IMAGE_DIR = Path(
    "data/raw/VisDrone/train/VisDrone2019-DET-train/images"
)

OUTPUT_DIR = Path("reports")
OUTPUT_FILE = OUTPUT_DIR / "image_resolution_analysis.csv"

widths = []
heights = []
records = []

for image_path in IMAGE_DIR.glob("*.jpg"):
    try:
        with Image.open(image_path) as img:
            width, height = img.size

        widths.append(width)
        heights.append(height)

        records.append([
            image_path.name,
            width,
            height,
            width * height
        ])

    except Exception as e:
        print(f"ERROR: {image_path.name} -> {e}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "image_name",
        "width",
        "height",
        "total_pixels"
    ])

    writer.writerows(records)

print("=" * 55)
print("VisDrone Image Resolution Analysis")
print("=" * 55)

print(f"Images analyzed : {len(records):,}")
print(f"Minimum width   : {min(widths):,} px")
print(f"Maximum width   : {max(widths):,} px")
print(f"Average width   : {statistics.mean(widths):,.1f} px")
print(f"Minimum height  : {min(heights):,} px")
print(f"Maximum height  : {max(heights):,} px")
print(f"Average height  : {statistics.mean(heights):,.1f} px")

print("=" * 55)
print(f"Report: {OUTPUT_FILE}")
print("STATUS: SUCCESS")