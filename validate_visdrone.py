from pathlib import Path

ANNOTATION_DIR = Path(
    "data/raw/VisDrone/train/VisDrone2019-DET-train/annotations"
)

EXPECTED_VALUES = 8

total_files = 0
total_rows = 0
bad_rows = 0

print("VisDrone Annotation Validator")
print("=" * 50)

for txt_file in ANNOTATION_DIR.glob("*.txt"):
    total_files += 1

    with txt_file.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            total_rows += 1

            # Remove whitespace and trailing commas
            cleaned = line.strip().rstrip(",")

            if not cleaned:
                continue

            values = [x.strip() for x in cleaned.split(",")]

            if len(values) != EXPECTED_VALUES:
                bad_rows += 1
                print(
                    f"BAD: {txt_file.name}:{line_number} "
                    f"-> {len(values)} values"
                )
                print(f"     {line.strip()}")

print()
print("=" * 50)
print(f"Annotation files : {total_files}")
print(f"Annotation rows  : {total_rows}")
print(f"Bad rows         : {bad_rows}")

if bad_rows == 0:
    print("STATUS: ALL ANNOTATIONS VALID")
else:
    print("STATUS: CHECK BAD ROWS")