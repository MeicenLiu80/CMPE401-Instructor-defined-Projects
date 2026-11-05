
from pathlib import Path
import shutil
from PIL import Image

ROOT = Path(r"D:\CMPE401_Project1")
SOURCE = ROOT / "datasets"
OUTPUT = SOURCE / "visdrone_yolo"

SPLITS = {
    "train": SOURCE / "VisDrone2019-DET-train",
    "val": SOURCE / "VisDrone2019-DET-val",
}

# VisDrone category IDs 1–10 mapped to YOLO IDs 0–9.
CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

# VisDrone IDs 0 (ignored regions) and 11 (others)
# are excluded from this 10-class detection task.
CATEGORY_MAP = {i: i - 1 for i in range(1, 11)}


def convert_split(split, source_dir):
    image_dir = source_dir / "images"
    annotation_dir = source_dir / "annotations"

    output_images = OUTPUT / "images" / split
    output_labels = OUTPUT / "labels" / split
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    converted = 0
    missing_annotations = 0
    invalid_boxes = 0
    ignored_objects = 0

    for image_path in images:
        annotation_path = annotation_dir / f"{image_path.stem}.txt"

        if not annotation_path.exists():
            missing_annotations += 1
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        yolo_lines = []

        for line in annotation_path.read_text(
            encoding="utf-8-sig"
        ).splitlines():
            if not line.strip():
                continue

            fields = line.split(",")
            if len(fields) < 8:
                invalid_boxes += 1
                continue

            try:
                x, y, w, h = map(float, fields[:4])
                score = int(fields[4])
                category = int(fields[5])
            except ValueError:
                invalid_boxes += 1
                continue

            # Keep valid annotated objects in categories 1–10.
            if score == 0 or category not in CATEGORY_MAP:
                ignored_objects += 1
                continue

            if w <= 0 or h <= 0 or width <= 0 or height <= 0:
                invalid_boxes += 1
                continue

            # Convert top-left xywh to normalized center xywh.
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            w_norm = w / width
            h_norm = h / height

            # Clip boxes to image boundaries.
            x1 = max(0.0, min(1.0, x / width))
            y1 = max(0.0, min(1.0, y / height))
            x2 = max(0.0, min(1.0, (x + w) / width))
            y2 = max(0.0, min(1.0, (y + h) / height))

            w_norm = x2 - x1
            h_norm = y2 - y1
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            if w_norm <= 0 or h_norm <= 0:
                invalid_boxes += 1
                continue

            class_id = CATEGORY_MAP[category]
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}"
            )

        shutil.copy2(
            image_path,
            output_images / image_path.name
        )
        (output_labels / f"{image_path.stem}.txt").write_text(
            "\n".join(yolo_lines),
            encoding="utf-8"
        )
        converted += 1

    print(f"\n[{split}]")
    print(f"Images converted: {converted}")
    print(f"Missing annotations: {missing_annotations}")
    print(f"Invalid boxes: {invalid_boxes}")
    print(f"Ignored objects: {ignored_objects}")


if __name__ == "__main__":
    for split, source_dir in SPLITS.items():
        if not (source_dir / "images").is_dir():
            raise FileNotFoundError(
                f"Image folder not found: {source_dir / 'images'}"
            )
        if not (source_dir / "annotations").is_dir():
            raise FileNotFoundError(
                f"Annotation folder not found: {source_dir / 'annotations'}"
            )
        convert_split(split, source_dir)

    print("\nConversion finished.")
    print(f"Output directory: {OUTPUT}")
    print("Class mapping:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  YOLO {i}: {name}")
