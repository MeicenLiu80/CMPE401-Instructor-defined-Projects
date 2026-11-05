
from pathlib import Path

ROOT = Path(r"D:\CMPE401_Project1\datasets\visdrone_yolo")

for split in ("train", "val"):
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    print(f"\n--- {split.upper()} ---")

    if not image_dir.is_dir() or not label_dir.is_dir():
        print("ERROR: Image or label directory not found.")
        print("Images:", image_dir)
        print("Labels:", label_dir)
        continue

    images = {
        p.stem for p in image_dir.iterdir()
        if p.is_file()
    }
    labels = {
        p.stem for p in label_dir.glob("*.txt")
    }

    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Images without labels:", len(images - labels))
    print("Labels without images:", len(labels - images))

    total_boxes = 0
    empty_labels = 0
    invalid_entries = 0

    for file in label_dir.glob("*.txt"):
        lines = [
            line.strip()
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        if not lines:
            empty_labels += 1

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                invalid_entries += 1
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_entries += 1
                continue

            if (
                not 0 <= class_id < 10
                or not all(0 <= v <= 1 for v in (x, y, w, h))
                or w <= 0
                or h <= 0
            ):
                invalid_entries += 1
                continue

            total_boxes += 1

    print("Empty label files:", empty_labels)
    print("Valid bounding boxes:", total_boxes)
    print("Invalid entries:", invalid_entries)
