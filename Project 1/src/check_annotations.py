
from pathlib import Path
from collections import Counter

root = Path(__file__).resolve().parents[1] / "datasets"

for split in ["train", "val"]:
    annotation_dir = root / f"VisDrone2019-DET-{split}" / "annotations"
    counts = Counter()

    if not annotation_dir.is_dir():
        print(f"Folder not found: {annotation_dir}")
        continue

    for file in annotation_dir.glob("*.txt"):
        for line in file.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue

            fields = line.split(",")
            if len(fields) < 8:
                counts["malformed"] += 1
                continue

            try:
                score = int(fields[4])
                category = int(fields[5])
            except ValueError:
                counts["malformed"] += 1
                continue

            counts[f"score={score}, category={category}"] += 1

    print(f"\n--- {split.upper()} ---")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")
