
from pathlib import Path
import csv
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "yolo11n_baseline"
OUT = ROOT / "results" / "baseline_summary.csv"

OUT.parent.mkdir(parents=True, exist_ok=True)

# Read the actual training configuration.
args_file = RUN / "args.yaml"
with args_file.open("r", encoding="utf-8") as f:
    args = yaml.safe_load(f)

# Values from the final validation of best.pt.
metrics = {
    "precision": 0.437,
    "recall": 0.329,
    "mAP50": 0.314,
    "mAP50_95": 0.176,
}

record = {
    "experiment": "yolo11n_baseline",
    "model": args.get("model"),
    "dataset": "VisDrone2019-DET",
    "train_images": 6471,
    "val_images": 548,
    "epochs_configured": args.get("epochs"),
    "imgsz": args.get("imgsz"),
    "batch": args.get("batch"),
    "seed": args.get("seed"),
    "optimizer_setting": args.get("optimizer"),
    "pretrained": args.get("pretrained"),
    **metrics,
    "best_weights": str(RUN / "weights" / "best.pt"),
    "last_weights": str(RUN / "weights" / "last.pt"),
    "results_csv": str(RUN / "results.csv"),
    "results_plot": str(RUN / "results.png"),
}

with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=record.keys())
    writer.writeheader()
    writer.writerow(record)

print("Baseline summary saved to:")
print(OUT)
print("\nRecorded metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

