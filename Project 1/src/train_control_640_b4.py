from pathlib import Path
from ultralytics import YOLO
import torch

PROJECT_DIR = Path(r"D:\CMPE401_Project1")
DATA_YAML = PROJECT_DIR / "configs" / "visdrone.yaml"
RUN_DIR = PROJECT_DIR / "runs" / "yolo11n_control_640_b4"

if RUN_DIR.exists():
    raise FileExistsError(
        f"Run directory already exists: {RUN_DIR}\n"
        "Please inspect it before running; do not overwrite existing results."
    )

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Training stopped.")

model = YOLO("yolo11n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=50,
    imgsz=640,
    batch=4,
    device=0,
    workers=2,
    seed=42,
    plots=True,
    save=True,
    save_period=10,
    patience=15,
    project=str(PROJECT_DIR / "runs"),
    name="yolo11n_control_640_b4",
    exist_ok=False,
)

print("Training completed.")
print("Results saved to:", RUN_DIR)
