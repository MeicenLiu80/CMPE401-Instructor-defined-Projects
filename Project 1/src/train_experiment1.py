from pathlib import Path
import os
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "configs" / "visdrone.yaml"
RUN_NAME = "yolo11n_exp1_960"

def main():
    os.chdir(PROJECT_ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA}")

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Experiment: YOLO11n, imgsz=960, batch=4")

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=960,
        batch=4,
        device=0,
        workers=2,
        project=str(PROJECT_ROOT / "runs"),
        name=RUN_NAME,
        seed=42,
        plots=True,
        save=True,
        save_period=10,
        patience=15,
        exist_ok=False,
    )

if __name__ == "__main__":
    main()
