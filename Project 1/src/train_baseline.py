from pathlib import Path
import os
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "configs" / "visdrone.yaml"

def main():
    os.chdir(PROJECT_ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        project=str(PROJECT_ROOT / "runs"),
        name="yolo11n_baseline",
        seed=42,
        plots=True,
        save=True,
        patience=15,
        exist_ok=False,
    )

if __name__ == "__main__":
    main()
