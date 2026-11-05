from pathlib import Path
import os
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "configs" / "visdrone.yaml"
RUN_NAME = "yolo11n_control_640_b4_nb"

def main():
    os.chdir(PROJECT_ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training was not started.")
    if not DATA.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA}")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Run name:", RUN_NAME)

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(DATA),
        epochs=50,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,
        seed=42,
        plots=True,
        save=True,
        save_period=10,
        patience=15,
        project=str(PROJECT_ROOT / "runs"),
        name=RUN_NAME,
        exist_ok=False,
    )

if __name__ == "__main__":
    main()
