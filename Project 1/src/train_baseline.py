
from ultralytics import YOLO
import torch

DATA = r"D:\CMPE401_Project1\configs\visdrone.yaml"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11n.pt")

    model.train(
        data=DATA,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        project=r"D:\CMPE401_Project1\runs",
        name="yolo11n_baseline",
        seed=42,
        plots=True,
        save=True,
        patience=15,
    )

if __name__ == "__main__":
    main()
