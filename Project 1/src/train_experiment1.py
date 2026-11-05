from ultralytics import YOLO
import torch

DATA = r"D:\CMPE401_Project1\configs\visdrone.yaml"

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Experiment: YOLOv11n, imgsz=960, batch=4")

    model = YOLO("yolo11n.pt")

    model.train(
        data=DATA,
        epochs=50,
        imgsz=960,
        batch=4,
        device=0,
        workers=4,
        project=r"D:\CMPE401_Project1\runs",
        name="yolo11n_exp1_img960",
        seed=42,
        plots=True,
        save=True,
        patience=15,
    )

if __name__ == "__main__":
    main()