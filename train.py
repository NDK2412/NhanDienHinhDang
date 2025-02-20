from torchgen.api.types import deviceT
from ultralytics import YOLO
import torch

def main():
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11s.pt")
    torch.backends.cudnn.benchmark = True  # Tăng tốc độ GPU
    model.model = torch.compile(model.model)


    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 50 epochs
    results = model.train(
    data="coco8.yaml",
    epochs=50,
    batch=4,
    imgsz=640,
    device=0,
    amp=True
)

if __name__ == '__main__':
    main()
