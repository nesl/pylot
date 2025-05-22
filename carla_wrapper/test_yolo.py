# from ultralytics import YOLO

# model = YOLO("yolo11x.pt", task="detect")

# for i in range(20):
#     model.predict("traffic-jam-getty.jpg", conf=0.1)


from ultralytics import YOLO
import torch

# Debug print
print("CUDA available:", torch.cuda.is_available())
print("Devices:", torch.cuda.device_count())

# Force YOLO to use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolov8x.pt")
model.to(device)
