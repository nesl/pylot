from ultralytics import YOLO

model = YOLO("yolo11x.pt", task="detect")

for i in range(20):
    model.predict("traffic-jam-getty.jpg", conf=0.1)
