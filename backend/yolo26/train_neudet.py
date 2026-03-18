from ultralytics import YOLO

model = YOLO("yolo26s.pt")   # detection model

model.train(
    data="neu_det/data.yaml",
    epochs=30,
    imgsz=512,
    batch=4,
    device=0,
    patience=20
)
