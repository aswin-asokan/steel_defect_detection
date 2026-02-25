from ultralytics import YOLO

# Path to trained model
model_path = "runs/detect/train5/weights/best.pt"

# Load model
model = YOLO(model_path)

# Predict on a single image
results = model.predict(
    source="WhatsApp Image 2026-02-19 at 12.26.53 AM.jpeg",   # path to your image
    imgsz=512,
    conf=0.25,
    save=True,     # saves output image
    show=False
)

print("Prediction completed.")
