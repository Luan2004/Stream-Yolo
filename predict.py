from ultralytics import YOLO

model = YOLO("y8s128helmet.pt")

model.predict(
    source="cctv_helmet10.mp4",
    imgsz=640,
    conf=0.5,
    save=True
)