from ultralytics import YOLO

model = YOLO("runs/detect/modeltrain/weights/best.pt")
results = model.predict("F:\KIM CHI\MARKER\data\9a9d4b44-Img00875 - Copy.png", show = True, save = True)
