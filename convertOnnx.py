from ultralytics import YOLO
model = YOLO('runs/detect/train-14/weights/best.pt') #change trained model
model.export(format = 'onnx',
             imgsz = 640,
             dynamic = True,
             nms = True,
             batch = 1,
             device = 0)