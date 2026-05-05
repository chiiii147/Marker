from ultralytics import YOLO
model = YOLO('yolov8n.pt') #change trained model
model.export(format = 'onnx',
             imgsz = 640,
             dynamic = True,
             nms = True,
             batch = 1,
             device = 0)