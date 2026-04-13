from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format = 'engine',
             imgsz = 640,
             haft = TRUE,
             dynamic = True, #cho phép kích thước đầu vào động. xử lý nhiều hình ảnh ở các kích thước khác nhau
             workspace = None, #None=TensorRT tự phân bổ đến mức tối đa của thiết bị
             batch = 1,
             data = "dataset.yaml",
             device=0) #or device=dla:0 / device=dla:1