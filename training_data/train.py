import ultralytics
from ultralytics import YOLO

if __name__ =="__main__":
    model = YOLO("yolov8n.pt")
    result = model.train(data = "dataset.yaml",
                         epochs = 100,
                         imgsz = 640,
                         save = True,
                         device = "0",
                         workers = 4,
                         multi_scale = 0.5,
                         profile = True,
                         weight_decay = 0.0005,
                         dropout = 0.05,
                         val = True,
                         plots = True,

                         hsv_h = 0.02,
                         hsv_s = 0.4,
                         hsv_v = 0.4,
                         degrees = 10,
                         translate = 0.1,
                         scale = 0.3,
                         shear = 3,
                         perspective = 0.0005,
                         fliplr = 0.2,
                         mosaic = 0.2,
                         mixup = 0.05,
                         cutmix = 0.05,
                         erasing = 0.05,
                         auto_augment = None
                        )