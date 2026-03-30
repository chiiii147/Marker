import ultralytics
from ultralytics import YOLO
if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data = "dataset.yaml",
        epochs = 1000,
        imgsz = 640,
        device = 0, #use GPU
        resume = True,
        patience = 150, #so epochs sau do model se dung train neu kq k cai thien
        save = True,
        workers = 4,
        name = "modeltrain",
        single_cls = True,
        profile = True, # cho phep lap profile ONXX va TensorRT
        box = 10.0,
        cls = 0.0,
        dropout = 40.0,
        #data-augmentation
        hvs_h = 0.1,
        hvs_s = 0.5,
        hvs_v = 0.5,
        degrees = 15.0,
        translate = 0.7,
        scale = 0.4,
        shear = 10.0,
        perspective = 0.0005,
        flipud = 0.3,
        fliplr = 0.0,
        mosaic = 0.2,
        mixup = 0.2,
        copy_paste = 0.05,
        auto_augment = None,
        erasing = 0.5,
    )