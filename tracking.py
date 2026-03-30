import ultralytics
from ultralytics import YOLO

if __name__ =="__main__":
    model = YOLO("modeltrain.pt")
    result = model.train(source = "",#file file.mp4 name
                        tracker = "bytetrack.yaml",
                        save = True,
                        show_conf = True,
                        )