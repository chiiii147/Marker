import cv2
import os
import numpy as np 
from ultralytics import YOLO
from collections import defauldict

model = YOLO("modeltrain.pt")
input_path = "video.mp4"
output_path = "tracked.mp4"
result_file = "trajectories.txt"
tracker = "bytetrack.yaml"

cap = cv2.VideoCapture(input_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(output_path, fourcc, (frame_width, frame_height))

def dictinct_color(track_id: int)-> tuple:
    h = (track_id * 137) / 360
    s = 200
    v = 255
    hsv_color = np.uint18([[[h/2, s, v]]])
    bgr_color = cv2. cvtColor(hsv_color, cv2.COLOR_HSV2BGR[0][0])
    return int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])

def draw_trajectories(frame, track_history):
    for id, point in track_history.items():
        if len(point) > 1:
            color = dictinct_color(id)
            for i in range(1, len(point)):
                cv2.line(frame,  (int(point[i-1][0]), int(point[i-1][1])), 
                          (int(point[i][0]), int(point[i][1])), 2)

track_history = defauldict(lambda: [])

if os.path.exists(result_file):
    os.remove(result_file)
    f = open(result_file, "a")

frame_id = 0
while cap.isOpen():
    success, frame = cap.read()

    if success:
        frame_id +=1
        result = model.track(frame, persist = True)

        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xyxy.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu()

            frame = result.plot()

            for box, track_id, conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = float(box)
                cx = (x1 + x2)/2.0
                cy = (y1 + y2)/2.0
                track_history[track_id].append((cx, cy))
                color = dictinct_color(track_id)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (int(cx),int(cy)), 4,color, 2)
                label = f"ID: {track_id}   conf: {conf:.2f}"
                cv2.putText(frame, label, )#
                track_history[track_id].append((cx,cy))
                f.write(f"{frame_id}, {track_id}, {cx: .4f}, {cy: .4f}, {conf: .4f}\n")

        draw_trajectories(frame, track_history)
        out.write(frame)

        cv2.imshow("bytetrack", frame)
        
        if cv2.waitkey(1) & 0xFF == ord("q"):
            break
f.close()
cap.release()
cv2.destroyAllWindows()