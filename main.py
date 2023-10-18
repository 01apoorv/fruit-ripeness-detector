import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('runs/detect/train5/weights/best.pt')
vid = cv2.VideoCapture(1)

while True:
    ret, frame = vid.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    res = model(frame, device='mps')[0]
    bbox = np.array(res.boxes.xyxy.cpu(), dtype='int')
    classes = np.array(res.boxes.cls.cpu(), dtype='int')
    for b, c in zip(bbox, classes):
        (x, y, x2, y2) = b
        cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, res.names[c], (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

vid.release()
cv2.destroyAllWindows()