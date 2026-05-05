import cv2
from ultralytics import YOLO

model = YOLO("backend/model/final_weapon_model.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Weapon Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()