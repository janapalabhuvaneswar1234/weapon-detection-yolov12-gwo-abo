import cv2
import os
import shutil
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()

# Directories
UPLOAD_DIR = "backend/static/uploads"
OUTPUT_DIR = "backend/static/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="backend/static"), name="static")

model = YOLO("backend/model/final_weapon_model.pt")

# ================= IMAGE =================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(file_path)

    output_path = os.path.join(OUTPUT_DIR, file.filename)
    results[0].save(filename=output_path)

    return {"output": f"http://127.0.0.1:8000/static/outputs/{file.filename}"}


# ================= VIDEO =================
@app.post("/predict-video")
async def predict_video(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    output_path = os.path.join(OUTPUT_DIR, file.filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            frame = results[0].plot()

        out.write(frame)

    cap.release()
    out.release()

    return {"output": f"http://127.0.0.1:8000/static/outputs/{file.filename}"}


# ================= LIVE WEBCAM =================
def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))

        results = model(frame)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            
            filtered_boxes = []

            for box in boxes:
                conf = float(box.conf)

            if conf > 0.5:   
                filtered_boxes.append(box)
            if len(filtered_boxes) > 0:
                frame = results[0].plot()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')