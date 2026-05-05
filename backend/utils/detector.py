import os
from ultralytics import YOLO

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "model", "final_weapon_model.pt")

# Load model once
model = YOLO(MODEL_PATH)


def detect_image(image_path):
    results = model(image_path)

    # Create output path
    output_path = image_path.replace("uploads", "outputs")

    # Save result
    results[0].save(filename=output_path)

    return os.path.basename(output_path)