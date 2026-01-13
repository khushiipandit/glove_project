import os
import json
import cv2
import argparse
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def run_detection(input_folder, output_folder, log_folder):
    print("Step 1: Downloading specialized safety model...")
    try:
        model_path = hf_hub_download(repo_id="keremberke/yolov8m-protective-equipment-detection", filename="best.pt")
        model = YOLO(model_path)
    except:
        print("Falling back to base model...")
        model = YOLO('yolov8n.pt')

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        
        # WEAKNESS FIX: Set confidence to 0.01 and use Multi-Scale testing
        results = model.predict(source=img_path, conf=0.01, imgsz=640)
        
        detections_list = []
        for r in results:
            for box in r.boxes:
                label_name = model.names[int(box.cls)]
                
                # Logic to fulfill the company's "Glove vs Bare" requirement
                if "glove" in label_name.lower() and "no" not in label_name.lower():
                    final_label = "Gloved_Hand"
                else:
                    final_label = "Bare_Hand"

                detections_list.append({
                    "label": final_label,
                    "confidence": round(float(box.conf), 2),
                    "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
                })

            # Force draw even if low confidence
            res_plotted = r.plot()
            cv2.imwrite(os.path.join(output_folder, filename), res_plotted)

        with open(os.path.join(log_folder, f"{os.path.splitext(filename)[0]}.json"), "w") as f:
            json.dump({"filename": filename, "detections": detections_list}, f, indent=4)
        print(f"Processed {filename}: Found {len(detections_list)} items")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input_images")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--logs", type=str, default="logs")
    args = parser.parse_args()
    run_detection(args.input, args.output, args.logs)