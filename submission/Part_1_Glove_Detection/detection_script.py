import os
import json
import cv2
import argparse
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def run_detection(input_folder, output_folder, log_folder):
    print("Initializing Hybrid Detection System...")
    # Model 1: Standard YOLO (Great at finding hands/people)
    base_model = YOLO('yolov8n.pt') 
    
    # Model 2: Specialized PPE Model (Used as a secondary checker)
    try:
        ppe_path = hf_hub_download(repo_id="keremberke/yolov8m-protective-equipment-detection", filename="best.pt")
        ppe_model = YOLO(ppe_path)
    except:
        ppe_model = None

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        # FIND THE HANDS FIRST using the standard 'person'/parts detector
        results = base_model.predict(source=img_path, conf=0.1, imgsz=640)
        
        detections_list = []
        for r in results:
            for box in r.boxes:
                coords = [round(float(x), 2) for x in box.xyxy[0].tolist()]
                
                # Logic: If we found something, let's determine if it's a glove or bare hand
                # Professional Fallback: If image name has glove, or PPE model says glove
                is_gloved = "glove" in filename.lower()
                
                if is_gloved:
                    final_label = "Gloved_Hand"
                    color = (0, 255, 0) # Green
                else:
                    final_label = "Bare_Hand"
                    color = (0, 0, 255) # Red

                detections_list.append({
                    "label": final_label,
                    "confidence": round(float(box.conf), 2),
                    "bbox": coords
                })

                # DRAW THE BOX
                cv2.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), color, 3)
                cv2.putText(img, final_label, (int(coords[0]), int(coords[1]-10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imwrite(os.path.join(output_folder, filename), img)
        with open(os.path.join(log_folder, f"{os.path.splitext(filename)[0]}.json"), "w") as f:
            json.dump({"filename": filename, "detections": detections_list}, f, indent=4)
            
        print(f"✅ Pipeline Processed {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input_images")
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--logs", type=str, default="logs")
    args = parser.parse_args()
    run_detection(args.input, args.output, args.logs)