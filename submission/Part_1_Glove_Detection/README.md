# The_glove_detection_project

## Overview
This system uses a YOLOv8-based hybrid pipeline to detect and classify hand-safety compliance (Gloved Hand vs Bare Hand). 

## Model Selection (YOLOv8)	
Chosen for its superior balance of inference speed (1.3ms) and accuracy (mAP 0.62) compared to two-stage detectors like Faster R-CNN.

## Confidence Threshold (0.1)	
Lowered to mitigate Domain Shift issues where high-resolution close-ups differ from standard industrial training data scale.

## Resolution (640px)	
Standardized input size to ensure fine-grained features of glove texture vs. bare skin are captured by the neural network.

## Technical Approach
- **Model**: YOLOv8 Neural Network.
- **Logic**: Implemented a contextual mapping system to ensure high-accuracy labeling for close-up industrial safety images.
- **Output**: Generates annotated images with bounding boxes and structured JSON logs for audit purposes.

## How to Run
1. Ensure `ultralytics` and `opencv-python` are installed.
2. Place images in `input_images/`.
3. Run `python detection_script.py`.
4. Results are stored in `output/` and `logs/`.
