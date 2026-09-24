# Phase 4 — Number Plate Detection

## Overview
Phase 4 deploys a YOLOv8 deep learning object detector integrated with an OpenCV morphological cascade pipeline to detect vehicle regions and pinpoint license plate bounding boxes.

## Architecture & Algorithm

### 1. Vehicle Detection
- Utilizes **YOLOv8 Nano (yolov8n)** object detection model trained on COCO dataset.
- Filters class detections for motor vehicles (`car`, `bus`, `truck`, `motorcycle`).

### 2. License Plate Region Localization
- **Morphological Filtering**: Applies Black-Hat transformation to isolate dark registration characters against high-contrast license plate background.
- **Sobel Gradient Analysis**: Calculates horizontal gradients ($\nabla_x$) to highlight high-density character boundaries.
- **Contour Filtering**: Filters candidate regions by aspect ratio constraint ($2.0 \le \frac{\text{width}}{\text{height}} \le 7.0$) and spatial position.

## Output Artefacts
- `detection_results.json`: Bounding box coordinates for vehicles and plates.
- `detected_output/`: Full vehicle images annotated with bounding box overlays.
- `cropped_plates/`: High-resolution cropped license plate image snippets ready for OCR processing in Phase 5.

## Execution
```bash
python phase_4_number_plate_detection/plate_detector.py
```
