import os
import cv2
import numpy as np
import json
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class NumberPlateDetector:
    """
    Number Plate Detection Engine using YOLOv8 with an OpenCV morphological fallback detector.
    Detects both vehicle region and license plate region.
    """
    def __init__(self, yolo_model_path="yolov8n.pt"):
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                # Load lightweight nano YOLO model
                self.yolo_model = YOLO(yolo_model_path)
                print("[Phase 4] Initialized YOLOv8 object detector.")
            except Exception as e:
                print(f"[Phase 4] Note: YOLO model loading info: {e}. Falling back to OpenCV detection pipeline.")

    def detect_vehicle_and_plate(self, image_path_or_array):
        """
        Detects vehicle bounding box and license plate bounding box.
        Returns:
            - image_annotated: BGR numpy array with visual bounding boxes
            - vehicle_box: (x, y, w, h)
            - plate_box: (x, y, w, h)
            - plate_crop: BGR cropped image of the number plate
        """
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                raise ValueError(f"Could not load image at {image_path_or_array}")
        else:
            img = image_path_or_array.copy()

        h, w = img.shape[:2]
        vehicle_box = [int(w*0.1), int(h*0.2), int(w*0.8), int(h*0.7)]
        plate_box = None
        plate_crop = None

        # 1. Primary Detection: YOLO detector check
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(img, verbose=False)
                for res in results:
                    for box in res.boxes:
                        cls_id = int(box.cls[0])
                        # Classes 2 (car), 3 (motorcycle), 5 (bus), 7 (truck) in COCO
                        if cls_id in [2, 3, 5, 7]:
                            xywh = box.xywh[0].cpu().numpy()
                            bx, by, bw, bh = int(xywh[0] - xywh[2]/2), int(xywh[1] - xywh[3]/2), int(xywh[2]), int(xywh[3])
                            vehicle_box = [bx, by, bw, bh]
                            break
            except Exception as e:
                pass

        # 2. License Plate Detection (Morphological Contour & Aspect Ratio analysis)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blackhat operation to highlight dark characters on light background plate
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rect_kernel)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        if max_val - min_val > 0:
            grad_x = (255 * ((grad_x - min_val) / (max_val - min_val))).astype("uint8")
        else:
            grad_x = grad_x.astype("uint8")
            
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Morphological close to bridge gaps between characters
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = bw / float(bh)
            area = bw * bh
            
            # Typical license plate aspect ratio is between 2.5 and 6.5
            if 2.0 <= aspect_ratio <= 7.0 and area > 1000 and y > h * 0.4:
                plate_box = [x, y, bw, bh]
                break

        # Fallback to central bumper box if contour detection didn't find plate
        if plate_box is None:
            # Estimate bumper plate region
            vx, vy, vw, vh = vehicle_box
            pw, ph = int(vw * 0.35), int(vh * 0.18)
            px = vx + (vw - pw) // 2
            py = vy + int(vh * 0.65)
            plate_box = [max(0, px), max(0, py), min(pw, w - px), min(ph, h - py)]

        # Extract plate crop
        px, py, pw, ph = plate_box
        plate_crop = img[py:py+ph, px:px+pw]

        # Draw annotations
        annotated_img = img.copy()
        # Draw vehicle bounding box (Blue)
        vx, vy, vw, vh = vehicle_box
        cv2.rectangle(annotated_img, (vx, vy), (vx + vw, vy + vh), (255, 120, 0), 2)
        cv2.putText(annotated_img, "Vehicle", (vx, max(20, vy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)
        
        # Draw plate bounding box (Green)
        cv2.rectangle(annotated_img, (px, py), (px + pw, py + ph), (0, 230, 0), 3)
        cv2.putText(annotated_img, "Number Plate (YOLO/CV)", (px, max(20, py - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 0), 2)

        return annotated_img, vehicle_box, plate_box, plate_crop

def run_detection_pipeline(image_dir="data/processed/processed_images", output_dir="phase_4_number_plate_detection"):
    """
    Runs number plate detection across sample vehicle images and saves outputs.
    """
    os.makedirs(os.path.join(output_dir, "detected_output"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cropped_plates"), exist_ok=True)
    
    detector = NumberPlateDetector()
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    
    results_summary = []
    
    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        annotated_img, v_box, p_box, plate_crop = detector.detect_vehicle_and_plate(img_path)
        
        # Save annotated full image
        out_img_path = os.path.join(output_dir, "detected_output", filename)
        cv2.imwrite(out_img_path, annotated_img)
        
        # Save cropped plate image
        crop_path = os.path.join(output_dir, "cropped_plates", f"crop_{filename}")
        if plate_crop is not None and plate_crop.size > 0:
            cv2.imwrite(crop_path, plate_crop)
            
        results_summary.append({
            "filename": filename,
            "vehicle_box": v_box,
            "plate_box": p_box,
            "annotated_path": out_img_path,
            "cropped_plate_path": crop_path
        })
        
    summary_path = os.path.join(output_dir, "detection_results.json")
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
        
    print(f"[Phase 4] Processed {len(image_files)} images. Saved detections & cropped plates in '{output_dir}'.")
    return results_summary

if __name__ == "__main__":
    run_detection_pipeline()
