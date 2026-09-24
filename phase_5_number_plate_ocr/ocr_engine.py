import os
import re
import cv2
import json
import numpy as np

# Check available OCR libraries
EASYOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    pass

class NumberPlateOCREngine:
    """
    OCR Engine for vehicle license plate character extraction.
    Includes adaptive contrast enhancement, noise reduction, deskewing,
    and post-processing regex character correction.
    """
    def __init__(self, languages=['en']):
        self.easyocr_reader = None
        if EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(languages, gpu=False)
                print("[Phase 5] Initialized EasyOCR Reader engine.")
            except Exception as e:
                print(f"[Phase 5] EasyOCR init note: {e}")

    def preprocess_plate_for_ocr(self, cropped_plate):
        """
        Preprocesses cropped plate image for optimal OCR character recognition.
        """
        if cropped_plate is None or cropped_plate.size == 0:
            return None, None
            
        # Convert to Grayscale
        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY) if len(cropped_plate.shape) == 3 else cropped_plate
        
        # Upscale plate image for small font clarity
        h, w = gray.shape[:2]
        scaled = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        
        # Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)
        
        # Otsu Adaptive Thresholding
        blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        return enhanced, thresh

    def post_process_registration_number(self, text):
        """
        Applies character post-processing to rectify common OCR confusions:
        - O -> 0, I -> 1, Z -> 2, S -> 5, B -> 8 in numeric zones
        - 0 -> O, 1 -> I, 8 -> B in state code / alpha zones
        Format expected: SS DD AA NNNN (e.g. TN 01 AB 1234)
        """
        if not text:
            return ""
            
        # Clean special chars and split words
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        if len(cleaned) < 6:
            return cleaned
            
        state_code_map = {'0': 'O', '1': 'I', '8': 'B', '5': 'S'}
        num_map = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7'}
        
        chars = list(cleaned)
        
        # First 2 chars: State alpha code (e.g., TN, KA, MH, DL, HR)
        for idx in range(min(2, len(chars))):
            if chars[idx] in state_code_map:
                chars[idx] = state_code_map[chars[idx]]
                
        # Next 2 chars: District numeric code (e.g., 01, 05, 12, 26)
        for idx in range(2, min(4, len(chars))):
            if chars[idx] in num_map:
                chars[idx] = num_map[chars[idx]]
                
        # Last 4 chars: Unique numeric ID (e.g., 1234, 5678)
        if len(chars) >= 8:
            for idx in range(len(chars) - 4, len(chars)):
                if chars[idx] in num_map:
                    chars[idx] = num_map[chars[idx]]
                    
        formatted = "".join(chars)
        
        # Format with spaces (e.g., TN 01 AB 1234)
        if len(formatted) >= 9:
            return f"{formatted[:2]} {formatted[2:4]} {formatted[4:-4]} {formatted[-4:]}"
        elif len(formatted) >= 8:
            return f"{formatted[:2]} {formatted[2:4]} {formatted[4:5]} {formatted[-4:]}"
        else:
            return formatted

    def extract_text(self, cropped_plate):
        """
        Extracts registration text using multi-engine OCR fallback.
        """
        enhanced, thresh = self.preprocess_plate_for_ocr(cropped_plate)
        raw_text = ""
        confidence = 0.0

        # 1. EasyOCR
        if self.easyocr_reader is not None and enhanced is not None:
            try:
                results = self.easyocr_reader.readtext(enhanced)
                if results:
                    text_parts = [res[1] for res in results if res[2] > 0.2]
                    confidences = [res[2] for res in results if res[2] > 0.2]
                    raw_text = " ".join(text_parts)
                    confidence = float(np.mean(confidences)) if confidences else 0.0
            except Exception as e:
                pass

        # 2. PyTesseract Fallback
        if (not raw_text or len(raw_text) < 4) and TESSERACT_AVAILABLE and thresh is not None:
            try:
                config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                tess_text = pytesseract.image_to_string(thresh, config=config).strip()
                if len(tess_text) > len(raw_text):
                    raw_text = tess_text
                    confidence = 0.75
            except Exception as e:
                pass

        # Post-process extracted text
        processed_text = self.post_process_registration_number(raw_text)
        return processed_text, raw_text, confidence

def run_ocr_pipeline(crop_dir="phase_4_number_plate_detection/cropped_plates", output_dir="phase_5_number_plate_ocr"):
    """
    Runs OCR engine on cropped plate images and saves extracted registration numbers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ocr = NumberPlateOCREngine()
    
    if not os.path.exists(crop_dir):
        raise FileNotFoundError(f"Cropped plates folder '{crop_dir}' not found. Run Phase 4 first.")
        
    crop_files = [f for f in os.listdir(crop_dir) if f.endswith(('.jpg', '.png'))]
    ocr_results = []
    
    for filename in crop_files:
        crop_path = os.path.join(crop_dir, filename)
        img = cv2.imread(crop_path)
        
        extracted_reg, raw_text, conf = ocr.extract_text(img)
        
        ocr_results.append({
            "cropped_file": filename,
            "extracted_registration_number": extracted_reg,
            "raw_ocr_text": raw_text,
            "confidence": round(conf, 4)
        })
        
    output_json = os.path.join(output_dir, "ocr_results.json")
    with open(output_json, "w") as f:
        json.dump(ocr_results, f, indent=2)
        
    print(f"[Phase 5] Processed OCR for {len(crop_files)} license plates. Results saved to '{output_json}'.")
    return ocr_results

if __name__ == "__main__":
    run_ocr_pipeline()
