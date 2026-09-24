# Phase 5 — Number Plate OCR & Character Post-Processing

## Overview
Phase 5 implements Optical Character Recognition (OCR) to extract vehicle registration numbers from cropped license plate images and applies heuristic post-processing rules.

## Processing Steps

### 1. Plate Preprocessing
- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization to normalize illumination variations across license plate crops.
- **Binarization**: Otsu thresholding combined with Gaussian smoothing to create clean binary character masks.

### 2. Multi-Engine OCR Recognition
- **EasyOCR / PyTesseract Engines**: Performs character recognition on preprocessed image crops.

### 3. Character Post-Processing & Verification
Standard license plate structure (e.g. `TN 01 AB 1234`):
- **Zone 1 (State Code)**: First 2 positions converted to alpha chars (e.g. `0` $\rightarrow$ `O`, `8` $\rightarrow$ `B`).
- **Zone 2 (District Code)**: Next 2 positions converted to numeric chars (e.g. `O` $\rightarrow$ `0`, `S` $\rightarrow$ `5`).
- **Zone 3 (Unique Vehicle ID)**: Final 4 positions constrained to numeric digits (e.g. `B` $\rightarrow$ `8`, `Z` $\rightarrow$ `2`).

## Output Files
- `phase_5_number_plate_ocr/ocr_results.json`: Extracted registration numbers, raw OCR text, and confidence scores.

## Execution
```bash
python phase_5_number_plate_ocr/ocr_engine.py
```
