import cv2
import numpy as np
import pickle
import os
from ultralytics import YOLO

# ── LOAD BOTH YOLO MODELS ─────────────────────────────
plate_model = YOLO('best.pt')
char_model  = YOLO('char_best.pt')

print("✅ Plate detector loaded (best.pt)")
print("✅ Character detector loaded (char_best.pt)")


def detect_all_plates(image):
    """
    Detect ALL license plates in an image.
    Returns list of (plate_img, coords, conf) tuples.
    """
    h, w    = image.shape[:2]
    results = plate_model(image, conf=0.25, verbose=False)
    plates  = []

    for box in results[0].boxes:
        conf = float(box.conf)
        x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0].tolist()]
        pad = 8
        x1  = max(0, x1-pad);  y1 = max(0, y1-pad)
        x2  = min(w, x2+pad);  y2 = min(h, y2+pad)
        if x2 > x1 and y2 > y1:
            plates.append((image[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1), conf))

    # Sort by confidence descending
    plates.sort(key=lambda p: p[2], reverse=True)
    return plates


def detect_plate_region(image):
    """Single plate - keeps backward compatibility."""
    plates = detect_all_plates(image)
    if not plates:
        h, w = image.shape[:2]
        return image, (0, 0, w, h), 0.0
    return plates[0]

def detect_all_plates(image):
    """Detect ALL license plates in an image."""
    h, w    = image.shape[:2]
    results = plate_model(image, conf=0.25, verbose=False)
    plates  = []
    for box in results[0].boxes:
        conf = float(box.conf)
        x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0].tolist()]
        pad = 8
        x1  = max(0, x1-pad);  y1 = max(0, y1-pad)
        x2  = min(w, x2+pad);  y2 = min(h, y2+pad)
        if x2 > x1 and y2 > y1:
            plates.append((image[y1:y2, x1:x2], (x1, y1, x2-x1, y2-y1), conf))
    plates.sort(key=lambda p: p[2], reverse=True)
    return plates


def preprocess_plate(plate_img):
    """Preprocess plate image. Returns gray, binary, steps."""
    steps = {}
    steps['original'] = plate_img.copy()

    # Grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    steps['grayscale'] = gray.copy()

    # Resize to standard height
    target_h = 200
    scale    = target_h / gray.shape[0]
    target_w = int(gray.shape[1] * scale)
    gray     = cv2.resize(gray, (target_w, target_h))

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray  = clahe.apply(gray)
    steps['clahe'] = gray.copy()

    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 17, 17)

    # Threshold
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    steps['binary'] = binary.copy()

    return gray, binary, steps


def sort_chars_reading_order(char_detections):
    """Sort characters row by row, left to right."""
    if not char_detections:
        return char_detections

    avg_h = np.mean([y2-y1 for x1,y1,x2,y2,conf,cls
                     in char_detections])

    char_detections.sort(key=lambda c: c[1])

    rows        = []
    current_row = [char_detections[0]]

    for i in range(1, len(char_detections)):
        curr_y = char_detections[i][1]
        prev_y = current_row[-1][1]
        if abs(curr_y - prev_y) < avg_h * 0.6:
            current_row.append(char_detections[i])
        else:
            rows.append(current_row)
            current_row = [char_detections[i]]
    rows.append(current_row)

    for row in rows:
        row.sort(key=lambda c: c[0])

    return [char for row in rows for char in row]


def correct_plate_text(text):
    """Fix common misreads on Nepali plates."""
    if not text:
        return text
    corrected = list(text)
    for i, ch in enumerate(corrected):
        if ch == 'B':
            left_d  = i > 0 and corrected[i-1].isdigit()
            right_d = i < len(corrected)-1 and corrected[i+1].isdigit()
            if left_d and right_d:
                corrected[i] = '8'
    return ''.join(corrected)


def segment_and_recognize(plate_img, cnn_model, idx_to_char):
    """
    Full pipeline:
    1. Preprocess plate
    2. YOLO finds each character box
    3. Sort reading order
    4. CNN recognizes each character
    """
    gray, binary, steps = preprocess_plate(plate_img)
    ph, pw = plate_img.shape[:2]

    # YOLO character detection
    results   = char_model(plate_img, conf=0.30, verbose=False)
    annotated = plate_img.copy()

    char_detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = [round(float(v)) for v in box.xyxy[0].tolist()]
        conf = float(box.conf)
        cls  = int(box.cls)

        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(pw, x2); y2 = min(ph, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        char_detections.append((x1, y1, x2, y2, conf, cls))
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)

    # Filter small detections (removes IND text etc)
    if char_detections:
        heights  = [y2-y1 for x1,y1,x2,y2,conf,cls in char_detections]
        median_h = np.median(heights)
        char_detections = [c for c in char_detections
                           if (c[3]-c[1]) > median_h * 0.50]

    # Sort reading order
    char_detections = sort_chars_reading_order(char_detections)

    steps['annotated']  = annotated
    steps['char_count'] = len(char_detections)

    # CNN recognition
    plate_text  = ''
    confidences = []
    char_boxes  = []

    for (x1, y1, x2, y2, det_conf, cls) in char_detections:
        scale_x = gray.shape[1] / pw
        scale_y = gray.shape[0] / ph
        gx1 = int(x1 * scale_x); gy1 = int(y1 * scale_y)
        gx2 = int(x2 * scale_x); gy2 = int(y2 * scale_y)

        char_crop = gray[gy1:gy2, gx1:gx2]
        if char_crop.size == 0:
            continue

        _, char_bin = cv2.threshold(char_crop, 0, 255,
                                     cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        pad_v  = max(4, int(char_bin.shape[0] * 0.1))
        pad_h  = max(4, int(char_bin.shape[1] * 0.1))
        padded = cv2.copyMakeBorder(char_bin,
                                     pad_v, pad_v, pad_h, pad_h,
                                     cv2.BORDER_CONSTANT, value=0)
        resized    = cv2.resize(padded, (28, 28))
        normalized = resized / 255.0

        idx, conf = cnn_model.predict_with_confidence(normalized)
        char      = idx_to_char.get(idx, '?')

        plate_text  += char
        confidences.append(conf)
        char_boxes.append((x1, y1, x2, y2))

        cv2.putText(annotated, char,
                    (x1, max(0, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Post process
    plate_text = correct_plate_text(plate_text)
    avg_conf   = float(np.mean(confidences)) if confidences else 0.0

    # Fallback if YOLO found nothing
    if len(char_detections) == 0:
        print("⚠ No chars by YOLO — using fallback")
        plate_text, avg_conf, char_boxes = \
            _fallback_segment(gray, binary, cnn_model,
                              idx_to_char, annotated)

    return plate_text, avg_conf, steps, char_boxes


def _fallback_segment(gray, binary, cnn_model, idx_to_char, annotated):
    """Connected components fallback."""
    h, w   = gray.shape
    roi    = binary[int(h*0.35):, :]
    rh, rw = roi.shape

    num_labels, labels, stats, _ = \
        cv2.connectedComponentsWithStats(roi, connectivity=8)

    chars = []
    for i in range(1, num_labels):
        x    = stats[i, cv2.CC_STAT_LEFT]
        y    = stats[i, cv2.CC_STAT_TOP]
        cw   = stats[i, cv2.CC_STAT_WIDTH]
        ch   = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        ar   = cw / float(ch) if ch > 0 else 0

        if (ch > rh*0.20 and cw > 3 and
            cw < rw*0.35 and 0.05 < ar < 2.0 and area > 50):
            crop = roi[y:y+ch, x:x+cw]
            pad  = cv2.copyMakeBorder(crop, 4, 4, 4, 4,
                                       cv2.BORDER_CONSTANT, value=0)
            crop = cv2.resize(pad, (28,28)) / 255.0
            chars.append((x, crop, (x, int(h*0.35)+y, cw, ch)))

    chars.sort(key=lambda c: c[0])

    plate_text  = ''
    confidences = []
    char_boxes  = []

    for (x, crop, box) in chars:
        idx, conf = cnn_model.predict_with_confidence(crop)
        char      = idx_to_char.get(idx, '?')
        plate_text  += char
        confidences.append(conf)
        char_boxes.append(box)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return plate_text, avg_conf, char_boxes


def draw_plate_box(image, coords, plate_text,
                   matched=False, confidence=0):
    """Draw plate bounding box and label on image."""
    if coords is None:
        return image
    x, y, w, h = coords
    color = (0, 0, 255) if matched else (0, 255, 0)
    cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
    label = (f"!! ALERT: {plate_text}" if matched
             else f"{plate_text} ({confidence:.0f}%)")
    (tw, th), _ = cv2.getTextSize(label,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(image, (x, y-th-15), (x+tw+10, y), color, -1)
    cv2.putText(image, label, (x+5, y-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    return image