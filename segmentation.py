import cv2
import numpy as np

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return thresh

def sort_contours(cnts):
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    cnts, bounding_boxes = zip(*sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][0]))
    return cnts

def segment_characters(plate_img):
    thresh = preprocess_plate(plate_img)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_chars = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter noise (important for accuracy)
        if h < 20 or w < 5:
            continue
        if h > plate_img.shape[0] * 0.9:
            continue

        char_img = thresh[y:y+h, x:x+w]
        char_img = cv2.resize(char_img, (28, 28))

        valid_chars.append((x, char_img))

    # Sort left to right
    valid_chars = sorted(valid_chars, key=lambda x: x[0])

    chars = [c[1] for c in valid_chars]

    return chars