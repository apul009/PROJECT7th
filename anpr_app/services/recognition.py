import os
import threading
import base64
import datetime
import cv2
from flask import session

from ..database import get_db
from ..config import EMAIL_ENABLED, EMAIL_PASSWORD, EMAIL_SENDER
from yolo_detector import detect_all_plates, draw_plate_box, detect_plate_region, segment_and_recognize
from label_map import load_label_map
from cnn_scratch import SimpleCNN

CHAR_TO_IDX, IDX_TO_CHAR, NUM_CLASSES, ALL_CLASSES = load_label_map('label_map.pkl')
model = SimpleCNN(num_classes=NUM_CLASSES)
if os.path.exists('model.pkl'):
    model.load('model.pkl')
else:
    print('model.pkl not found!')


def check_watchlist(plate_text):
    with get_db() as db:
        row = db.execute("SELECT * FROM watchlist WHERE plate=?", (plate_text,)).fetchone()
    return dict(row) if row else None


def save_image(image, filename):
    path = os.path.join('static', 'uploads', filename)
    cv2.imwrite(path, image)
    return path


def recognize_all_plates(image):
    all_plates = detect_all_plates(image)
    annotated = image.copy()
    results = []

    if not all_plates:
        return [], annotated

    for plate_img, coords, det_conf in all_plates:
        if plate_img is None or plate_img.size == 0:
            continue

        plate_text, confidence, steps, char_boxes = segment_and_recognize(plate_img, model, IDX_TO_CHAR)
        if not plate_text or plate_text in ["NOT DETECTED", "NO CHARS", ""]:
            continue

        matched = check_watchlist(plate_text)
        annotated = draw_plate_box(annotated, coords, plate_text, matched is not None, confidence)
        results.append({
            'plate': plate_text,
            'confidence': round(confidence, 1),
            'det_conf': round(det_conf * 100, 1),
            'matched': matched is not None,
            'info': matched,
            'coords': coords,
            'steps': steps,
        })

    return results, annotated


def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def log_detection(plate, matched, confidence, timestamp, image_path, detected_by):
    with get_db() as db:
        db.execute(
            "INSERT INTO logs(plate,matched,confidence,timestamp,image_path,detected_by) VALUES(?,?,?,?,?,?)",
            (plate, 1 if matched else 0, confidence, timestamp, image_path, detected_by),
        )


def send_alert_email(plate_text, matched_info, image_path, timestamp):
    if not EMAIL_ENABLED:
        return
    with get_db() as db:
        authorities = db.execute("SELECT * FROM authorities").fetchall()
    if not authorities:
        return

    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.image import MIMEImage

    for auth in authorities:
        try:
            msg = MIMEMultipart('related')
            msg['Subject'] = f"ANPR ALERT - Watchlist Vehicle Detected: {plate_text}"
            msg['From'] = EMAIL_SENDER
            msg['To'] = auth['email']
            html = f"""
            <html><body style="font-family:Arial,sans-serif;background:#f5f5f5;padding:20px">
            <div style="max-width:600px;margin:0 auto;background:#fff;border-radius:10px;overflow:hidden">
                <div style="background:#cc0000;padding:25px;text-align:center">
                    <h1 style="color:#fff;margin:0">WATCHLIST ALERT</h1>
                    <p style="color:#ffaaaa;margin:5px 0 0">ANPR Surveillance System</p>
                </div>
                <div style="padding:25px">
                    <table style="width:100%;border-collapse:collapse">
                        <tr style="background:#fff0f0">
                            <td style="padding:12px;border:1px solid #ddd;font-weight:bold">Plate Number</td>
                            <td style="padding:12px;border:1px solid #ddd;font-size:1.3rem;font-weight:bold;color:#cc0000;letter-spacing:3px">{plate_text}</td>
                        </tr>
                        <tr>
                            <td style="padding:12px;border:1px solid #ddd;font-weight:bold">Owner</td>
                            <td style="padding:12px;border:1px solid #ddd">{matched_info.get('owner', 'Unknown')}</td>
                        </tr>
                        <tr style="background:#fff0f0">
                            <td style="padding:12px;border:1px solid #ddd;font-weight:bold">Reason</td>
                            <td style="padding:12px;border:1px solid #ddd;color:#cc0000">{matched_info.get('reason', 'Unknown')}</td>
                        </tr>
                        <tr>
                            <td style="padding:12px;border:1px solid #ddd;font-weight:bold">Detection Time</td>
                            <td style="padding:12px;border:1px solid #ddd">{timestamp}</td>
                        </tr>
                        <tr style="background:#fff0f0">
                            <td style="padding:12px;border:1px solid #ddd;font-weight:bold">Alert Sent To</td>
                            <td style="padding:12px;border:1px solid #ddd">{auth['name']} ({auth['role']})</td>
                        </tr>
                    </table>
                    <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:15px;margin-top:20px">
                        <strong>Immediate Action Required</strong><br>
                        A vehicle on the watchlist has been detected. Please take appropriate action.
                    </div>
                    <p style="color:#999;font-size:0.8rem;margin-top:20px;border-top:1px solid #eee;padding-top:15px">
                        Automated alert from ANPR System. Generated: {timestamp}
                    </p>
                </div>
            </div>
            </body></html>
            """
            msg.attach(MIMEText(html, 'html'))
            if image_path and os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename='detection.jpg')
                    msg.attach(img)
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
            print(f'Alert email sent to {auth["email"]}')
        except Exception as e:
            print(f'Email failed to {auth["email"]}: {e}')


def trigger_alert(plate_text, matched_info, image_path, timestamp):
    threading.Thread(target=send_alert_email, args=(plate_text, matched_info, image_path, timestamp), daemon=True).start()
