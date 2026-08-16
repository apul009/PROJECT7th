import base64
import os
import time
import datetime
import cv2
import numpy as np
from flask import Blueprint, Response, jsonify, render_template, request, session

from yolo_detector import detect_all_plates, detect_plate_region, draw_plate_box, segment_and_recognize

from ..database import get_db
from ..services.recognition import (
    IDX_TO_CHAR,
    check_watchlist,
    encode_image,
    is_valid_plate_format,
    log_detection,
    model,
    recognize_all_plates,
    save_image,
    trigger_alert,
)

detection_bp = Blueprint('detection', __name__)

camera = None
latest_detection = {}
camera_active = False
active_tracks = []
session_seen_plates = set()


def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera


@detection_bp.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image'}), 400

    results, annotated = recognize_all_plates(image)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    filename = f"det_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path = save_image(annotated, filename)

    if not results:
        log_detection('NOT DETECTED', False, 0.0, timestamp, img_path, session.get('user', 'system'))
        return jsonify({
            'plate': 'NOT DETECTED',
            'confidence': 0.0,
            'det_conf': 0.0,
            'matched': False,
            'info': None,
            'timestamp': timestamp,
            'image': encode_image(annotated),
            'steps': {},
            'all_plates': []
        })

    all_plates_info = []
    best_steps_b64 = {}

    for r in results:
        log_detection(r['plate'], r['matched'], r['confidence'], timestamp, img_path, session.get('user', 'system'))
        if r['matched'] and r['info']:
            trigger_alert(r['plate'], r['info'], img_path, timestamp)
        all_plates_info.append({
            'plate': r['plate'],
            'confidence': r['confidence'],
            'det_conf': r['det_conf'],
            'matched': r['matched'],
            'info': r['info'],
        })

    steps = results[0].get('steps', {})
    for key in ['grayscale', 'clahe', 'binary', 'annotated']:
        if key in steps:
            step_img = steps[key]
            if len(step_img.shape) == 2:
                step_img = cv2.cvtColor(step_img, cv2.COLOR_GRAY2BGR)
            best_steps_b64[key] = encode_image(step_img)

    primary = results[0]
    return jsonify({
        'plate': primary['plate'],
        'confidence': primary['confidence'],
        'det_conf': primary['det_conf'],
        'matched': primary['matched'],
        'info': primary['info'],
        'timestamp': timestamp,
        'image': encode_image(annotated),
        'steps': best_steps_b64,
        'all_plates': all_plates_info,
    })


@detection_bp.route('/detect_video', methods=['GET', 'POST'])
def detect_video():
    if request.method == 'GET':
        return render_template('video.html', user=session.get('user'), role=session.get('role'))

    if 'video' not in request.files:
        return jsonify({'error': 'No video'}), 400

    file = request.files['video']
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    vid_path = f'uploads/video_{timestamp}.mp4'
    out_path = f'uploads/output_{timestamp}.mp4'
    os.makedirs('uploads', exist_ok=True)
    file.save(vid_path)

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    interval = max(1, int(fps * 0.5))
    plate_best = {}
    frame_count = 0
    last_results = []
    last_frame_cnt = 0
    DISPLAY_FRAMES = int(fps * 2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % interval == 0:
            all_plates = detect_all_plates(frame)
            new_results = []
            for plate_img, coords, det_conf in all_plates:
                if det_conf < 0.40:
                    continue
                plate_text, confidence, steps, char_boxes = segment_and_recognize(plate_img, model, IDX_TO_CHAR)
                plate_text = plate_text.upper() if plate_text else plate_text
                if not is_valid_plate_format(plate_text) or confidence < 70.0:
                    continue
                matched = check_watchlist(plate_text)
                if plate_text not in plate_best or confidence > plate_best[plate_text]['confidence']:
                    plate_best[plate_text] = {
                        'confidence': confidence,
                        'det_conf': det_conf,
                        'frame': frame.copy(),
                        'coords': coords,
                        'matched': matched is not None,
                        'info': matched,
                    }
                new_results.append({'plate': plate_text, 'coords': coords, 'matched': matched is not None, 'conf': confidence})
            if new_results:
                last_results = new_results
                last_frame_cnt = DISPLAY_FRAMES

        annotated = frame.copy()
        if last_frame_cnt > 0:
            for r in last_results:
                annotated = draw_plate_box(annotated, r['coords'], r['plate'], r['matched'], r['conf'])
            last_frame_cnt -= 1
        writer.write(annotated)

    cap.release()
    writer.release()

    results_list = []
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for plate_text, best in plate_best.items():
        best_frame = draw_plate_box(best['frame'].copy(), best['coords'], plate_text, best['matched'], best['confidence'])
        fname = f"vid_{timestamp}_{plate_text}.jpg"
        img_path = save_image(best_frame, fname)
        log_detection(plate_text, best['matched'], best['confidence'], ts, img_path, 'video_upload')
        if best['matched'] and best['info']:
            trigger_alert(plate_text, best['info'], img_path, ts)
        results_list.append({
            'plate': plate_text,
            'confidence': round(best['confidence'], 1),
            'matched': best['matched'],
            'timestamp': ts,
            'image': encode_image(best_frame),
        })

    results_list.sort(key=lambda r: r['confidence'], reverse=True)

    output_b64 = ''
    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            output_b64 = base64.b64encode(f.read()).decode('utf-8')

    try:
        os.remove(vid_path)
        os.remove(out_path)
    except Exception:
        pass

    return jsonify({
        'total_frames': frame_count,
        'plates_found': len(results_list),
        'results': results_list,
        'output_video': output_b64,
    })


@detection_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@detection_bp.route('/webcam')
def webcam():
    return render_template('webcam.html', user=session.get('user'), role=session.get('role'))


@detection_bp.route('/latest_detection')
def get_latest_detection():
    return jsonify(latest_detection)


@detection_bp.route('/release_camera', methods=['POST'])
def release_camera():
    global camera, camera_active, active_tracks, session_seen_plates, latest_detection
    camera_active = False
    active_tracks = []
    session_seen_plates = set()
    latest_detection = {}
    time.sleep(0.1)
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'released'})


def generate_frames():
    global latest_detection, camera_active, camera, active_tracks, session_seen_plates
    camera_active = True
    cam = get_camera()

    # These are per-camera-session state.  They keep valid labels visible between
    # recognition passes while avoiding repeated snapshot/log creation.
    active_tracks = []
    session_seen_plates = set()
    latest_detection = {}

    frame_count = 0
    DETECT_EVERY = 15
    MIN_CONF = 0.5

    try:
        while camera_active:
            if camera is None:
                break
            success, frame = cam.read()
            if not success or not camera_active:
                break
            frame_count += 1
            annotated = frame.copy()

            if frame_count % DETECT_EVERY == 0 and camera_active:
                try:
                    plate_img, coords, det_conf = detect_plate_region(frame)
                    if det_conf >= MIN_CONF:
                        plate_text, confidence, steps, char_boxes = segment_and_recognize(plate_img, model, IDX_TO_CHAR)
                        plate_text = plate_text.upper() if plate_text else plate_text
                        if is_valid_plate_format(plate_text) and confidence >= 70.0:
                            matched = check_watchlist(plate_text)
                            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            active_tracks = [{
                                'plate': plate_text,
                                'coords': coords,
                                'matched': matched is not None,
                                'conf': confidence,
                                # Show the last valid detection during the next recognition interval.
                                'frames_left': DETECT_EVERY,
                            }]

                            # A plate is recorded once per live-camera session, not every sample.
                            if plate_text not in session_seen_plates:
                                filename = f"cam_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                                snapshot = draw_plate_box(frame.copy(), coords, plate_text, matched is not None, confidence)
                                img_path = save_image(snapshot, filename)
                                log_detection(plate_text, matched is not None, confidence, timestamp, img_path, 'webcam')
                                session_seen_plates.add(plate_text)
                            else:
                                img_path = None
                            latest_detection = {
                                'plate': plate_text,
                                'confidence': confidence,
                                'matched': matched is not None,
                                'info': matched,
                                'timestamp': timestamp,
                            }
                            if matched and img_path:
                                trigger_alert(plate_text, matched, img_path, timestamp)
                except Exception:
                    pass

            # Never show unvalidated recognition text.  Valid labels remain live
            # on the scene between inference passes instead of flashing briefly.
            live_tracks = []
            for track in active_tracks:
                if track['frames_left'] <= 0:
                    continue
                annotated = draw_plate_box(annotated, track['coords'], track['plate'], track['matched'], track['conf'])
                track['frames_left'] -= 1
                live_tracks.append(track)
            active_tracks = live_tracks

            if not camera_active:
                break
            _, buffer = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except GeneratorExit:
        pass
    finally:
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None
