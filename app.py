import re
from flask import (Flask, render_template, request, redirect,
                   jsonify, Response, session, url_for, send_file)
import sqlite3, cv2, numpy as np, datetime, os, csv, pickle, hashlib
import smtplib, threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from cnn_scratch import SimpleCNN
from yolo_detector import (detect_plate_region, detect_all_plates,
                            segment_and_recognize, draw_plate_box)
from label_map import load_label_map
import base64

app = Flask(__name__)
app.secret_key = 'anpr_secret_2024'
DB  = 'database.db'

# ── EMAIL CONFIG ─────────────────────────────────────
EMAIL_SENDER   = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"
EMAIL_ENABLED  = False  # set True after configuring above

# ── LOAD LABEL MAP ───────────────────────────────────
CHAR_TO_IDX, IDX_TO_CHAR, NUM_CLASSES, ALL_CLASSES = load_label_map('label_map.pkl')

# ── LOAD CNN MODEL ───────────────────────────────────
model = SimpleCNN(num_classes=NUM_CLASSES)
if os.path.exists('model.pkl'):
    model.load('model.pkl')
else:
    print("model.pkl not found!")

# ── DATABASE ─────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS watchlist (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            plate   TEXT UNIQUE NOT NULL,
            owner   TEXT,
            reason  TEXT,
            added   TEXT
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS logs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            plate       TEXT,
            matched     INTEGER,
            confidence  REAL,
            timestamp   TEXT,
            image_path  TEXT,
            detected_by TEXT
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role     TEXT DEFAULT 'user',
            created  TEXT
        )''')
        db.execute('''CREATE TABLE IF NOT EXISTS authorities (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,
            name  TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role  TEXT,
            phone TEXT,
            added TEXT
        )''')
        admin_pass = hashlib.sha256('admin123'.encode()).hexdigest()
        db.execute('''INSERT OR IGNORE INTO users
                      (username, password, role, created)
                      VALUES (?, ?, ?, ?)''',
                   ('admin', admin_pass, 'admin',
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print("Database ready")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ── AUTH HELPERS ─────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('root'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('root'))
        if session.get('role') != 'admin':
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated

# ── EMAIL ─────────────────────────────────────────────
def send_alert_email(plate_text, matched_info, image_path, timestamp):
    if not EMAIL_ENABLED:
        return
    with get_db() as db:
        authorities = db.execute("SELECT * FROM authorities").fetchall()
    if not authorities:
        return
    for auth in authorities:
        try:
            msg = MIMEMultipart('related')
            msg['Subject'] = f"ANPR ALERT - Watchlist Vehicle Detected: {plate_text}"
            msg['From']    = EMAIL_SENDER
            msg['To']      = auth['email']
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
                            <td style="padding:12px;border:1px solid #ddd">{matched_info.get('owner','Unknown')}</td>
                        </tr>
                        <tr style="background:#fff0f0">
                            <td style="padding:12px;border:1px solid #ddd;font-weight:bold">Reason</td>
                            <td style="padding:12px;border:1px solid #ddd;color:#cc0000">{matched_info.get('reason','Unknown')}</td>
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
                    img.add_header('Content-Disposition', 'attachment',
                                   filename='detection.jpg')
                    msg.attach(img)
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
            print(f"Alert email sent to {auth['email']}")
        except Exception as e:
            print(f"Email failed to {auth['email']}: {e}")

# ── RECOGNITION ──────────────────────────────────────
def recognize_all_plates(image):
    """
    Detect and recognize ALL plates in an image.
    Returns (results_list, annotated_image).
    results_list may be empty if nothing found.
    """
    all_plates = detect_all_plates(image)
    annotated  = image.copy()

    if not all_plates:
        return [], annotated

    results = []
    best_steps = {}

    for plate_img, coords, det_conf in all_plates:
        if plate_img is None or plate_img.size == 0:
            continue

        plate_text, confidence, steps, char_boxes = \
            segment_and_recognize(plate_img, model, IDX_TO_CHAR)

        if (not plate_text or
            plate_text in ["NOT DETECTED", "NO CHARS", ""] or
            not is_valid_plate_format(plate_text)):
            continue

        matched   = check_watchlist(plate_text)
        annotated = draw_plate_box(annotated, coords,
                                   plate_text, matched is not None,
                                   confidence)

        # Keep steps from the highest confidence detection
        if not best_steps or confidence > results[-1]['confidence'] if results else True:
            best_steps = steps

        results.append({
            'plate'     : plate_text,
            'confidence': round(confidence, 1),
            'det_conf'  : round(det_conf * 100, 1),
            'matched'   : matched is not None,
            'info'      : matched,
            'coords'    : coords,
            'steps'     : steps,
        })

    return results, annotated

def check_watchlist(plate_text):
    with get_db() as db:
        row = db.execute(
            "SELECT * FROM watchlist WHERE plate=?",
            (plate_text,)).fetchone()
    return dict(row) if row else None

def is_valid_plate_format(plate_text):
    """
    Validates Nepali embossed plate format:
    First row - 1 letter + space + 2 letters (read as 3 letters, no space)
    Second row - 4 digits
    Combined string must be exactly 3 letters followed by 4 digits.
    """
    if not plate_text or len(plate_text) != 7:
        return False
    pattern = r'^[A-Z]{3}[0-9]{4}$'
    return bool(re.match(pattern, plate_text))

def save_image(image, filename):
    path = os.path.join('static', 'uploads', filename)
    cv2.imwrite(path, image)
    return path

# ── LANDING ───────────────────────────────────────────
@app.route('/')
def root():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('landing.html')

@app.route('/landing')
def landing():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('landing.html')

# ── AUTH ROUTES ──────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        hashed   = hash_password(password)
        with get_db() as db:
            user = db.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, hashed)).fetchone()
        if user:
            session['user'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('index'))
        else:
            error = 'Invalid username or password'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('root'))

# ── DASHBOARD ─────────────────────────────────────────
@app.route('/dashboard')
@login_required
def index():
    with get_db() as db:
        total_logs      = db.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        total_watchlist = db.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        total_alerts    = db.execute("SELECT COUNT(*) FROM logs WHERE matched=1").fetchone()[0]
        recent_logs     = db.execute(
            "SELECT * FROM logs ORDER BY id DESC LIMIT 5").fetchall()
    return render_template('index.html',
                           total_logs=total_logs,
                           total_watchlist=total_watchlist,
                           total_alerts=total_alerts,
                           recent_logs=recent_logs,
                           user=session.get('user'),
                           role=session.get('role'))

# ── IMAGE DETECTION ───────────────────────────────────
@app.route('/detect', methods=['POST'])
@login_required
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

    # Recognize ALL plates in image
    results, annotated = recognize_all_plates(image)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    filename  = f"det_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    img_path  = save_image(annotated, filename)

    if not results:
        # Nothing detected
        with get_db() as db:
            db.execute(
                "INSERT INTO logs(plate,matched,confidence,timestamp,image_path,detected_by) VALUES(?,?,?,?,?,?)",
                ("NOT DETECTED", 0, 0.0, timestamp, img_path,
                 session.get('user', 'system')))
        _, buffer  = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'plate'     : 'NOT DETECTED',
            'confidence': 0.0,
            'det_conf'  : 0.0,
            'matched'   : False,
            'info'      : None,
            'timestamp' : timestamp,
            'image'     : img_base64,
            'steps'     : {},
            'all_plates': []
        })

    # Log every detected plate
    all_plates_info = []
    best_steps_b64  = {}

    for r in results:
        matched = r['info']
        with get_db() as db:
            db.execute(
                "INSERT INTO logs(plate,matched,confidence,timestamp,image_path,detected_by) VALUES(?,?,?,?,?,?)",
                (r['plate'], 1 if r['matched'] else 0, r['confidence'],
                 timestamp, img_path, session.get('user', 'system')))

        # Send email alert if matched
        if r['matched'] and matched:
            threading.Thread(
                target=send_alert_email,
                args=(r['plate'], matched, img_path, timestamp),
                daemon=True
            ).start()

        all_plates_info.append({
            'plate'     : r['plate'],
            'confidence': r['confidence'],
            'det_conf'  : r['det_conf'],
            'matched'   : r['matched'],
            'info'      : r['info'],
        })

    # Use steps from first (highest confidence) result
    steps = results[0].get('steps', {})
    for key in ['grayscale', 'clahe', 'binary', 'annotated']:
        if key in steps:
            step_img = steps[key]
            if len(step_img.shape) == 2:
                step_img = cv2.cvtColor(step_img, cv2.COLOR_GRAY2BGR)
            _, buf = cv2.imencode('.jpg', step_img)
            best_steps_b64[key] = base64.b64encode(buf).decode('utf-8')

    _, buffer  = cv2.imencode('.jpg', annotated)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # Primary result = highest confidence plate
    primary = results[0]

    return jsonify({
        'plate'     : primary['plate'],
        'confidence': primary['confidence'],
        'det_conf'  : primary['det_conf'],
        'matched'   : primary['matched'],
        'info'      : primary['info'],
        'timestamp' : timestamp,
        'image'     : img_base64,
        'steps'     : best_steps_b64,
        'all_plates': all_plates_info
    })

# ── VIDEO DETECTION ───────────────────────────────────
@app.route('/detect_video', methods=['GET', 'POST'])
@login_required
def detect_video():
    if request.method == 'GET':
        return render_template('video.html',
                               user=session.get('user'),
                               role=session.get('role'))

    if 'video' not in request.files:
        return jsonify({'error': 'No video'}), 400

    file      = request.files['video']
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    vid_path  = f'uploads/video_{timestamp}.mp4'
    out_path  = f'uploads/output_{timestamp}.mp4'
    os.makedirs('uploads', exist_ok=True)
    file.save(vid_path)

    cap    = cv2.VideoCapture(vid_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    interval       = max(1, int(fps * 0.5))
    plate_best     = {}
    frame_count    = 0
    last_results   = []
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

                plate_text, confidence, steps, char_boxes = \
                    segment_and_recognize(plate_img, model, IDX_TO_CHAR)

                if (not plate_text or
                    plate_text in ["NOT DETECTED", "NO CHARS", ""] or
                    not is_valid_plate_format(plate_text) or
                    confidence < 70.0):
                    continue

                matched = check_watchlist(plate_text)

                # Keep only the best result per unique plate
                if (plate_text not in plate_best or
                        confidence > plate_best[plate_text]['confidence']):
                    plate_best[plate_text] = {
                        'confidence': confidence,
                        'det_conf'  : det_conf,
                        'frame'     : frame.copy(),
                        'coords'    : coords,
                        'matched'   : matched is not None,
                        'info'      : matched,
                    }

                new_results.append({
                    'plate'  : plate_text,
                    'coords' : coords,
                    'matched': matched is not None,
                    'conf'   : confidence,
                })

            if new_results:
                last_results   = new_results
                last_frame_cnt = DISPLAY_FRAMES

        # Draw tracked plates continuously on every frame
        annotated = frame.copy()
        if last_frame_cnt > 0:
            for r in last_results:
                annotated = draw_plate_box(annotated, r['coords'],
                                           r['plate'], r['matched'],
                                           r['conf'])
            last_frame_cnt -= 1

        writer.write(annotated)

    cap.release()
    writer.release()

    # Save only the single best frame per plate
    results_list = []
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for plate_text, best in plate_best.items():
        best_frame = draw_plate_box(
            best['frame'].copy(), best['coords'],
            plate_text, best['matched'], best['confidence'])

        fname    = f"vid_{timestamp}_{plate_text}.jpg"
        img_path = save_image(best_frame, fname)

        with get_db() as db:
            db.execute(
                "INSERT INTO logs(plate,matched,confidence,timestamp,image_path,detected_by) VALUES(?,?,?,?,?,?)",
                (plate_text, 1 if best['matched'] else 0,
                 best['confidence'], ts, img_path, 'video_upload'))

        _, buf  = cv2.imencode('.jpg', best_frame)
        img_b64 = base64.b64encode(buf).decode('utf-8')

        results_list.append({
            'plate'     : plate_text,
            'confidence': round(best['confidence'], 1),
            'matched'   : best['matched'],
            'timestamp' : ts,
            'image'     : img_b64
        })

        if best['matched'] and best['info']:
            threading.Thread(
                target=send_alert_email,
                args=(plate_text, best['info'], img_path, ts),
                daemon=True
            ).start()

    results_list.sort(key=lambda r: r['confidence'], reverse=True)

    output_b64 = ''
    if os.path.exists(out_path):
        with open(out_path, 'rb') as f:
            output_b64 = base64.b64encode(f.read()).decode('utf-8')

    try:
        os.remove(vid_path)
        os.remove(out_path)
    except:
        pass

    return jsonify({
        'total_frames': frame_count,
        'plates_found': len(results_list),
        'results'     : results_list,
        'output_video': output_b64
    })

# ── WEBCAM ────────────────────────────────────────────
camera           = None
latest_detection = {}
camera_active    = False

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def generate_frames():
    global latest_detection, camera_active, camera
    camera_active = True
    cam = get_camera()

    frame_count  = 0
    DETECT_EVERY = 15
    MIN_CONF     = 0.5

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
                        plate_text, confidence, steps, char_boxes = \
                            segment_and_recognize(plate_img, model, IDX_TO_CHAR)

                        if (plate_text and
                            plate_text not in ["NOT DETECTED","NO CHARS",""] and
                            is_valid_plate_format(plate_text) and
                            confidence >= 70.0):

                            matched   = check_watchlist(plate_text)
                            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            filename  = f"cam_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                            annotated = draw_plate_box(frame.copy(), coords,
                                                       plate_text,
                                                       matched is not None,
                                                       confidence)
                            img_path  = save_image(annotated, filename)

                            with get_db() as db:
                                db.execute(
                                    "INSERT INTO logs(plate,matched,confidence,timestamp,image_path,detected_by) VALUES(?,?,?,?,?,?)",
                                    (plate_text, 1 if matched else 0,
                                     confidence, timestamp, img_path, 'webcam'))

                            latest_detection = {
                                'plate'     : plate_text,
                                'confidence': confidence,
                                'matched'   : matched is not None,
                                'info'      : matched,
                                'timestamp' : timestamp
                            }

                            if matched:
                                threading.Thread(
                                    target=send_alert_email,
                                    args=(plate_text, matched, img_path, timestamp),
                                    daemon=True
                                ).start()

                        elif coords is not None:
                            annotated = draw_plate_box(frame.copy(), coords,
                                                       plate_text if plate_text else 'Scanning...',
                                                       False, confidence)
                except Exception:
                    pass

            if not camera_active:
                break

            _, buffer   = cv2.imencode('.jpg', annotated)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

    except GeneratorExit:
        pass
    finally:
        # Always release camera when generator exits for any reason
        camera_active = False
        if camera is not None:
            camera.release()
            camera = None

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
@login_required
def webcam():
    return render_template('webcam.html',
                           user=session.get('user'),
                           role=session.get('role'))

@app.route('/latest_detection')
@login_required
def get_latest_detection():
    return jsonify(latest_detection)

@app.route('/release_camera', methods=['POST'])
@login_required
def release_camera():
    global camera, camera_active
    camera_active = False
    import time
    time.sleep(0.1)  # give generator time to notice flag
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'released'})

# ── WATCHLIST ─────────────────────────────────────────
@app.route('/watchlist')
@login_required
def watchlist():
    with get_db() as db:
        entries = db.execute(
            "SELECT * FROM watchlist ORDER BY id DESC").fetchall()
    return render_template('watchlist.html', entries=entries,
                           user=session.get('user'),
                           role=session.get('role'))

@app.route('/watchlist/add', methods=['POST'])
@login_required
def add_vehicle():
    plate  = request.form['plate'].upper().strip()
    owner  = request.form['owner'].strip()
    reason = request.form['reason'].strip()
    added  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with get_db() as db:
        db.execute(
            "INSERT OR IGNORE INTO watchlist(plate,owner,reason,added) VALUES(?,?,?,?)",
            (plate, owner, reason, added))
    return redirect('/watchlist')

@app.route('/watchlist/delete/<int:vid>')
@login_required
def delete_vehicle(vid):
    with get_db() as db:
        db.execute("DELETE FROM watchlist WHERE id=?", (vid,))
    return redirect('/watchlist')

@app.route('/watchlist/update/<int:vid>', methods=['POST'])
@login_required
def update_vehicle(vid):
    owner  = request.form['owner'].strip()
    reason = request.form['reason'].strip()
    plate  = request.form['plate'].upper().strip()
    with get_db() as db:
        db.execute(
            "UPDATE watchlist SET owner=?,reason=?,plate=? WHERE id=?",
            (owner, reason, plate, vid))
    return redirect('/watchlist')

# ── LOGS ──────────────────────────────────────────────
@app.route('/logs')
@login_required
def logs():
    with get_db() as db:
        all_logs = db.execute(
            "SELECT * FROM logs ORDER BY id DESC").fetchall()
    return render_template('logs.html', logs=all_logs,
                           user=session.get('user'),
                           role=session.get('role'))

@app.route('/logs/clear')
@admin_required
def clear_logs():
    with get_db() as db:
        db.execute("DELETE FROM logs")
    return redirect('/logs')

@app.route('/logs/export')
@login_required
def export_logs():
    with get_db() as db:
        all_logs = db.execute(
            "SELECT * FROM logs ORDER BY id DESC").fetchall()
    csv_path = 'logs_export.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Plate', 'Matched', 'Confidence',
                         'Timestamp', 'Image', 'Detected By'])
        for log in all_logs:
            writer.writerow([
                log['id'], log['plate'], log['matched'],
                log['confidence'], log['timestamp'],
                log['image_path'],
                log['detected_by'] if 'detected_by' in log.keys() else ''
            ])
    return send_file(csv_path, as_attachment=True)

# ── AUTHORITIES ───────────────────────────────────────
@app.route('/authorities')
@login_required
def authorities():
    with get_db() as db:
        auths = db.execute(
            "SELECT * FROM authorities ORDER BY id DESC").fetchall()
    return render_template('authorities.html', authorities=auths,
                           user=session.get('user'),
                           role=session.get('role'))

@app.route('/authorities/add', methods=['POST'])
@admin_required
def add_authority():
    name  = request.form['name'].strip()
    email = request.form['email'].strip()
    role  = request.form['role'].strip()
    phone = request.form['phone'].strip()
    added = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with get_db() as db:
        try:
            db.execute(
                "INSERT INTO authorities(name,email,role,phone,added) VALUES(?,?,?,?,?)",
                (name, email, role, phone, added))
        except:
            pass
    return redirect('/authorities')

@app.route('/authorities/delete/<int:aid>')
@admin_required
def delete_authority(aid):
    with get_db() as db:
        db.execute("DELETE FROM authorities WHERE id=?", (aid,))
    return redirect('/authorities')

@app.route('/authorities/test_email')
@admin_required
def test_email():
    with get_db() as db:
        auths = db.execute("SELECT * FROM authorities").fetchall()
    if not auths:
        return jsonify({'error': 'No authorities added yet'}), 400
    threading.Thread(
        target=send_alert_email,
        args=('TEST-PLATE', {'owner': 'Test Owner', 'reason': 'Test Alert'},
              None, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        daemon=True
    ).start()
    return jsonify({'status': 'Test email sent!'})

# ── ABOUT ─────────────────────────────────────────────
@app.route('/about')
def about():
    return render_template('about.html',
                           user=session.get('user'),
                           role=session.get('role'))

# ── USER MANAGEMENT ───────────────────────────────────
@app.route('/users')
@admin_required
def users():
    with get_db() as db:
        all_users = db.execute(
            "SELECT id,username,role,created FROM users ORDER BY id").fetchall()
    return render_template('users.html', users=all_users,
                           user=session.get('user'),
                           role=session.get('role'))

@app.route('/users/add', methods=['POST'])
@admin_required
def add_user():
    username = request.form['username'].strip()
    password = request.form['password'].strip()
    role     = request.form['role'].strip()
    created  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if len(password) < 6:
        return "Password must be at least 6 characters", 400
    with get_db() as db:
        try:
            db.execute(
                "INSERT INTO users(username,password,role,created) VALUES(?,?,?,?)",
                (username, hash_password(password), role, created))
        except:
            pass
    return redirect('/users')

@app.route('/users/delete/<int:uid>')
@admin_required
def delete_user(uid):
    with get_db() as db:
        db.execute("DELETE FROM users WHERE id=?", (uid,))
    return redirect('/users')

# ── MAIN ──────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    os.makedirs('static/uploads', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    print("\nANPR System running at http://127.0.0.1:5000\n")
    app.run(debug=True, use_reloader=False)