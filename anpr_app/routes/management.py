import csv
import datetime
import os

from flask import Blueprint, jsonify, redirect, render_template, request, session, send_file

from .auth import admin_required, login_required
from ..database import get_db, hash_password
from ..services.recognition import send_alert_email

management_bp = Blueprint('management', __name__)


@management_bp.route('/dashboard')
@login_required
def index():
    with get_db() as db:
        total_logs = db.execute("SELECT COUNT(*) FROM logs").fetchone()[0]
        total_watchlist = db.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        total_alerts = db.execute("SELECT COUNT(*) FROM logs WHERE matched=1").fetchone()[0]
        recent_logs = db.execute("SELECT * FROM logs ORDER BY id DESC LIMIT 5").fetchall()
    return render_template('index.html', total_logs=total_logs, total_watchlist=total_watchlist, total_alerts=total_alerts, recent_logs=recent_logs, user=session.get('user'), role=session.get('role'))


@management_bp.route('/watchlist')
@login_required
def watchlist():
    with get_db() as db:
        entries = db.execute("SELECT * FROM watchlist ORDER BY id DESC").fetchall()
    return render_template('watchlist.html', entries=entries, user=session.get('user'), role=session.get('role'))


@management_bp.route('/watchlist/add', methods=['POST'])
@login_required
def add_vehicle():
    plate = request.form['plate'].upper().strip()
    owner = request.form['owner'].strip()
    reason = request.form['reason'].strip()
    added = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with get_db() as db:
        db.execute("INSERT OR IGNORE INTO watchlist(plate,owner,reason,added) VALUES(?,?,?,?)", (plate, owner, reason, added))
    return redirect('/watchlist')


@management_bp.route('/watchlist/delete/<int:vid>')
@login_required
def delete_vehicle(vid):
    with get_db() as db:
        db.execute("DELETE FROM watchlist WHERE id=?", (vid,))
    return redirect('/watchlist')


@management_bp.route('/watchlist/update/<int:vid>', methods=['POST'])
@login_required
def update_vehicle(vid):
    owner = request.form['owner'].strip()
    reason = request.form['reason'].strip()
    plate = request.form['plate'].upper().strip()
    with get_db() as db:
        db.execute("UPDATE watchlist SET owner=?,reason=?,plate=? WHERE id=?", (owner, reason, plate, vid))
    return redirect('/watchlist')


@management_bp.route('/logs')
@login_required
def logs():
    with get_db() as db:
        all_logs = db.execute("SELECT * FROM logs ORDER BY id DESC").fetchall()
    return render_template('logs.html', logs=all_logs, user=session.get('user'), role=session.get('role'))


@management_bp.route('/logs/clear')
@admin_required
def clear_logs():
    with get_db() as db:
        db.execute("DELETE FROM logs")
    return redirect('/logs')


@management_bp.route('/logs/export')
@login_required
def export_logs():
    with get_db() as db:
        all_logs = db.execute("SELECT * FROM logs ORDER BY id DESC").fetchall()
    csv_path = 'logs_export.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Plate', 'Matched', 'Confidence', 'Timestamp', 'Image', 'Detected By'])
        for log in all_logs:
            writer.writerow([log['id'], log['plate'], log['matched'], log['confidence'], log['timestamp'], log['image_path'], log['detected_by'] if 'detected_by' in log.keys() else ''])
    return send_file(csv_path, as_attachment=True)


@management_bp.route('/authorities')
@login_required
def authorities():
    with get_db() as db:
        auths = db.execute("SELECT * FROM authorities ORDER BY id DESC").fetchall()
    return render_template('authorities.html', authorities=auths, user=session.get('user'), role=session.get('role'))


@management_bp.route('/authorities/add', methods=['POST'])
@admin_required
def add_authority():
    name = request.form['name'].strip()
    email = request.form['email'].strip()
    role = request.form['role'].strip()
    phone = request.form['phone'].strip()
    added = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with get_db() as db:
        try:
            db.execute("INSERT INTO authorities(name,email,role,phone,added) VALUES(?,?,?,?,?)", (name, email, role, phone, added))
        except Exception:
            pass
    return redirect('/authorities')


@management_bp.route('/authorities/delete/<int:aid>')
@admin_required
def delete_authority(aid):
    with get_db() as db:
        db.execute("DELETE FROM authorities WHERE id=?", (aid,))
    return redirect('/authorities')


@management_bp.route('/authorities/test_email')
@admin_required
def test_email():
    with get_db() as db:
        auths = db.execute("SELECT * FROM authorities").fetchall()
    if not auths:
        return jsonify({'error': 'No authorities added yet'}), 400
    send_alert_email('TEST-PLATE', {'owner': 'Test Owner', 'reason': 'Test Alert'}, None, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return jsonify({'status': 'Test email sent!'})


@management_bp.route('/about')
def about():
    return render_template('about.html', user=session.get('user'), role=session.get('role'))


@management_bp.route('/users')
@admin_required
def users():
    with get_db() as db:
        all_users = db.execute("SELECT id,username,role,created FROM users ORDER BY id").fetchall()
    return render_template('users.html', users=all_users, user=session.get('user'), role=session.get('role'))


@management_bp.route('/users/add', methods=['POST'])
@admin_required
def add_user():
    username = request.form['username'].strip()
    password = request.form['password'].strip()
    role = request.form['role'].strip()
    created = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if len(password) < 6:
        return "Password must be at least 6 characters", 400
    with get_db() as db:
        try:
            db.execute("INSERT INTO users(username,password,role,created) VALUES(?,?,?,?)", (username, hash_password(password), role, created))
        except Exception:
            pass
    return redirect('/users')


@management_bp.route('/users/delete/<int:uid>')
@admin_required
def delete_user(uid):
    with get_db() as db:
        db.execute("DELETE FROM users WHERE id=?", (uid,))
    return redirect('/users')
