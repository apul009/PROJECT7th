from functools import wraps

from flask import Blueprint, redirect, render_template, request, session, url_for

from ..database import get_db, hash_password

auth_bp = Blueprint('auth', __name__)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('auth.root'))
        return f(*args, **kwargs)

    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('auth.root'))
        if session.get('role') != 'admin':
            return redirect(url_for('management.index'))
        return f(*args, **kwargs)

    return decorated


@auth_bp.route('/')
def root():
    if 'user' in session:
        return redirect(url_for('management.index'))
    return render_template('landing.html')


@auth_bp.route('/landing')
def landing():
    if 'user' in session:
        return redirect(url_for('management.index'))
    return render_template('landing.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        hashed = hash_password(password)
        with get_db() as db:
            user = db.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, hashed),
            ).fetchone()
        if user:
            session['user'] = user['username']
            session['role'] = user['role']
            return redirect(url_for('management.index'))
        error = 'Invalid username or password'
    return render_template('login.html', error=error)


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('auth.root'))
