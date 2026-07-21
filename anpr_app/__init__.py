from flask import Flask

from .config import BASE_DIR, SECRET_KEY
from .database import init_db
from .routes.auth import auth_bp
from .routes.detection import detection_bp
from .routes.management import management_bp


def create_app():
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    app.secret_key = SECRET_KEY

    app.register_blueprint(auth_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(management_bp)

    with app.app_context():
        init_db()

    return app


app = create_app()
