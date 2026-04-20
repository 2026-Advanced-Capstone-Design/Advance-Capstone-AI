from flask import Flask
from flask_cors import CORS
from routes.analyze import analyze_bp
from routes.status import status_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(analyze_bp)
app.register_blueprint(status_bp)


@app.get("/health")
def health():
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
