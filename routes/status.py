from flask import Blueprint, jsonify
from models.task import get_task

status_bp = Blueprint("status", __name__)


@status_bp.route("/status/<task_id>", methods=["GET"])
def get_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        return jsonify({"error": "태스크를 찾을 수 없습니다."}), 404
    return jsonify(task.to_dict()), 200
