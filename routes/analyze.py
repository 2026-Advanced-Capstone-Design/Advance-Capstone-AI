import threading
import requests
from flask import Blueprint, request, jsonify
from models.task import create_task, update_task, TaskStatus
from services.preprocessor import preprocess
from services.summarizer import summarize
from services.labeler import label
from services.factcheck import check_facts
from services.analyzer import analyze as run_analysis
from config import SPRING_CALLBACK_URL

analyze_bp = Blueprint("analyze", __name__)


def _notify_spring(payload: dict):
    try:
        requests.post(SPRING_CALLBACK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"[WARN] Spring 콜백 전송 실패: {e}")


def _run_pipeline(task_id: str, article_id: int, text: str, input_type: str):
    try:
        update_task(task_id, TaskStatus.ANALYZING)

        # Step 1: 전처리
        is_html = input_type == "URL"
        preprocessed = preprocess(text, is_html=is_html)
        cleaned_text = preprocessed["cleaned"]

        # Step 2: 요약 / 키워드 생성
        summary = summarize(cleaned_text)

        # Step 3: Fact Check (Google 우선 → GPT 종합)
        factcheck_result = check_facts(summary.get("key_facts", []))
        fact_ratio_source = factcheck_result.get("fact_ratio")

        # Step 4: 섹션별 편향 레이블 (Few-shot + CoT 3단계)
        label_result = label(cleaned_text)
        bias_label = label_result.get("overall_label", "uncertain")

        # Step 5: Generated Knowledge + CoT 2단계 편향 분석
        analysis = run_analysis(
            text=cleaned_text,
            topic=summary.get("topic", ""),
            keywords=summary.get("keywords", []),
            bias_label=bias_label,
            sentences=preprocessed["sentences"],
            sections=label_result.get("sections", []),
        )

        # fact_ratio: Google 결과 우선, 없으면 CoT 기반
        fact_ratio = fact_ratio_source if fact_ratio_source is not None \
            else analysis.get("fact_ratio", 0.5)

        bias_score = analysis.get("bias_score", 0.5)
        total_score = round((1.0 - bias_score) * 100)

        result = {
            "article_id": article_id,
            "compressed_text": summary.get("compressed_text", ""),
            "keywords": summary.get("keywords", []),
            "topic": summary.get("topic", ""),
            "bias_label": bias_label,
            "bias_confidence": label_result.get("overall_confidence", 0.0),
            "bias_reason": label_result.get("overall_reason", ""),
            "sections": label_result.get("sections", []),
            "highlighted_sentences": analysis.get("highlighted_sentences", []),
            "emotion_neutrality": analysis.get("emotion_neutrality", 0.5),
            "fact_ratio": fact_ratio,
            "fact_ratio_source": fact_ratio_source,
            "bias_score": bias_score,
            "total_score": total_score,
            "cot_emotion_reason": analysis.get("cot_emotion_reason", ""),
            "fact_check_reason": factcheck_result.get("fact_check_reason", ""),
        }

        update_task(task_id, TaskStatus.DONE, result=result)
        _notify_spring({"status": "DONE", **result})

    except Exception as e:
        update_task(task_id, TaskStatus.FAILED, error=str(e))
        _notify_spring({"article_id": article_id, "status": "FAILED", "error": str(e)})


@analyze_bp.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)

    article_id = data.get("article_id")
    text = data.get("text", "")
    input_type = data.get("input_type", "TEXT")

    if not article_id:
        return jsonify({"error": "article_id is required"}), 400
    if not text or not text.strip():
        return jsonify({"error": "text is required"}), 400

    task = create_task(article_id)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(task.task_id, article_id, text, input_type),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "task_id": task.task_id,
        "article_id": article_id,
        "status": task.status.value,
        "message": "분석이 시작되었습니다.",
    }), 202
