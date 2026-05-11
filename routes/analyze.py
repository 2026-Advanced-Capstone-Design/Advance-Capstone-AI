import threading
import requests
from flask import Blueprint, request, jsonify
from models.task import create_task, update_task, TaskStatus
from services.preprocessor import preprocess
from services.summarizer import summarize
from services.labeler import label
from services.embedder import embed_and_store
from services.analyzer import analyze as run_analysis
from config import SPRING_CALLBACK_URL

# 블루 프린트 객체를 생성하여 준다. 
analyze_bp = Blueprint("analyze", __name__)


def _notify_spring(payload: dict):
    """Spring 콜백 API에 결과를 전송합니다."""
    try:
        requests.post(SPRING_CALLBACK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"[WARN] Spring 콜백 전송 실패: {e}")

# 파이브 라인 실행
def _run_pipeline(task_id: str, article_id: int, text: str, input_type: str):
    """전처리 → 요약 → 임베딩 파이프라인 (별도 스레드에서 실행)"""
    try:
        update_task(task_id, TaskStatus.ANALYZING)

        # Step 1: 전처리 
        #1.) html 태그를 제거한다.
        is_html = input_type == "URL"
        preprocessed = preprocess(text, is_html=is_html)
        cleaned_text = preprocessed["cleaned"]

        # Step 2: Zero-shot + Constraint + Role 요약 / 키워드 생성
        summary = summarize(cleaned_text)

        # Step 3: Few-shot 각 섹션에 맞는 편향도를 반환
        label_result = label(cleaned_text)
        bias_label = label_result.get("label", "uncertain")

        # Step 4: Generated Knowledge + CoT 편향 분석 (Prompt Merging + Model Tiering)
        analysis = run_analysis(
            text=cleaned_text,
            topic=summary.get("topic", ""),
            keywords=summary.get("keywords", []),
            bias_label=bias_label,
            sentences=preprocessed["sentences"],
        )

        # Step 5: ChromaDB 임베딩 저장
        embed_and_store(
            article_id=article_id,
            text=cleaned_text,
            metadata={
                "topic": summary.get("topic", ""),
                "keywords": summary.get("keywords", []),
            },
        )

        result = {
            "article_id": article_id,
            "one_line_summary": summary.get("one_line_summary", ""),
            "key_facts": summary.get("key_facts", []),
            "keywords": summary.get("keywords", []),
            "topic": summary.get("topic", ""),
            "sentence_count": len(preprocessed["sentences"]),
            "sources": preprocessed.get("sources", []),
            "bias_label": bias_label,
            "bias_confidence": label_result.get("confidence", 0.0),
            "bias_reason": label_result.get("reason", ""),
            "highlighted_sentences": analysis.get("highlighted_sentences", []),
            "bias_direction": analysis.get("bias_direction", "center"),
            "spectrum_label": analysis.get("spectrum_label", "중립"),
            "emotion_neutrality": analysis.get("emotion_neutrality", 0.5),
            "fact_ratio": analysis.get("fact_ratio", 0.5),
            "source_balance": analysis.get("source_balance", 0.5),
            "bias_score": analysis.get("bias_score", 0.5),
            "total_score": analysis.get("total_score", 50),
        }

        update_task(task_id, TaskStatus.DONE, result=result)

        # Step 4: Spring에 완료 콜백 전송
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
