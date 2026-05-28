import os
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from flask import Blueprint, request, jsonify
from models.task import create_task, update_task, TaskStatus
from services.preprocessor import preprocess
from services.summarizer import summarize
from services.labeler import label
from services.factcheck import check_facts
from services.analyzer import analyze as run_analysis, precompute_background
from config import SPRING_CALLBACK_URL

MOCK_MODE = os.environ.get("MOCK_MODE", "false").lower() == "true"

MOCK_RESULT = {
    "key_facts": ["mock 핵심 사실 1입니다.", "mock 핵심 사실 2입니다.", "mock 핵심 사실 3입니다."],
    "keywords": ["테스트", "부하", "성능"],
    "topic": "부하 테스트",
    "bias_label": "neutral",
    "bias_confidence": 0.85,
    "bias_reason": "mock 데이터입니다.",
    "sections": [],
    "highlighted_sentences": [],
    "emotion_neutrality": 0.8,
    "fact_ratio": 0.7,
    "bias_score": 0.2,
    "total_score": 80,
    "cot_emotion_reason": "mock",
    "fact_check_reason": "mock",
}

analyze_bp = Blueprint("analyze", __name__)


def _notify_spring(payload: dict):
    try:
        requests.post(SPRING_CALLBACK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"[WARN] Spring 콜백 전송 실패: {e}")


def _run_pipeline(task_id: str, article_id: int, text: str, input_type: str):
    try:
        update_task(task_id, TaskStatus.ANALYZING)

        if MOCK_MODE:
            result = {"article_id": article_id, **MOCK_RESULT}
            update_task(task_id, TaskStatus.DONE, result=result)
            _notify_spring({"status": "DONE", **result})
            return

        # Step 1: 전처리
        is_html = input_type == "URL"
        preprocessed = preprocess(text, is_html=is_html)
        cleaned_text = preprocessed["cleaned"]

        # Step 2 + Step 4: 요약/라벨 병렬 실행
        # summarize 완료 즉시 background 사전 적재 → label 남은 시간에 겹쳐 실행
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_summary = executor.submit(summarize, cleaned_text)
            future_label   = executor.submit(label, cleaned_text)

            # summarize가 끝나는 순간 background 생성 시작 (label은 아직 실행 중)
            summary = future_summary.result()
            future_bg = executor.submit(
                precompute_background,
                summary.get("topic", ""),
                summary.get("keywords", []),
            )

            label_result = future_label.result()  # label 완료 대기
            future_bg.result()                    # background도 완료 보장

        bias_label = label_result.get("overall_label", "uncertain")

        # Step 3 + Step 5: Fact Check & 편향 분석 병렬 실행
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_factcheck = executor.submit(check_facts, summary.get("key_facts", []))
            future_analysis = executor.submit(
                run_analysis,
                text=cleaned_text,
                topic=summary.get("topic", ""),
                keywords=summary.get("keywords", []),
                bias_label=bias_label,
                sentences=preprocessed["sentences"],
                sections=label_result.get("sections", []),
            )
            factcheck_result = future_factcheck.result()
            analysis = future_analysis.result()

        # fact_ratio: Google 결과 우선, 없으면 CoT 기반
        fact_ratio_source = factcheck_result.get("fact_ratio")
        fact_ratio = fact_ratio_source if fact_ratio_source is not None \
            else analysis.get("fact_ratio", 0.5)

        bias_score = analysis.get("bias_score", 0.5)
        total_score = round((1.0 - bias_score) * 100)

        result = {
            "article_id": article_id,
            "key_facts": summary.get("key_facts", []),
            "keywords": summary.get("keywords", []),
            "topic": summary.get("topic", ""),
            "bias_label": bias_label,
            "bias_confidence": label_result.get("overall_confidence", 0.0),
            "bias_reason": label_result.get("overall_reason", ""),
            "sections": label_result.get("sections", []),
            "highlighted_sentences": analysis.get("highlighted_sentences", []),
            "emotion_neutrality": analysis.get("emotion_neutrality", 0.5),
            "fact_ratio": fact_ratio,
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
