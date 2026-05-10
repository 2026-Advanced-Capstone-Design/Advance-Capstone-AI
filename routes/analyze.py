import threading
import requests
from flask import Blueprint, request, jsonify
from models.task import create_task, update_task, TaskStatus
from services.preprocessor import preprocess
from services.summarizer import summarize
from services.labeler import label
from services.analyzer import analyze as run_analysis
from services.factcheck import check_facts
from config import SPRING_CALLBACK_URL

# 블루 프린트 객체를 생성하여 준다. 
analyze_bp = Blueprint("analyze", __name__)


_NEUTRAL_LABELS = {"neutral", "balanced"}

def _compute_section_bias_score(sections: list) -> float:
    """
    labeler.py 섹션별 편향 결과 → 섹션 편향도 점수 (0.0~1.0, 높을수록 중립).
    neutral/balanced: confidence 그대로 (확신할수록 좋음)
    편향 레이블:       1 - confidence (확신할수록 나쁨)
    """
    if not sections:
        return 0.5
    scores = []
    for s in sections:
        label = s.get("bias_label", "neutral")
        conf  = s.get("confidence", 0.5)
        scores.append(conf if label in _NEUTRAL_LABELS else 1.0 - conf)
    return round(sum(scores) / len(scores), 3)


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
        # Spring CrawlerService가 Jsoup으로 이미 HTML→plain text 변환 후 전송하므로
        # input_type 무관하게 항상 plain text로 처리
        preprocessed = preprocess(text, is_html=False)
        cleaned_text = preprocessed["cleaned"]

        # Step 2: 압축 본문 / 키워드 생성
        summary = summarize(cleaned_text)
        compressed_text = summary.get("compressed_text") or cleaned_text

        # Step 3: 섹션 분리 + 섹션별 편향 라벨링
        label_result = label(compressed_text)
        bias_label = label_result.get("overall_label", "uncertain")

        # Step 3-1: Google Fact Check API 검증 (혼합 방식 — 결과 없으면 CoT fallback)
        factcheck_result    = check_facts(summary.get("key_facts", []))
        external_fact_ratio = factcheck_result.get("fact_ratio")  # None이면 CoT 기반으로 대체
        fact_check_results  = factcheck_result.get("results", [])
        section_bias_score  = _compute_section_bias_score(label_result.get("sections", []))

        # Step 4: Generated Knowledge + CoT 편향 분석 (Prompt Merging + Model Tiering)
        analysis = run_analysis(
            text=compressed_text,
            topic=summary.get("topic", ""),
            keywords=summary.get("keywords", []),
            bias_label=bias_label,
            sentences=preprocessed["sentences"],
        )

        result = {
            "article_id": article_id,
            "compressed_text": compressed_text,
            "key_facts": summary.get("key_facts", []),
            "keywords": summary.get("keywords", []),
            "topic": summary.get("topic", ""),
            "sentence_count": len(preprocessed["sentences"]),
            "sections": label_result.get("sections", []),
            "bias_label": bias_label,
            "bias_confidence": label_result.get("overall_confidence", 0.0),
            "bias_reason": label_result.get("overall_reason", ""),
            "highlighted_sentences": analysis.get("highlighted_sentences", []),
            "bias_direction": analysis.get("bias_direction", "center"),
            "spectrum_label": analysis.get("spectrum_label", "중립"),
            "emotion_neutrality": analysis.get("emotion_neutrality", 0.5),
            # 혼합 방식: Google Fact Check 결과 우선, 없으면 CoT fact_basis 반전값 사용
            "fact_ratio": external_fact_ratio if external_fact_ratio is not None else analysis.get("fact_ratio", 0.5),
            "fact_ratio_source": "google" if external_fact_ratio is not None else "cot",
            "omission_neutrality": analysis.get("omission_neutrality", 0.5),
            "bias_score": analysis.get("bias_score", 0.5),
            # 총점 = (사실 기반도 + 감정 중립도 + 섹션별 편향도) / 3
            "section_bias_score": section_bias_score,
            "total_score": int((
                (external_fact_ratio if external_fact_ratio is not None else analysis.get("fact_ratio", 0.5))
                + analysis.get("emotion_neutrality", 0.5)
                + section_bias_score
            ) / 3 * 100),
            "background": analysis.get("background", ""),
            "cot_emotion_reason":    analysis.get("cot_emotion_reason", ""),
            "cot_fact_ratio_reason": analysis.get("cot_fact_ratio_reason", ""),
            "fact_check_results":    fact_check_results,
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
