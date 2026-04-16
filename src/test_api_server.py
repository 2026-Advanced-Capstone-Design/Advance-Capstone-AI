import requests
import json

API_URL = "http://localhost:8000"

# 1. 헬스체크
print("===== 서버 상태 확인 =====")
res = requests.get(f"{API_URL}/health")
print(res.json())

# 2. 단일 댓글 분석 (POST /analyze/comments 에 1개 전송)
print("\n===== 단일 댓글 분석 =====")
res = requests.post(
    f"{API_URL}/analyze/comments",
    json=[{"text": "진짜 편파보도 너무하다 ㅋㅋ"}]
)
result = res.json()
if res.status_code != 200:
    print(f"오류: {res.status_code} - {result}")
else:
    c = result['comments'][0]
    print(f"감정: {c['sentiment']} ({c['sentiment_score']:.0%})")
    print(f"봇: {'의심' if c['is_bot'] else '정상'} ({c['bot_score']}점)")

# 3. 댓글 목록 분석
print("\n===== 댓글 목록 분석 =====")
comments = [
    {"text": "진짜 편파보도 너무하다 ㅋㅋ"},
    {"text": "좋은 기사 감사합니다!"},
    {"text": "오늘 법안이 통과됐군요"},
    {"text": "이게 말이 돼? 완전 거짓말이잖아"},
    {"text": "유익한 정보 잘 봤습니다"},
    {"text": "이 정책은 반드시 지지해야 합니다. 국민을 위한 올바른 방향입니다."},
    {"text": "이 정책은 반드시 지지해야 합니다. 국민을 위한 올바른 방향입니다."},
    {"text": "수박들 싹다 정리합시다"},
    {"text": "내용 확인했습니다"},
    {"text": "진짜 열받는다....."},
]

res = requests.post(
    f"{API_URL}/analyze/comments",
    json=comments
)
result = res.json()

if res.status_code != 200:
    print(f"오류: {res.status_code} - {result}")
else:
    print(f"\n총 댓글: {result['total']}개")
    print(f"😊 긍정: {result['positive']}개 ({result['positive_pct']}%)")
    print(f"😡 부정: {result['negative']}개 ({result['negative_pct']}%)")
    print(f"😐 중립: {result['neutral']}개 ({result['neutral_pct']}%)")
    print(f"🤖 봇 의심: {result['bot_count']}개 ({result['bot_pct']}%)")

    if result.get('summary'):
        print(f"\n📝 요약:\n{result['summary']}")

    print("\n----- 개별 댓글 결과 -----")
    emoji_map = {"긍정": "😊", "부정": "😡", "중립": "😐"}
    for c in result['comments']:
        emoji = emoji_map.get(c['sentiment'], "😐")
        bot_icon = "🤖" if c['is_bot'] else "👤"
        text = c['text'][:35] + "..." if len(c['text']) > 35 else c['text']
        reasons_str = f" [{', '.join(c['bot_reasons'])}]" if c['is_bot'] and c['bot_reasons'] else ""
        print(f"{emoji}[{c['sentiment']} {c['sentiment_score']:.0%}] {bot_icon}[봇{c['bot_score']}점]{reasons_str} {text}")

# 4. YouTube 영상 댓글 분석
print("\n===== YouTube 댓글 분석 =====")
video_input = input("유튜브 영상 ID 또는 URL 입력 (엔터 시 건너뜀): ").strip()

if not video_input:
    print("건너뜀.")
else:
    if "youtube.com/watch?v=" in video_input:
        video_id = video_input.split("v=")[1].split("&")[0]
    elif "youtu.be/" in video_input:
        video_id = video_input.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = video_input

    res = requests.get(f"{API_URL}/analyze/youtube/{video_id}")

    if res.status_code != 200:
        print(f"오류: {res.status_code} - {res.json().get('detail', '')}")
    else:
        result = res.json()
        print(f"\n🎬 영상: {result['video_title']}")
        print(f"💬 전체 댓글 수: {result['video_comment_count']}개")
        print(f"\n📊 분석 결과 ({result['total']}개 댓글)")
        print(f"😊 긍정: {result['positive']}개 ({result['positive_pct']}%)")
        print(f"😡 부정: {result['negative']}개 ({result['negative_pct']}%)")
        print(f"😐 중립: {result['neutral']}개 ({result['neutral_pct']}%)")
        print(f"🤖 봇 의심: {result['bot_count']}개 ({result['bot_pct']}%)")

        if result.get('summary'):
            print(f"\n📝 요약:\n{result['summary']}")

        bot_comments = [c for c in result['comments'] if c['is_bot']]
        if bot_comments:
            print("\n⚠️  봇 의심 댓글")
            print("-" * 60)
            for c in bot_comments:
                text = c['text'][:40] + "..." if len(c['text']) > 40 else c['text']
                reasons_str = ", ".join(c['bot_reasons']) if c['bot_reasons'] else "AI탐지"
                print(f"🤖 [봇점수 {c['bot_score']}점] [{reasons_str}]")
                print(f"   {text}")

        print("\n📋 전체 댓글")
        print("-" * 60)
        emoji_map = {"긍정": "😊", "부정": "😡", "중립": "😐"}
        for c in result['comments']:
            emoji = emoji_map.get(c['sentiment'], "😐")
            bot_icon = "🤖" if c['is_bot'] else "👤"
            text = c['text'][:35] + "..." if len(c['text']) > 35 else c['text']
            print(f"{emoji}[{c['sentiment']} {c['sentiment_score']:.0%}] {bot_icon}[봇{c['bot_score']}점] {text}")
