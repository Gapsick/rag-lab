"""
대화 세대(generation) 요약 + 장기 기억(fact) 추출 모듈.
Context Window의 한 세대(user+assistant 5쌍)를 짧은 요약으로 압축하고,
이후에도 기억할 가치가 있는 사용자 관련 사실을 별도로 추출한다.
"""

import json
import re
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(Path(__file__).parent.parent / ".env")
client = Anthropic()

SUMMARY_SYSTEM = (
    "다음은 사용자와 AI 어시스턴트의 대화 일부입니다.\n"
    "이후 대화에서 문맥으로 참고할 수 있도록, 어떤 질문들이 있었고 "
    "어떤 답변/결론이 나왔는지를 한국어 3~5문장으로 간결하게 요약하세요.\n"
    "요약문만 출력하세요."
)

FACT_EXTRACTION_SYSTEM = (
    "다음은 사용자와 AI 어시스턴트의 대화 일부입니다.\n"
    "이 대화에서 앞으로의 대화에서도 계속 기억해두면 좋을 "
    "사용자에 대한 사실(선호, 배경지식 수준, 관심 주제, 진행 중인 작업, "
    "확정된 결정 등)이 있다면 짧은 문장으로 추출하세요.\n"
    "단순한 질문/답변 내용이나 일회성 정보는 제외하세요.\n"
    "기억할 만한 내용이 없으면 빈 배열을 반환하세요.\n"
    "반드시 아래 JSON 배열만 출력하세요 (설명 없이):\n"
    '["사실 1", "사실 2"]'
)


def summarize_messages(messages: list[dict]) -> str:
    text = "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    return resp.content[0].text.strip()


def extract_facts(messages: list[dict]) -> list[str]:
    text = "\n\n".join(f"[{m['role']}] {m['content']}" for m in messages)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=FACT_EXTRACTION_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    raw = resp.content[0].text.strip()
    try:
        facts = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if not match:
            return []
        try:
            facts = json.loads(match.group())
        except json.JSONDecodeError:
            return []
    return [f for f in facts if isinstance(f, str) and f.strip()]
