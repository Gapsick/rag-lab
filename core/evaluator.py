"""
RAG 평가 모듈
- Faithfulness:      답변이 컨텍스트에만 근거하는가
- Answer Relevance:  답변이 질문에 답하는가
- Context Relevance: 검색된 청크가 질문과 관련 있는가
"""

import json
import re
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic
from prompts.evaluate import EVALUATE_PROMPT

load_dotenv(Path(__file__).parent.parent / ".env")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
client = Anthropic()


def evaluate(query: str, context: str, answer: str) -> dict:
    """
    RAG 결과를 3개 지표로 평가.
    반환: {
      "faithfulness":      {"score": 1~5, "reason": str},
      "answer_relevance":  {"score": 1~5, "reason": str},
      "context_relevance": {"score": 1~5, "reason": str},
    }
    """
    prompt = (
        f"[질문]\n{query}\n\n"
        f"[검색된 컨텍스트]\n{context}\n\n"
        f"[생성된 답변]\n{answer}"
    )
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=EVALUATE_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        text = resp.content[0].text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError("JSON 파싱 실패")
    except Exception as e:
        return {
            "faithfulness":      {"score": 0, "reason": f"평가 실패: {e}"},
            "answer_relevance":  {"score": 0, "reason": ""},
            "context_relevance": {"score": 0, "reason": ""},
        }
