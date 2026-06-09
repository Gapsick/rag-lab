"""
RAG 파이프라인
"""

import json
import re

from dotenv import load_dotenv
from pathlib import Path
from anthropic import Anthropic
from core import vectordb
from core import doc_context
from prompts.rag import RAG_SYSTEM_PROMPT
from prompts.query_rewrite import QUERY_REWRITE_PROMPT
from prompts.intent import INTENT_PROMPT
from prompts.rerank import RERANK_PROMPT

load_dotenv(Path(__file__).parent.parent / ".env")

CLAUDE_MODEL        = "claude-haiku-4-5-20251001"
client              = Anthropic()
SIMILARITY_THRESHOLD = 0.3
RERANK_TOP_N        = 5   # 벡터 검색 후보 수
RERANK_USE_N        = 3   # re-ranking 후 실제 사용 수
RERANK_MIN_SCORE    = 5   # re-ranking 최소 사용 점수


# ── 공통 JSON 파서 ────────────────────────────────────────────
def _parse_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("JSON을 찾을 수 없음")


# ── STEP 1: 의도 파악 ─────────────────────────────────────────
def _check_intent(query: str) -> dict:
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=100,
            system=INTENT_PROMPT,
            messages=[{"role": "user", "content": query}]
        )
        return _parse_json(resp.content[0].text)
    except Exception as e:
        return {"is_lecture": True, "reason": f"판단 실패 → 강의 질문으로 처리 ({e})"}


# ── STEP 2: Query Rewriting ───────────────────────────────────
def _rewrite_query(query: str) -> dict:
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            system=QUERY_REWRITE_PROMPT,
            messages=[{"role": "user", "content": query}]
        )
        return _parse_json(resp.content[0].text)
    except Exception as e:
        return {"rewritten": query, "reason": f"재작성 실패 → 원본 사용 ({e})"}


# ── STEP 4: Re-ranking ────────────────────────────────────────
def _rerank(query: str, chunks: list) -> list:
    docs = [
        {"index": i, "source": src, "content": content[:300]}
        for i, (content, _, src) in enumerate(chunks)
    ]
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            system=RERANK_PROMPT,
            messages=[{
                "role": "user",
                "content": f"질문: {query}\n\n문서:\n{json.dumps(docs, ensure_ascii=False, indent=2)}"
            }]
        )
        results = _parse_json(resp.content[0].text)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return results
    except Exception as e:
        # 실패 시 원래 순서, 점수 5 유지
        return [{"index": i, "score": 5, "reason": f"re-ranking 실패: {e}"}
                for i in range(len(chunks))]


# ── 메인 파이프라인 ───────────────────────────────────────────
def answer(query: str, history: list[dict] | None = None) -> dict:

    # STEP 1: 의도 파악
    intent = _check_intent(query)
    if not intent.get("is_lecture", True):
        # 문서 무관 질문 → RAG 없이 직접 답변
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=(history or []) + [{"role": "user", "content": query}]
        )
        return {
            "answer": resp.content[0].text,
            "sources": [],
            "debug": {"query": query, "intent": intent},
        }

    # STEP 2: Query Rewriting
    rewrite = _rewrite_query(query)
    search_query = rewrite.get("rewritten", query)

    # STEP 3: 벡터 검색 (후보 더 많이 가져옴)
    chunks = vectordb.search(search_query, top_k=RERANK_TOP_N)

    debug = {
        "query": query,
        "intent": intent,
        "rewrite": rewrite,
        "retrieved": [
            {
                "rank": i + 1,
                "source": src,
                "similarity": round(sim, 3),
                "preview": content[:120].replace('\n', ' '),
            }
            for i, (content, sim, src) in enumerate(chunks)
        ],
    }

    if not chunks:
        return {
            "answer": "강의자료가 색인되지 않았습니다.",
            "sources": [],
            "debug": debug,
        }

    # STEP 4: Re-ranking
    rerank_results = _rerank(query, chunks)

    debug["reranked"] = [
        {
            "original_rank": r["index"] + 1,
            "new_rank": i + 1,
            "source": chunks[r["index"]][2],
            "similarity": round(chunks[r["index"]][1], 3),
            "score": r.get("score", 0),
            "reason": r.get("reason", ""),
            "used": i < RERANK_USE_N and r.get("score", 0) >= RERANK_MIN_SCORE,
        }
        for i, r in enumerate(rerank_results)
    ]

    relevant_chunks = [
        chunks[r["index"]]
        for i, r in enumerate(rerank_results)
        if i < RERANK_USE_N and r.get("score", 0) >= RERANK_MIN_SCORE
    ]
    debug["used"] = len(relevant_chunks)

    if not relevant_chunks:
        return {
            "answer": "질문과 관련된 강의자료 내용을 찾지 못했습니다. 다른 키워드로 질문해보세요.",
            "sources": [],
            "debug": debug,
        }

    # STEP 5: 답변 생성
    context = "\n\n---\n\n".join(
        f"[출처: {src}]\n{content}" for content, _, src in relevant_chunks
    )
    sources = [src for _, _, src in relevant_chunks]

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=doc_context.get_prompt(RAG_SYSTEM_PROMPT),
        messages=(history or []) + [{
            "role": "user",
            "content": f"[참고 자료]\n{context}\n\n[질문] {query}"
        }]
    )

    return {
        "answer": resp.content[0].text,
        "sources": sources,
        "debug": {
            **debug,
            "eval_context": context,   # 평가용 전체 컨텍스트
        },
    }
