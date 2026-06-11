"""
Agentic RAG — LLM 자율 플래닝 (학습용)

Modular RAG와의 차이:
  - Intent/Rewrite/Rerank 코드 없음 — Claude가 스스로 판단
  - 검색 쿼리를 Claude가 직접 만들고 결정
  - 결과가 불충분하면 다른 쿼리로 재검색 가능
  - 검색 횟수와 순서를 LLM이 자율 결정 (최대 MAX_TURNS)
"""

from dotenv import load_dotenv
from pathlib import Path
from anthropic import Anthropic
from core import vectordb
from core import doc_context

load_dotenv(Path(__file__).parent.parent.parent / ".env")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
client       = Anthropic()
MAX_TURNS    = 6   # 최대 검색 횟수 제한

AGENT_SYSTEM = (
    "당신은 업로드된 자료 기반 Q&A 에이전트입니다.\n"
    "자료에 관한 질문에는 반드시 search_lecture 도구를 먼저 사용하여 검색한 후 답하세요.\n"
    "특정 개념·기술·용어에 대한 질문은, 당신이 일반 지식으로 이미 알고 있는 "
    "내용이라도 업로드된 자료에 관련 설명이 있을 수 있으므로 먼저 검색하세요. "
    "자기 지식만으로 바로 답하지 마세요.\n"
    "자료에 없는 내용은 '해당 자료에 내용이 없습니다'라고 솔직히 말하세요.\n"
    "인사, 잡담 등 자료와 완전히 무관한 질문만 도구 없이 바로 답하세요.\n"
    "\n"
    "search_lecture 호출 시 인자를 다음과 같이 채우세요:\n"
    "  - reason: 이 검색어를 선택한 근거. 처음 검색이면 사용자 질문의 표현을 그대로 "
    "썼는지/다르게 바꿨는지와 그 이유를 설명.\n"
    "  - prev_issue: 재검색일 때만 작성. 직전 검색 결과의 무엇이 문제였는지 "
    "(예: 유사도가 낮음 / 다른 주제 내용임 / 질문의 일부에만 답이 됨 등)를 구체적으로 "
    "설명하고, 그래서 이번엔 검색어를 어떻게 바꿨는지 reason에 이어서 적으세요.\n"
    "\n"
    "검색 결과를 받으면 질문에 답하기 충분한지 먼저 판단하세요.\n"
    "충분하면 그 결과로 바로 답변하고, 부족하거나 관련이 없으면 다른 키워드로 재검색하세요."
)

TOOLS = [{
    "name": "search_lecture",
    "description": "강의자료 벡터DB에서 관련 청크를 검색합니다. 결과가 부족하면 다른 쿼리로 재호출 가능합니다.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "검색 쿼리. 구체적인 키워드와 영어 표현을 함께 쓰면 더 정확합니다."
            },
            "reason": {
                "type": "string",
                "description": (
                    "이 검색어를 선택한 근거. 첫 검색이면 사용자의 원래 질문 표현을 "
                    "그대로 썼는지/다르게 바꿨는지와 그 이유, 재검색이면 이번 검색어를 "
                    "어떻게/왜 바꿨는지를 설명."
                )
            },
            "prev_issue": {
                "type": "string",
                "description": (
                    "재검색일 때만 작성: 직전 검색 결과의 어떤 점이 부족하거나 "
                    "잘못되어 재검색하는지 (예: 유사도 낮음, 주제 불일치, 질문의 "
                    "일부만 커버 등). 첫 검색이면 빈 문자열."
                )
            }
        },
        "required": ["query", "reason", "prev_issue"]
    }
}]


def _search(query: str) -> tuple[str, list[str]]:
    """검색 실행 → (Claude에게 줄 텍스트, 출처 목록)"""
    chunks = vectordb.search(query, top_k=3)
    if not chunks:
        return "검색 결과 없음", []

    # parent_content(페이지 전체 등)를 컨텍스트로 사용, 같은 출처는 중복 제거
    parts, sources, seen = [], [], set()
    for content, sim, src, parent_content in chunks:
        if src not in seen:
            seen.add(src)
            parts.append(f"[출처: {src}] (유사도: {sim:.3f})\n{parent_content}")
        if sim >= 0.3:
            sources.append(src)

    return "\n\n---\n\n".join(parts), sources


def _with_facts(system: str, facts: list[str] | None) -> str:
    if not facts:
        return system
    facts_block = "\n".join(f"- {f}" for f in facts)
    return f"{system}\n\n[사용자에 대해 기억하고 있는 것]\n{facts_block}"


def answer(query: str, history: list[dict] | None = None, facts: list[str] | None = None) -> dict:
    messages   = (history or []) + [{"role": "user", "content": query}]
    tool_calls = []
    all_sources = []

    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=_with_facts(doc_context.get_prompt(AGENT_SYSTEM), facts),
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        # 도구 호출 없이 답변 완료
        if resp.stop_reason == "end_turn":
            text = "".join(b.text for b in resp.content if b.type == "text")
            return {
                "answer": text,
                "sources": list(dict.fromkeys(all_sources)),
                "debug": {
                    "mode": "agent",
                    "query": query,
                    "tools": TOOLS,
                    "tool_calls": tool_calls,
                    "turns": len(tool_calls),
                }
            }

        # 도구 호출 처리
        # tool_use 직전에 Claude가 내놓는 text 블록 = 이번 턴의 planning(추론)
        thought = "".join(b.text for b in resp.content if b.type == "text").strip()

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            search_query          = block.input.get("query", "")
            search_reason         = block.input.get("reason", "")
            prev_issue            = block.input.get("prev_issue", "")
            result_text, sources  = _search(search_query)
            all_sources.extend(sources)

            tool_calls.append({
                "thought":    thought,
                "reason":     search_reason,
                "prev_issue": prev_issue,
                "query":      search_query,
                "sources":    sources,
                "preview":    result_text[:150],
                "content":    result_text,
            })

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_text,
            })

        # 도구 호출 없이 (예: max_tokens로 중간에 끊김) 종료된 경우
        # → 빈 user 메시지를 보내면 API 오류가 나므로 지금까지의 텍스트로 답변 종료
        if not tool_results:
            return {
                "answer": thought or "응답이 중간에 잘렸습니다. 다시 질문해주세요.",
                "sources": list(dict.fromkeys(all_sources)),
                "debug": {
                    "mode": "agent",
                    "query": query,
                    "tools": TOOLS,
                    "tool_calls": tool_calls,
                    "turns": len(tool_calls),
                }
            }

        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": "최대 검색 횟수에 도달했습니다.",
        "sources": list(dict.fromkeys(all_sources)),
        "debug": {"mode": "agent", "query": query, "tools": TOOLS, "tool_calls": tool_calls, "turns": MAX_TURNS}
    }
