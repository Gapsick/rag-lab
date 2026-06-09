"""
Agentic RAG - Claude가 직접 검색 전략을 결정

고정 파이프라인(rag.py)과의 차이:
  - 검색 쿼리를 Claude가 직접 만들고 결정
  - 결과가 불충분하면 다른 쿼리로 재검색 가능
  - 검색 횟수와 순서를 LLM이 판단
"""

from dotenv import load_dotenv
from pathlib import Path
from anthropic import Anthropic
from core import vectordb
from core import doc_context

load_dotenv(Path(__file__).parent.parent / ".env")

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
client       = Anthropic()
MAX_TURNS    = 6   # 최대 검색 횟수 제한

AGENT_SYSTEM = (
    "당신은 업로드된 자료 기반 Q&A 에이전트입니다.\n"
    "자료에 관한 질문에는 반드시 search_lecture 도구를 먼저 사용하여 검색한 후 답하세요.\n"
    "검색 결과가 불충분하거나 관련 없으면 다른 키워드로 재검색할 수 있습니다.\n"
    "자료에 없는 내용은 '해당 자료에 내용이 없습니다'라고 솔직히 말하세요.\n"
    "인사, 잡담 등 자료와 완전히 무관한 질문만 도구 없이 바로 답하세요."
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
            }
        },
        "required": ["query"]
    }
}]


def _search(query: str) -> tuple[str, list[str]]:
    """검색 실행 → (Claude에게 줄 텍스트, 출처 목록)"""
    chunks = vectordb.search(query, top_k=3)
    if not chunks:
        return "검색 결과 없음", []

    parts, sources = [], []
    for content, sim, src in chunks:
        parts.append(f"[출처: {src}] (유사도: {sim:.3f})\n{content[:400]}")
        if sim >= 0.3:
            sources.append(src)

    return "\n\n---\n\n".join(parts), sources


def answer(query: str, history: list[dict] | None = None) -> dict:
    messages   = (history or []) + [{"role": "user", "content": query}]
    tool_calls = []
    all_sources = []

    for _ in range(MAX_TURNS):
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=doc_context.get_prompt(AGENT_SYSTEM),
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
                    "tool_calls": tool_calls,
                    "turns": len(tool_calls),
                }
            }

        # 도구 호출 처리
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue

            search_query          = block.input.get("query", "")
            result_text, sources  = _search(search_query)
            all_sources.extend(sources)

            tool_calls.append({
                "query":   search_query,
                "sources": sources,
                "preview": result_text[:150],
            })

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_text,
            })

        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": "최대 검색 횟수에 도달했습니다.",
        "sources": list(dict.fromkeys(all_sources)),
        "debug": {"mode": "agent", "tool_calls": tool_calls, "turns": MAX_TURNS}
    }
