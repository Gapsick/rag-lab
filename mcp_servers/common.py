"""
컬렉션(=MCP 서버) 공통 로직.
각 MCP 서버는 자신의 collection 값으로 build_server()를 호출해
list_documents / search_document / search_all 3개 tool을 가진
FastMCP 인스턴스를 받는다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP
from core import vectordb


def _doc_name(source: str) -> str:
    """'파일명.pdf - 3페이지 (Vision)' → '파일명.pdf'"""
    return source.split(" - ")[0] if " - " in source else source


def _format_chunks(chunks: list[tuple[str, float, str, str]]) -> str:
    if not chunks:
        return "관련 내용을 찾지 못했습니다."
    parts, seen = [], set()
    for _, sim, src, parent_content in chunks:
        if src in seen:
            continue
        seen.add(src)
        parts.append(f"[출처: {src}] (유사도: {sim:.3f})\n{parent_content}")
    return "\n\n---\n\n".join(parts)


def build_server(name: str, collection: str, label: str) -> FastMCP:
    """collection으로 검색 범위가 제한된 FastMCP 서버를 생성한다."""
    mcp = FastMCP(
        name,
        instructions=f"'{label}' 관련 문서를 검색하는 도구 모음입니다. "
                      "list_documents로 보유 문서를 먼저 확인한 뒤 검색하세요.",
    )

    @mcp.tool()
    def list_documents() -> str:
        """현재 색인된 문서 목록을 (파일명, 청크 수)로 반환한다.
        search_document를 호출하기 전에 어떤 문서가 있는지 확인할 때 사용한다.
        """
        sources = vectordb.get_sources(collection=collection)
        if not sources:
            return "색인된 문서가 없습니다."

        counts: dict[str, int] = {}
        for s in sources:
            doc = _doc_name(s["source"])
            counts[doc] = counts.get(doc, 0) + s["count"]

        return "\n".join(f"- {doc} ({count}개 청크)" for doc, count in counts.items())

    @mcp.tool()
    def search_document(query: str, document: str) -> str:
        """특정 문서 안에서만 관련 내용을 검색한다.
        document에는 list_documents로 확인한 파일명을 정확히 입력한다.
        """
        chunks = vectordb.search(query, top_k=3, source_prefix=document, collection=collection)
        return _format_chunks(chunks)

    @mcp.tool()
    def search_all(query: str) -> str:
        """색인된 모든 문서를 대상으로 관련 내용을 검색한다.
        어떤 문서에 답이 있을지 모를 때, 또는 여러 문서를 아울러 검색할 때 사용한다.
        """
        chunks = vectordb.search(query, top_k=5, collection=collection)
        return _format_chunks(chunks)

    return mcp


async def get_tool_specs(collection: str, label: str) -> list[dict]:
    """LLM에게 노출되는 tool 목록을 (이름, 설명, 파라미터)로 반환한다."""
    mcp = build_server("tmp", collection=collection, label=label)
    tools = await mcp.list_tools()
    return [
        {
            "name": t.name,
            "description": (t.description or "").strip(),
            "params": list((t.inputSchema or {}).get("properties", {}).keys()),
        }
        for t in tools
    ]
