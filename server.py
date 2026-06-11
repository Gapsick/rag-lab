"""
강의자료 Q&A 백엔드 - FastAPI

실행:
  python server.py
"""

import asyncio
import json
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

from core import vectordb
from core import doc_context
from core import conversation
from core.pipelines import modular_rag, agentic_rag
from core.pipelines.agentic_rag import AGENT_SYSTEM
from core.evaluator import evaluate
from core.pdf_loader import load_pdf, stream_pdf
from prompts.intent import INTENT_PROMPT
from prompts.query_rewrite import QUERY_REWRITE_PROMPT
from prompts.rerank import RERANK_PROMPT
from prompts.rag import RAG_SYSTEM_PROMPT
import core.pdf_loader as _fitz_loader
# Docling은 torch/transformers를 포함해 무거움 → reload 시간을 줄이기 위해
# 실제로 사용하는 시점(loader == "docling")에만 import
_docling_loader = None

load_dotenv(Path(__file__).parent / ".env")


# next(gen)을 run_in_executor에 직접 넘기면 StopIteration이 asyncio Future로
# 전파되며 TypeError가 발생함(PEP 479) → sentinel로 감싸서 종료를 표현
_SENTINEL = object()


def _next_or_sentinel(gen):
    try:
        return next(gen)
    except StopIteration:
        return _SENTINEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectordb.wait_for_db()
    vectordb.setup()
    yield


app = FastAPI(lifespan=lifespan)

# Vision 처리 취소 플래그
_cancel_event = threading.Event()


@app.get("/", response_class=HTMLResponse)
async def root():
    return open("index.html", encoding="utf-8").read()


@app.post("/upload")
async def upload(file: UploadFile = File(...), collection: str = Form(vectordb.DEFAULT_COLLECTION)):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        chunks = load_pdf(tmp_path, filename=file.filename)
        vectordb.insert(chunks, collection=collection)
        doc_context.generate_and_store(chunks)
        return {"ok": True, "message": f"{file.filename} — {len(chunks)}페이지 색인 완료"}
    finally:
        os.unlink(tmp_path)


@app.post("/preview-stream")
async def preview_stream(
    file: UploadFile = File(...),
    vision: bool = Form(False),
    loader: str = Form("fitz"),
    page_start: int = Form(None),
    page_end: int = Form(None),
):
    """청크 실시간 스트리밍 - 페이지 처리될 때마다 SSE로 전송
    loader: "fitz" (기본, 학습용) | "docling" (오픈소스 ML)
    """
    _cancel_event.clear()
    data = await file.read()
    filename = file.filename

    # loader 파라미터로 사용할 stream_pdf 함수 선택
    # Docling은 처음 요청 시에만 import (torch/transformers 로딩 → 수 초~수십 초 소요)
    global _docling_loader
    if loader == "docling":
        if _docling_loader is None:
            try:
                import core.pdf_loader_docling as _docling_loader_mod
                _docling_loader = _docling_loader_mod
            except Exception as e:
                print(f"⚠️ Docling import 실패: {e}")
        if _docling_loader is not None:
            _stream = _docling_loader.stream_pdf
        else:
            _stream = _fitz_loader.stream_pdf
    else:
        _stream = _fitz_loader.stream_pdf

    async def generate():
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            loop = asyncio.get_event_loop()
            gen  = _stream(tmp_path, filename, vision=vision,
                           cancel_check=_cancel_event.is_set,
                           page_start=page_start, page_end=page_end)
            # run_in_executor: 블로킹 next() 호출을 스레드에서 실행 → 이벤트 루프 비차단
            # next를 직접 넘기면 StopIteration이 asyncio Future로 전파되며 에러 발생(PEP 479)
            # → _SENTINEL로 감싸서 종료를 표현
            while True:
                event = await loop.run_in_executor(None, _next_or_sentinel, gen)
                if event is _SENTINEL:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/cancel")
async def cancel():
    """Vision 처리 취소"""
    _cancel_event.set()
    return {"ok": True}


@app.post("/index")
async def index_chunks(body: dict):
    """편집된 청크를 DB에 저장"""
    chunks = body.get("chunks", [])
    force  = body.get("force", False)
    collection = body.get("collection", vectordb.DEFAULT_COLLECTION)
    if not chunks:
        return {"ok": False, "message": "청크가 없습니다"}
    count = vectordb.insert(chunks, force=force, collection=collection)
    doc_context.generate_and_store(chunks, force=force)
    return {"ok": True, "message": f"{count}개 청크 색인 완료"}


@app.post("/chat")
async def chat(body: dict):
    session_id = body.get("session_id", "")
    query = body.get("query", "")
    history = vectordb.get_conversation_messages(session_id) if session_id else body.get("history", [])
    facts = [f["fact"] for f in vectordb.get_facts(session_id)] if session_id else []
    result = modular_rag.answer(query=query, history=history, facts=facts)
    if session_id:
        vectordb.save_turn(session_id, query, result["answer"])
    return result


@app.post("/chat-agent")
async def chat_agent(body: dict):
    session_id = body.get("session_id", "")
    query = body.get("query", "")
    history = vectordb.get_conversation_messages(session_id) if session_id else body.get("history", [])
    facts = [f["fact"] for f in vectordb.get_facts(session_id)] if session_id else []
    result = agentic_rag.answer(query=query, history=history, facts=facts)
    if session_id:
        vectordb.save_turn(session_id, query, result["answer"])
    return result


@app.post("/evaluate")
async def evaluate_answer(body: dict):
    return evaluate(
        query=body.get("query", ""),
        context=body.get("context", ""),
        answer=body.get("answer", ""),
    )


@app.get("/status")
async def status():
    return {"count": vectordb.count()}


@app.get("/db/sources")
async def db_sources():
    return vectordb.get_sources()


@app.get("/collections")
async def collections():
    """컬렉션(=MCP 서버) 별 문서 목록 + 등록된 MCP 서버 정보"""
    from mcp_servers.common import get_tool_specs

    available = []
    for cid, label in vectordb.COLLECTIONS.items():
        available.append({
            "id": cid,
            "label": label,
            "tools": await get_tool_specs(cid, label),
        })

    return {
        "collections": vectordb.get_collections(),
        "available": available,
    }


@app.get("/db/chunks")
async def db_chunks(source: str = None, limit: int = 50, offset: int = 0):
    return vectordb.get_chunks(source=source, limit=limit, offset=offset)


@app.delete("/db/chunk/{chunk_id}")
async def db_delete_chunk(chunk_id: int):
    vectordb.delete_chunk(chunk_id)
    return {"ok": True}


@app.delete("/db/all")
async def db_delete_all():
    vectordb.clear()
    return {"ok": True}


@app.get("/db/duplicates")
async def db_duplicates(threshold: float = 0.92):
    return vectordb.find_duplicates(threshold=threshold)


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    return vectordb.get_conversation_full(session_id)


@app.post("/conversation/summarize")
async def summarize_conversation(body: dict):
    session_id = body.get("session_id", "")
    generation = body.get("generation", 0)
    messages = vectordb.get_generation_messages(session_id, generation)
    if not messages:
        return {"summary": ""}
    summary = conversation.summarize_messages(messages)
    vectordb.summarize_generation(session_id, generation, summary)

    facts = conversation.extract_facts(messages)
    vectordb.save_facts(session_id, facts, generation)

    return {"summary": summary, "facts": facts}


@app.delete("/conversation/{session_id}")
async def delete_conversation(session_id: str):
    vectordb.clear_conversation(session_id)
    return {"ok": True}


@app.get("/memory/{session_id}")
async def get_memory(session_id: str):
    return vectordb.get_facts(session_id)


@app.delete("/memory/fact/{fact_id}")
async def delete_memory_fact(fact_id: int):
    vectordb.delete_fact(fact_id)
    return {"ok": True}


@app.get("/prompts")
async def get_prompts():
    return {
        "modular_rag": {
            "intent": INTENT_PROMPT,
            "query_rewrite": QUERY_REWRITE_PROMPT,
            "rerank": RERANK_PROMPT,
            "answer": RAG_SYSTEM_PROMPT,
        },
        "agentic_rag": {
            "agent_system": AGENT_SYSTEM,
        },
        "conversation": {
            "summary": conversation.SUMMARY_SYSTEM,
            "fact_extraction": conversation.FACT_EXTRACTION_SYSTEM,
        },
        "doc_context": {
            "current": doc_context.get_prompt("(자동 생성된 프롬프트 없음 — 문서 업로드 후 생성됨)"),
        },
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
