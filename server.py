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
from core.pipelines import modular_rag, agentic_rag
from core.evaluator import evaluate
from core.pdf_loader import load_pdf, stream_pdf
import core.pdf_loader as _fitz_loader
try:
    import core.pdf_loader_docling as _docling_loader
    _DOCLING_OK = True
except Exception:
    _DOCLING_OK = False

load_dotenv(Path(__file__).parent / ".env")


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
async def upload(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        chunks = load_pdf(tmp_path, filename=file.filename)
        vectordb.insert(chunks)
        doc_context.generate_and_store(chunks)
        return {"ok": True, "message": f"{file.filename} — {len(chunks)}페이지 색인 완료"}
    finally:
        os.unlink(tmp_path)


@app.post("/preview-stream")
async def preview_stream(
    file: UploadFile = File(...),
    vision: bool = Form(False),
    loader: str = Form("fitz"),
):
    """청크 실시간 스트리밍 - 페이지 처리될 때마다 SSE로 전송
    loader: "fitz" (기본, 학습용) | "docling" (오픈소스 ML)
    """
    _cancel_event.clear()
    data = await file.read()
    filename = file.filename

    # loader 파라미터로 사용할 stream_pdf 함수 선택
    if loader == "docling" and _DOCLING_OK:
        _stream = _docling_loader.stream_pdf
    else:
        _stream = _fitz_loader.stream_pdf

    async def generate():
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            loop = asyncio.get_event_loop()
            gen  = _stream(tmp_path, filename, vision=vision,
                           cancel_check=_cancel_event.is_set)
            # run_in_executor: 블로킹 next() 호출을 스레드에서 실행 → 이벤트 루프 비차단
            while True:
                try:
                    event = await loop.run_in_executor(None, next, gen)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except StopIteration:
                    break
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
    if not chunks:
        return {"ok": False, "message": "청크가 없습니다"}
    count = vectordb.insert(chunks, force=force)
    doc_context.generate_and_store(chunks, force=force)
    return {"ok": True, "message": f"{count}개 청크 색인 완료"}


@app.post("/chat")
async def chat(body: dict):
    return modular_rag.answer(
        query=body.get("query", ""),
        history=body.get("history", []),
    )


@app.post("/chat-agent")
async def chat_agent(body: dict):
    return agentic_rag.answer(
        query=body.get("query", ""),
        history=body.get("history", []),
    )


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


@app.get("/db/chunks")
async def db_chunks(source: str = None, limit: int = 50, offset: int = 0):
    return vectordb.get_chunks(source=source, limit=limit, offset=offset)


@app.delete("/db/chunk/{chunk_id}")
async def db_delete_chunk(chunk_id: int):
    vectordb.delete_chunk(chunk_id)
    return {"ok": True}


@app.get("/db/duplicates")
async def db_duplicates(threshold: float = 0.92):
    return vectordb.find_duplicates(threshold=threshold)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
