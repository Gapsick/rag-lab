"""
[오픈소스] PDF 로더 — Docling 기반 구현
=======================================================
Docling (IBM Research, 2024): https://github.com/DS4SD/docling

pdf_loader.py(학습용)와의 차이:
  - 우리가 직접 만든 _analyze_page(), _route() 로직 불필요
  - Docling 내부의 레이아웃 분석 ML 모델이 표/제목/본문을 자동 분류
  - 계층적 청킹: 같은 섹션의 내용을 묶어서 청크 생성
  - 표는 DataFrame → 마크다운 자동 변환

설치:
  pip install docling

같은 인터페이스(stream_pdf, load_pdf)를 유지하므로 server.py에서
  from core.pdf_loader_docling import stream_pdf, load_pdf
로 바꾸면 바로 교체 가능.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.chunking import HierarchicalChunker
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("⚠️  Docling 미설치. `pip install docling` 실행 필요.", file=sys.stderr)


# ── Docling 컨버터 초기화 ─────────────────────────────────────
# 최신 Docling API: format_options에 PdfFormatOption(pipeline_options=...) 으로 감싸야 함
# PdfPipelineOptions를 직접 넣으면 'backend' 속성 오류 발생
def _get_converter():
    options = PdfPipelineOptions()
    options.do_ocr = False              # OCR 끄기 (속도 향상, 텍스트 PDF에 충분)
    options.do_table_structure = False  # TableFormer 모델은 cv2/libxcb 필요 → Docker에서 미지원

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )


# ── 변환 + 청킹 ───────────────────────────────────────────────
# 의미 없는 장식 기호(≫, ·, → 등)만 단독으로 있는 청크 제거
NOISE_MIN_CHARS = 10


def _convert_to_chunks(pdf_path: str, filename: str) -> list[dict]:
    """Docling으로 PDF 변환 후 HierarchicalChunker로 청크 생성."""
    converter = _get_converter()

    print(f"  [Docling] {filename} 변환 중...", file=sys.stderr)
    result = converter.convert(source=pdf_path)
    doc    = result.document

    chunker = HierarchicalChunker()
    chunks  = []

    for chunk in chunker.chunk(doc):
        text = chunk.text.strip()

        # ≫, ·, → 같은 장식 기호만 있는 노이즈 청크 제거
        if len(text) < NOISE_MIN_CHARS:
            continue

        # headings: 상위→하위 전체 경로 ["1장 사업 개요", "1.1 추진 배경", ...]
        headings = getattr(chunk.meta, "headings", []) or []
        page_nos = set()
        for ref in getattr(chunk.meta, "doc_items", []):
            for prov in getattr(ref, "prov", []):
                page_nos.add(prov.page_no)

        page_str     = f"{min(page_nos)}페이지" if page_nos else "알 수 없음"
        heading_path = " > ".join(headings) if headings else ""

        # 섹션 경로를 content 앞에 붙여 임베딩이 문맥을 갖게 함
        content = f"[{heading_path}]\n{text}" if heading_path else text

        chunks.append({
            "content": content,
            "source":  f"{filename} - {page_str}" + (f" [{heading_path}]" if heading_path else ""),
        })
        print(f"  [Docling] 청크: {page_str} [{heading_path}] ({len(text)}자)",
              file=sys.stderr)

    return chunks


# ── 공개 API (pdf_loader.py와 동일한 인터페이스) ──────────────
def load_pdf(pdf_path: str, filename: str = None, vision: bool = False,
             cancel_check=None) -> list[dict]:
    """청크 리스트 반환"""
    if not DOCLING_AVAILABLE:
        raise RuntimeError("Docling 미설치. `pip install docling` 실행 필요.")
    name = filename or Path(pdf_path).name
    return _convert_to_chunks(pdf_path, name)


def stream_pdf(pdf_path: str, filename: str = None, vision: bool = False,
               cancel_check=None):
    """
    pdf_loader.py와 동일한 이벤트 형식으로 yield.
    Docling은 전체 PDF를 한 번에 처리하므로
    변환 완료 후 청크를 순서대로 스트리밍.

    이벤트 형식:
      {"type": "chunk", "chunk": {...}, "page": N, "total": N, "mode": "docling"}
      {"type": "done",  "total": N}
    """
    if not DOCLING_AVAILABLE:
        yield {"type": "done", "total": 0}
        return

    name   = filename or Path(pdf_path).name
    chunks = _convert_to_chunks(pdf_path, name)
    total  = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        if cancel_check and cancel_check():
            yield {"type": "cancelled", "processed": i - 1}
            return
        yield {
            "type":  "chunk",
            "chunk": chunk,
            "page":  i,       # Docling은 섹션 단위라 페이지 번호가 정확하지 않을 수 있음
            "total": total,
            "mode":  "docling",
        }

    yield {"type": "done", "total": total}


def load_pdfs_from_dir(pdf_dir: str, vision: bool = False) -> list[dict]:
    chunks = []
    for pdf_file in sorted(Path(pdf_dir).glob("*.pdf")):
        chunks.extend(load_pdf(str(pdf_file)))
        print(f"  로드: {pdf_file.name}")
    return chunks
