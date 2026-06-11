"""
[학습용] PDF 로더 — fitz(PyMuPDF) 기반 직접 구현
=======================================================
실제 프로덕션에서는 Docling, Unstructured 같은 라이브러리를 씁니다.
이 파일은 "내부에서 어떤 일이 벌어지는지" 이해하기 위한 학습용 구현입니다.

참고: 오픈소스 라이브러리 버전 → core/pdf_loader_docling.py

흐름:
  PDF 파일
    → _analyze_page()  : 페이지 구성 분석 (이미지/도형/표 감지)
    → _route()         : 어떤 Agent로 보낼지 결정
    → _process_page()  : 실제 추출 실행
    → stream_pdf()     : 여러 페이지를 병렬 처리 + SSE 스트리밍

3가지 Agent:
  [Table]  : fitz.find_tables()  → 마크다운 표 (API 호출 없음, 무료)
  [Text]   : fitz.get_text()     → 텍스트 추출  (API 호출 없음, 무료)
  [Vision] : Claude Vision API   → 이미지/흐름도 설명 (API 호출, 유료)
"""

import base64
import concurrent.futures
import sys
from pathlib import Path

from dotenv import load_dotenv
import fitz                 # PyMuPDF: PDF를 직접 파싱하는 C++ 기반 라이브러리
from anthropic import Anthropic

load_dotenv(Path(__file__).parent.parent / ".env")
client = Anthropic()

from prompts.vision import VISION_PROMPT


# ── 설정 ──────────────────────────────────────────────────────
VISION_MAX_WORKERS = 4   # Vision API를 동시에 최대 몇 페이지까지 처리할지
                         # 너무 높이면 Anthropic rate limit에 걸릴 수 있음
IMG_AREA_RATIO     = 0.10 # 래스터 이미지가 페이지 면적의 10% 이상이면 Vision으로 분류
BOX_MIN_AREA_RATIO = 0.01 # 이 면적(페이지의 1%) 이상인 채워진 도형만 '의미있는 도형'으로 봄
                          # → 작은 글머리 기호, 밑줄 등 장식 요소 제외용
BOX_MIN_COUNT      = 3   # 의미있는 도형이 3개 이상 + 텍스트 부족 → 흐름도로 판단
SPARSE_TEXT_LEN    = 200  # 텍스트가 이 길이보다 짧으면 "텍스트가 부족한 페이지"로 봄
TABLE_QUALITY_THRESHOLD = 0.5  # 표 셀의 50% 이상이 비어있으면 품질 불량 → Vision 폴백

CHUNK_SIZE    = 400  # text 모드에서 child 청크 목표 크기 (글자 수)
CHUNK_OVERLAP = 80   # 인접 청크 간 겹치는 글자 수 (경계에서 문맥 끊김 완화)


# ── Recursive Text Splitter (parent-child용 child 생성) ────────
def _split_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    문단(\\n\\n) → 줄(\\n) → 문장(. ) → 글자 순으로 자연스러운 경계를 찾아
    chunk_size 근처에서 분할. PDF 종류에 상관없이 동작하는 범용 분할.

    overlap: 인접 청크끼리 끝-시작 부분을 겹쳐서, 경계에서 문맥이
             잘려도 한쪽 청크에는 온전히 남도록 함.

    반환: [{"text": 청크 본문(오버랩 포함), "overlap": 앞에 붙은 오버랩 글자 수}, ...]
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [{"text": text, "overlap": 0}] if text else []

    def recursive(s: str, seps: list[str]) -> list[str]:
        if len(s) <= chunk_size:
            return [s]
        if not seps:
            # 더 쪼갤 구분자가 없으면 글자 수로 강제 분할
            return [s[i:i + chunk_size] for i in range(0, len(s), chunk_size)]

        sep, *rest = seps
        parts = s.split(sep) if sep else list(s)
        if len(parts) == 1:
            return recursive(s, rest)

        chunks, buf = [], ""
        for part in parts:
            piece = part + sep
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf:
                    chunks.append(buf)
                if len(piece) > chunk_size:
                    chunks.extend(recursive(piece, rest))
                    buf = ""
                else:
                    buf = piece
        if buf:
            chunks.append(buf)
        return chunks

    raw = recursive(text, ["\n\n", "\n", ". ", " "])

    # overlap 적용: 이전 청크의 끝부분을 다음 청크 앞에 붙임
    result = []
    for i, chunk in enumerate(raw):
        overlap_len = 0
        if i > 0 and overlap > 0:
            tail        = raw[i - 1][-overlap:]
            overlap_len = len(tail.strip())
            chunk       = tail + chunk
        chunk = chunk.strip()
        if chunk:
            result.append({"text": chunk, "overlap": min(overlap_len, len(chunk))})

    return result


# ── STEP 1: 페이지 분석 ───────────────────────────────────────
def _analyze_page(pdf_path: str, page_index: int) -> dict:
    """
    한 페이지의 구성 요소를 분석해서 딕셔너리로 반환.
    이 정보를 _route()가 보고 어떤 Agent로 보낼지 결정함.

    반환값:
      text      : 페이지에서 추출된 원본 텍스트
      img_ratio : 래스터 이미지(JPG/PNG 등)가 차지하는 면적 비율 (0~1)
      box_count : 채워진 벡터 도형(흐름도 박스 등) 개수
      tables    : fitz가 감지한 표 목록
    """
    doc  = fitz.open(pdf_path)
    page = doc[page_index]

    # 텍스트 추출 — 레이아웃 정보 없이 순수 텍스트만
    text      = page.get_text().strip()

    # 래스터 이미지 목록 — JPG, PNG처럼 픽셀 기반 이미지
    # 흐름도처럼 PDF 도형으로 그린 것들은 여기서 안 잡힘 (→ box_count에서 잡음)
    images    = page.get_images()
    page_area = page.rect.width * page.rect.height

    img_ratio = 0.0
    if images:
        # 각 이미지의 bounding box로 면적 계산
        img_area = sum(
            (i["bbox"][2] - i["bbox"][0]) * (i["bbox"][3] - i["bbox"][1])
            for i in page.get_image_info()
        )
        img_ratio = img_area / page_area if page_area > 0 else 0

    # 벡터 도형 — PDF에서 선, 사각형, 원 등으로 직접 그린 것들
    # 흐름도, 다이어그램의 박스들이 여기에 해당
    # 단순 장식(밑줄, 테두리)은 크기 조건으로 필터링
    drawings  = page.get_drawings()
    box_count = sum(
        1 for d in drawings
        if d.get("fill") is not None           # 속이 채워진 도형만 (테두리만 있는 건 제외)
        and d["rect"].width > 40               # 최소 너비 40pt 이상
        and d["rect"].height > 15              # 최소 높이 15pt 이상
        and (d["rect"].width * d["rect"].height) > page_area * BOX_MIN_AREA_RATIO
    )

    # 표 감지 — fitz가 격자 구조를 분석해서 표 경계를 자동으로 찾아줌
    try:
        tables    = page.find_tables()
        table_list = tables.tables if tables else []
    except Exception:
        table_list = []

    doc.close()
    return {
        "text": text,
        "img_ratio": img_ratio,
        "box_count": box_count,
        "tables": table_list,
    }


# ── STEP 2: 라우팅 결정 ───────────────────────────────────────
def _route(info: dict) -> str:
    """
    분석 결과를 보고 어떤 Agent로 처리할지 결정.
    우선순위: table > vision > text

    table  : fitz가 표를 감지했을 때 — 무료이고 정확하므로 Vision보다 우선
    vision : 이미지가 크거나 흐름도처럼 시각적 요소가 중요할 때
    text   : 그 외 일반 텍스트 페이지
    """
    # 표가 있으면 Table Agent 우선 (API 호출 없이 정확하게 처리 가능)
    if info["tables"]:
        return "table"

    # 래스터 이미지가 페이지의 10% 이상 → Vision (사진, 그래프 등)
    if info["img_ratio"] >= IMG_AREA_RATIO:
        return "vision"

    # 텍스트가 별로 없는데 박스 도형이 많음 → 흐름도/다이어그램 → Vision
    if len(info["text"]) < SPARSE_TEXT_LEN and info["box_count"] >= BOX_MIN_COUNT:
        return "vision"

    # 그 외 → 텍스트 직접 추출 (가장 빠르고 무료)
    return "text"


# ── Table Agent ───────────────────────────────────────────────
def _table_quality(rows: list) -> float:
    """
    표 추출 품질 점수 반환 (0~1).
    병합 셀이 많으면 fitz가 빈 칸으로 처리해서 품질이 낮아짐.
    품질이 낮으면 Vision으로 폴백해서 더 정확하게 처리.
    """
    if not rows:
        return 0.0
    total  = sum(len(r) for r in rows)
    if total == 0:
        return 0.0
    filled = sum(1 for r in rows for c in r if c and str(c).strip())
    return filled / total


def _extract_tables_md(pdf_path: str, page_index: int, text: str) -> str | None:
    """
    fitz find_tables()로 표를 마크다운으로 변환.
    품질이 TABLE_QUALITY_THRESHOLD 미만이면 None 반환 → 호출부에서 Vision 폴백.

    마크다운 표 형식:
    | 컬럼1 | 컬럼2 |
    |------|------|
    | 값1  | 값2  |
    """
    doc    = fitz.open(pdf_path)
    page   = doc[page_index]
    tables = page.find_tables()
    doc.close()

    parts = []
    if text:
        parts.append(text)   # 표 위에 있는 제목/설명 텍스트도 포함

    for tbl in (tables.tables or []):
        rows = tbl.extract()  # [[행1셀1, 행1셀2], [행2셀1, 행2셀2], ...]
        if not rows:
            continue

        # 행이 1개 이하이거나 열이 1개 이하면 섹션 헤더/장식 박스 → 표 아님
        if len(rows) <= 1 or len(rows[0]) <= 1:
            continue

        # 병합 셀이 많으면 품질 불량 → Vision으로 넘김
        if _table_quality(rows) < TABLE_QUALITY_THRESHOLD:
            return None

        header = rows[0]
        md  = "| " + " | ".join(_clean_cell(c) for c in header) + " |\n"
        md += "| " + " | ".join("---" for _ in header) + " |\n"
        for row in rows[1:]:
            md += "| " + " | ".join(_clean_cell(c) for c in row) + " |\n"
        parts.append(md)

    return "\n\n".join(parts) if parts else None


def _clean_cell(val) -> str:
    """셀 값 정리 — None 처리, 줄바꿈 제거, 마크다운 파이프 문자 이스케이프"""
    if val is None:
        return ""
    return str(val).replace("\n", " ").replace("|", "\\|").strip()


# ── Vision Agent ──────────────────────────────────────────────
def _page_to_image(pdf_path: str, page_index: int) -> str:
    """
    PDF 페이지를 PNG 이미지로 렌더링 후 base64 인코딩.
    Matrix(2, 2) = 2배 해상도로 렌더링 → Vision 모델이 더 잘 읽음
    (해상도 높이면 토큰도 늘어남 — 트레이드오프)
    """
    doc  = fitz.open(pdf_path)
    page = doc[page_index]
    pix  = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    data = pix.tobytes("png")
    doc.close()
    return base64.standard_b64encode(data).decode()


def _describe_page(pdf_path: str, page_index: int) -> str:
    """
    Claude Vision API로 페이지 이미지를 텍스트 설명으로 변환.
    표/흐름도/그래프를 구조적으로 설명하도록 VISION_PROMPT에 지시.
    """
    img_b64 = _page_to_image(pdf_path, page_index)
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    )
    return resp.content[0].text


# ── STEP 3: 한 페이지 처리 ────────────────────────────────────
def _process_page(pdf_path: str, name: str, page_index: int, cancel_check) -> dict | None:
    """
    분석 → 라우팅 → 추출을 한 페이지에 대해 실행.
    ThreadPoolExecutor에서 병렬로 호출되므로 thread-safe하게 작성.
    (fitz.open()을 매번 새로 여는 이유: fitz Document는 thread-safe하지 않음)
    """
    if cancel_check and cancel_check():
        return None

    page_num = page_index + 1
    info     = _analyze_page(pdf_path, page_index)
    mode     = _route(info)

    print(f"  [{mode.upper():6}] {page_num}페이지", file=sys.stderr)

    try:
        if mode == "table":
            content = _extract_tables_md(pdf_path, page_index, info["text"])
            if content is None:
                # 품질 불량 → Vision으로 폴백
                print(f"  [TABLE→VISION] {page_num}페이지 (병합셀 감지)", file=sys.stderr)
                mode    = "vision"
                content = _describe_page(pdf_path, page_index)
                source  = f"{name} - {page_num}페이지 (Vision)"
                # parent == content (Vision 설명 전체 보존), child는 분할해 검색 정밀도 확보
                chunks = [
                    {"content": part["text"], "overlap": part["overlap"],
                     "source": source, "parent_content": content}
                    for part in _split_text(content)
                ]
            else:
                source  = f"{name} - {page_num}페이지 (표)"
                # parent == content (텍스트+표 전체 보존), child는 분할해 검색 정밀도 확보
                chunks = [
                    {"content": part["text"], "overlap": part["overlap"],
                     "source": source, "parent_content": content}
                    for part in _split_text(content)
                ]

        elif mode == "vision":
            content = _describe_page(pdf_path, page_index)
            if content.strip() == "표지":
                return None  # 표지 페이지는 색인 불필요
            source = f"{name} - {page_num}페이지 (Vision)"
            # parent == content (Vision 설명 전체 보존), child는 분할해 검색 정밀도 확보
            chunks = [
                {"content": part["text"], "overlap": part["overlap"],
                 "source": source, "parent_content": content}
                for part in _split_text(content)
            ]

        else:  # text
            doc     = fitz.open(pdf_path)
            content = doc[page_index].get_text().strip()
            doc.close()
            if not content:
                return None
            source = f"{name} - {page_num}페이지"
            # child: 검색용 작은 조각 / parent: 페이지 전체 (답변 컨텍스트용)
            chunks = [
                {"content": part["text"], "overlap": part["overlap"],
                 "source": source, "parent_content": content}
                for part in _split_text(content)
            ]

        return {"page_num": page_num, "mode": mode, "chunks": chunks}

    except Exception as e:
        print(f"  ⚠️ {page_num}페이지 {mode} 실패: {e}", file=sys.stderr)
        # 최후 폴백: 텍스트 직접 추출
        if info["text"]:
            source = f"{name} - {page_num}페이지"
            chunks = [
                {"content": part["text"], "overlap": part["overlap"],
                 "source": source, "parent_content": info["text"]}
                for part in _split_text(info["text"])
            ]
            return {"page_num": page_num, "mode": "text", "chunks": chunks}
        return None


# ── 공개 API ──────────────────────────────────────────────────
def load_pdf(pdf_path: str, filename: str = None, vision: bool = False,
             cancel_check=None, page_start: int = None, page_end: int = None) -> list[dict]:
    """청크 리스트 반환 (스트리밍 불필요 시 사용)"""
    return [e["chunk"] for e in stream_pdf(pdf_path, filename, vision, cancel_check, page_start, page_end)
            if e["type"] == "chunk"]


def stream_pdf(pdf_path: str, filename: str = None, vision: bool = False,
               cancel_check=None, page_start: int = None, page_end: int = None):
    """
    페이지 처리될 때마다 SSE 이벤트를 yield하는 제너레이터.

    vision=False : 텍스트만 순차 추출 (API 호출 없음, 빠름)
    vision=True  : 3방향 자동 라우팅 + ThreadPoolExecutor 병렬 처리

    page_start/page_end: 1-indexed, 둘 다 포함(inclusive). None이면 전체 페이지.

    이벤트 형식:
      {"type": "chunk",     "chunk": {content, source}, "page": N, "total": N, "mode": ...}
      {"type": "cancelled", "processed": N}
      {"type": "done",      "total": N}
    """
    name = filename or Path(pdf_path).name
    doc  = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    # 페이지 범위 결정 (1-indexed → 0-indexed)
    start = (page_start - 1) if page_start else 0
    end   = page_end if page_end else total_pages
    page_indices = [i for i in range(total_pages) if start <= i < end]

    # ── 텍스트 전용 모드 (Vision API 호출 없음) ──────────────────
    if not vision:
        processed = 0
        for i in page_indices:
            if cancel_check and cancel_check():
                yield {"type": "cancelled", "processed": processed}
                return
            doc     = fitz.open(pdf_path)
            content = doc[i].get_text().strip()
            doc.close()
            if not content:
                continue
            source = f"{name} - {i+1}페이지"
            # child: 검색용 작은 조각 / parent: 페이지 전체 (답변 컨텍스트용)
            for part in _split_text(content):
                processed += 1
                yield {"type": "chunk",
                       "chunk": {"content": part["text"], "overlap": part["overlap"],
                                 "source": source, "parent_content": content},
                       "page": i + 1, "total": len(page_indices), "mode": "text"}
        yield {"type": "done", "total": processed}
        return

    # ── 이미지 분석 포함 모드 ─────────────────────────────────────
    processed = 0
    # ThreadPoolExecutor: I/O 바운드 작업(Vision API 호출)을 병렬 실행
    # max_workers=4 → 동시에 4페이지까지 API 요청 (rate limit 대비)
    # as_completed: 먼저 완료된 페이지부터 yield → 실시간 스트리밍 효과
    with concurrent.futures.ThreadPoolExecutor(max_workers=VISION_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_process_page, pdf_path, name, i, cancel_check): i
            for i in page_indices
        }
        for future in concurrent.futures.as_completed(future_map):
            if cancel_check and cancel_check():
                executor.shutdown(wait=False, cancel_futures=True)
                yield {"type": "cancelled", "processed": processed}
                return
            result = future.result()
            if result:
                for chunk in result["chunks"]:
                    processed += 1
                    yield {"type": "chunk", "chunk": chunk,
                           "page": result["page_num"], "total": len(page_indices),
                           "mode": result["mode"]}

    yield {"type": "done", "total": processed}


def load_pdfs_from_dir(pdf_dir: str, vision: bool = False) -> list[dict]:
    """디렉토리 내 모든 PDF 처리"""
    chunks = []
    for pdf_file in sorted(Path(pdf_dir).glob("*.pdf")):
        chunks.extend(load_pdf(str(pdf_file), vision=vision))
        print(f"  로드: {pdf_file.name}")
    return chunks
