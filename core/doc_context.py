"""
업로드된 문서를 분석해 시스템 프롬프트를 자동 생성하고 저장하는 모듈.
PDF가 추가될 때마다 전체 문서 목록을 기반으로 프롬프트를 재생성한다.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(Path(__file__).parent.parent / ".env")
client = Anthropic()

_system_prompt: str | None = None
_doc_samples: dict[str, str] = {}  # {문서명: 샘플 텍스트}

_META_PROMPT = (
    "사용자가 제공하는 하나 이상의 문서 샘플을 읽고, 이 문서들에 대한 질문에 답하는 AI 도우미를 위한 시스템 프롬프트를 한 문단으로 작성하세요.\n"
    "포함할 내용:\n"
    "1. 문서들의 종류와 목적에 맞는 도우미 역할 정의\n"
    "2. 예상 사용자가 누구인지 반영한 답변 스타일\n"
    "3. 마지막에 반드시 이 문장 포함: '자료에 없는 내용은 해당 자료에 내용이 없습니다라고 솔직히 말하세요.'\n"
    "시스템 프롬프트 텍스트만 출력하세요. 설명이나 따옴표 없이."
)


def _extract_doc_name(chunks: list[dict]) -> str:
    """청크의 source 필드에서 파일명을 추출한다."""
    if not chunks:
        return "unknown"
    source = chunks[0].get("source", "")
    # "파일명.pdf - 1페이지" 형태에서 파일명 부분만 추출
    return source.split(" - ")[0] if " - " in source else source


def _regenerate_prompt() -> str:
    """저장된 모든 문서 샘플로 프롬프트를 재생성한다."""
    global _system_prompt

    sections = [
        f"[문서: {name}]\n{sample}"
        for name, sample in _doc_samples.items()
    ]
    combined = "\n\n===\n\n".join(sections)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=_META_PROMPT,
        messages=[{
            "role": "user",
            "content": f"다음은 업로드된 문서들의 샘플입니다:\n\n{combined}"
        }],
    )

    _system_prompt = resp.content[0].text.strip()

    print("\n" + "=" * 60, file=sys.stderr)
    print(f"[자동 생성 시스템 프롬프트] (문서 {len(_doc_samples)}개 기준)", file=sys.stderr)
    for name in _doc_samples:
        print(f"  - {name}", file=sys.stderr)
    print(_system_prompt, file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    return _system_prompt


def generate_and_store(chunks: list[dict], force: bool = False) -> str:
    """
    청크에서 문서 샘플을 추출해 누적한 뒤 프롬프트를 재생성한다.
    force=True면 기존 문서 샘플을 모두 지우고 새로 시작한다.
    """
    global _doc_samples

    if force:
        _doc_samples.clear()

    doc_name = _extract_doc_name(chunks)
    _doc_samples[doc_name] = "\n\n".join(c["content"][:300] for c in chunks[:3])

    return _regenerate_prompt()


def get_prompt(fallback: str) -> str:
    """저장된 프롬프트가 있으면 반환, 없으면 fallback 사용."""
    return _system_prompt or fallback
