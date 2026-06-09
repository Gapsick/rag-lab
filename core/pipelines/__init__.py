"""
RAG 파이프라인 모음 (학습용)

pipelines/
  modular_rag.py  — 단계별 고정 파이프라인
                    Intent → Query Rewrite → Vector Search → Rerank → Answer
  agentic_rag.py  — LLM 자율 플래닝
                    Claude가 검색 전략·횟수를 스스로 결정
"""
from core.pipelines import modular_rag, agentic_rag

__all__ = ["modular_rag", "agentic_rag"]
