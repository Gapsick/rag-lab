"""
VectorDB 모듈 - PostgreSQL + pgvector

실행 전: docker compose up -d
"""

import os
import sys
import time
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "database": "handoutdb",
    "user": "postgres",
    "password": "postgres",
}
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_DIM = 384

print("임베딩 모델 로딩 중...", file=sys.stderr)
embedder = SentenceTransformer(EMBED_MODEL)
print("임베딩 모델 로딩 완료", file=sys.stderr)


def _raw_conn():
    return psycopg2.connect(**DB_CONFIG)


def _vec_conn():
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def wait_for_db(retries: int = 15, delay: float = 2.0):
    for i in range(retries):
        try:
            _raw_conn().close()
            return
        except Exception:
            print(f"[DB] 대기 중... ({i + 1}/{retries})", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("PostgreSQL 연결 실패. `docker compose up -d` 확인")


def setup():
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id        SERIAL PRIMARY KEY,
            content   TEXT NOT NULL,
            source    TEXT,
            embedding vector({EMBED_DIM})
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def count() -> int:
    try:
        conn = _raw_conn()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM documents")
        n = cur.fetchone()[0]
        cur.close()
        conn.close()
        return n
    except Exception:
        return 0


def insert(chunks: list[dict], force: bool = False) -> int:
    """
    chunks: [{"content": ..., "source": ...}, ...]
    force=True: 기존 데이터 전부 삭제 후 재삽입
    force=False: 기존 데이터에 추가
    """
    conn = _vec_conn()
    cur = conn.cursor()

    if force:
        cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")
        conn.commit()

    texts = [c["content"] for c in chunks]
    print(f"[DB] {len(texts)}개 청크 임베딩 생성 중...", file=sys.stderr)
    embeddings = embedder.encode(texts, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        cur.execute(
            "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
            (chunk["content"], chunk.get("source", ""), emb.tolist()),
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] {len(chunks)}개 청크 삽입 완료", file=sys.stderr)
    return len(chunks)


def clear():
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE documents RESTART IDENTITY")
    conn.commit()
    cur.close()
    conn.close()


def get_sources() -> list[dict]:
    """저장된 파일별 청크 수 반환"""
    try:
        conn = _raw_conn()
        cur = conn.cursor()
        cur.execute("SELECT source, COUNT(*) FROM documents GROUP BY source ORDER BY source")
        rows = cur.fetchall()
        cur.close(); conn.close()
        return [{"source": r[0], "count": r[1]} for r in rows]
    except Exception:
        return []


def get_chunks(source: str = None, limit: int = 50, offset: int = 0) -> list[dict]:
    """청크 목록 반환 (source 필터 가능)"""
    conn = _raw_conn()
    cur = conn.cursor()
    if source:
        cur.execute(
            "SELECT id, content, source FROM documents WHERE source LIKE %s ORDER BY id LIMIT %s OFFSET %s",
            (source + '%', limit, offset)
        )
    else:
        cur.execute(
            "SELECT id, content, source FROM documents ORDER BY id LIMIT %s OFFSET %s",
            (limit, offset)
        )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [{"id": r[0], "content": r[1], "source": r[2]} for r in rows]


def find_duplicates(threshold: float = 0.92, limit: int = 50) -> list[dict]:
    """유사도 threshold 이상인 청크 쌍 반환"""
    conn = _vec_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.id, a.content, a.source,
               b.id, b.content, b.source,
               1 - (a.embedding <=> b.embedding) AS similarity
        FROM   documents a
        JOIN   documents b ON a.id < b.id
        WHERE  1 - (a.embedding <=> b.embedding) > %s
        ORDER  BY similarity DESC
        LIMIT  %s
    """, (threshold, limit))
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [
        {
            "a": {"id": r[0], "content": r[1], "source": r[2]},
            "b": {"id": r[3], "content": r[4], "source": r[5]},
            "similarity": round(float(r[6]), 3),
        }
        for r in rows
    ]


def delete_chunk(chunk_id: int):
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM documents WHERE id = %s", (chunk_id,))
    conn.commit()
    cur.close(); conn.close()


def search(query: str, top_k: int = 3) -> list[tuple[str, float, str]]:
    """반환: [(content, similarity, source), ...]"""
    query_emb = embedder.encode([query])[0].tolist()

    conn = _vec_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT content,
               1 - (embedding <=> %s::vector) AS similarity,
               source
        FROM   documents
        ORDER  BY embedding <=> %s::vector
        LIMIT  %s
        """,
        (query_emb, query_emb, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [(content, float(sim), source) for content, sim, source in rows]
