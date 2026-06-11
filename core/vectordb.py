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
CONV_WINDOW_SIZE = 5  # 한 세대 = user+assistant 5쌍

# 컬렉션(=MCP 서버) 정의: id → 표시 이름
COLLECTIONS = {
    "lecture": "강의자료",
    "notice":  "공지·문서",
}
DEFAULT_COLLECTION = "lecture"

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
    # parent-child 청킹: parent_content = 답변 생성용 전체 컨텍스트 (페이지 전체 등)
    cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS parent_content TEXT")
    # 컬렉션(=MCP 서버) 단위 분리: lecture(강의자료), notice(공지·문서) 등
    cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS collection TEXT NOT NULL DEFAULT 'default'")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            generation INT NOT NULL,
            is_summary BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT now()
        )
    """)
    # 장기 메모리: 세대 요약 시 추출된 사용자 관련 fact
    cur.execute("""
        CREATE TABLE IF NOT EXISTS memory_facts (
            id         SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            fact       TEXT NOT NULL,
            generation INT,
            created_at TIMESTAMP DEFAULT now()
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


def insert(chunks: list[dict], force: bool = False, collection: str = DEFAULT_COLLECTION) -> int:
    """
    chunks: [{"content": ..., "source": ...}, ...]
    force=True: 같은 collection의 기존 데이터 삭제 후 재삽입
    force=False: 기존 데이터에 추가
    """
    conn = _vec_conn()
    cur = conn.cursor()

    if force:
        cur.execute("DELETE FROM documents WHERE collection = %s", (collection,))
        conn.commit()

    texts = [c["content"] for c in chunks]
    print(f"[DB] {len(texts)}개 청크 임베딩 생성 중...", file=sys.stderr)
    embeddings = embedder.encode(texts, show_progress_bar=True)

    for chunk, emb in zip(chunks, embeddings):
        cur.execute(
            "INSERT INTO documents (content, source, embedding, parent_content, collection) VALUES (%s, %s, %s, %s, %s)",
            (chunk["content"], chunk.get("source", ""), emb.tolist(),
             chunk.get("parent_content", chunk["content"]), collection),
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


def get_sources(collection: str | None = None) -> list[dict]:
    """저장된 파일별 청크 수 반환 (collection 지정 시 해당 컬렉션만)"""
    try:
        conn = _raw_conn()
        cur = conn.cursor()
        if collection:
            cur.execute(
                "SELECT source, COUNT(*) FROM documents WHERE collection = %s GROUP BY source ORDER BY source",
                (collection,)
            )
        else:
            cur.execute("SELECT source, COUNT(*) FROM documents GROUP BY source ORDER BY source")
        rows = cur.fetchall()
        cur.close(); conn.close()
        return [{"source": r[0], "count": r[1]} for r in rows]
    except Exception:
        return []


def get_collections() -> list[dict]:
    """컬렉션(=MCP 서버) 별 문서 목록과 청크 수 반환"""
    try:
        conn = _raw_conn()
        cur = conn.cursor()
        cur.execute("SELECT collection, source, COUNT(*) FROM documents GROUP BY collection, source ORDER BY collection, source")
        rows = cur.fetchall()
        cur.close(); conn.close()
    except Exception:
        return []

    result: dict[str, dict] = {}
    for collection, source, count in rows:
        doc_name = source.split(" - ")[0] if " - " in source else source
        entry = result.setdefault(collection, {
            "collection": collection,
            "label": COLLECTIONS.get(collection, collection),
            "documents": {},
        })
        entry["documents"][doc_name] = entry["documents"].get(doc_name, 0) + count

    return [
        {
            "collection": v["collection"],
            "label": v["label"],
            "documents": [{"name": name, "count": c} for name, c in v["documents"].items()],
            "chunk_count": sum(v["documents"].values()),
        }
        for v in result.values()
    ]


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


def search(query: str, top_k: int = 3, source_prefix: str | None = None,
           collection: str | None = None) -> list[tuple[str, float, str, str]]:
    """반환: [(content, similarity, source, parent_content), ...]
    source_prefix: 지정 시 source가 해당 접두사로 시작하는 문서로만 검색 범위 제한
    collection: 지정 시 해당 컬렉션(=MCP 서버)에 속한 문서로만 검색 범위 제한
    """
    query_emb = embedder.encode([query])[0].tolist()

    conditions, params = [], []
    if source_prefix:
        conditions.append("source LIKE %s")
        params.append(source_prefix + "%")
    if collection:
        conditions.append("collection = %s")
        params.append(collection)
    where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    conn = _vec_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT content,
               1 - (embedding <=> %s::vector) AS similarity,
               source,
               parent_content
        FROM   documents
        {where_clause}
        ORDER  BY embedding <=> %s::vector
        LIMIT  %s
        """,
        [query_emb, *params, query_emb, top_k],
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [(content, float(sim), source, parent_content) for content, sim, source, parent_content in rows]


# ── 대화 이력 (Context Window) ──────────────────────────────────
def get_conversation_messages(session_id: str) -> list[dict]:
    """Claude API용 history: [{"role":..., "content":...}, ...]"""
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM conversations WHERE session_id = %s ORDER BY id",
        (session_id,)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def get_conversation_full(session_id: str) -> list[dict]:
    """UI 패널용: id/role/content/generation/is_summary 포함"""
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, role, content, generation, is_summary FROM conversations "
        "WHERE session_id = %s ORDER BY id",
        (session_id,)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [
        {"id": r[0], "role": r[1], "content": r[2], "generation": r[3], "is_summary": r[4]}
        for r in rows
    ]


def get_generation_messages(session_id: str, generation: int) -> list[dict]:
    """특정 세대의 raw 메시지 (요약 대상)"""
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM conversations "
        "WHERE session_id = %s AND generation = %s ORDER BY id",
        (session_id, generation)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def save_turn(session_id: str, user_msg: str, assistant_msg: str):
    """user+assistant 한 턴을 현재 세대에 저장"""
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT generation, is_summary FROM conversations "
        "WHERE session_id = %s ORDER BY id DESC LIMIT 1",
        (session_id,)
    )
    row = cur.fetchone()
    if row is None:
        generation = 0
    else:
        last_gen, last_is_summary = row
        if last_is_summary:
            generation = last_gen + 1
        else:
            cur.execute(
                "SELECT COUNT(*) FROM conversations "
                "WHERE session_id = %s AND generation = %s AND role = 'user'",
                (session_id, last_gen)
            )
            user_count = cur.fetchone()[0]
            generation = last_gen + 1 if user_count >= CONV_WINDOW_SIZE else last_gen
    cur.execute(
        "INSERT INTO conversations (session_id, role, content, generation) VALUES (%s, %s, %s, %s)",
        (session_id, "user", user_msg, generation)
    )
    cur.execute(
        "INSERT INTO conversations (session_id, role, content, generation) VALUES (%s, %s, %s, %s)",
        (session_id, "assistant", assistant_msg, generation)
    )
    conn.commit()
    cur.close(); conn.close()


def summarize_generation(session_id: str, generation: int, summary_text: str):
    """해당 세대를 (이전 대화 요약) user/assistant 한 쌍으로 교체"""
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM conversations WHERE session_id = %s AND generation = %s",
        (session_id, generation)
    )
    cur.execute(
        "INSERT INTO conversations (session_id, role, content, generation, is_summary) "
        "VALUES (%s, 'user', %s, %s, TRUE)",
        (session_id, "(이전 대화 요약)", generation)
    )
    cur.execute(
        "INSERT INTO conversations (session_id, role, content, generation, is_summary) "
        "VALUES (%s, 'assistant', %s, %s, TRUE)",
        (session_id, summary_text, generation)
    )
    conn.commit()
    cur.close(); conn.close()


def clear_conversation(session_id: str):
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM conversations WHERE session_id = %s", (session_id,))
    conn.commit()
    cur.close(); conn.close()


# ── 장기 메모리 (memory_facts) ──────────────────────────────────
def save_facts(session_id: str, facts: list[str], generation: int):
    if not facts:
        return
    conn = _raw_conn()
    cur = conn.cursor()
    for fact in facts:
        cur.execute(
            "INSERT INTO memory_facts (session_id, fact, generation) VALUES (%s, %s, %s)",
            (session_id, fact, generation)
        )
    conn.commit()
    cur.close(); conn.close()


def get_facts(session_id: str) -> list[dict]:
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, fact, generation, created_at FROM memory_facts "
        "WHERE session_id = %s ORDER BY id",
        (session_id,)
    )
    rows = cur.fetchall()
    cur.close(); conn.close()
    return [
        {"id": r[0], "fact": r[1], "generation": r[2], "created_at": r[3].isoformat()}
        for r in rows
    ]


def delete_fact(fact_id: int):
    conn = _raw_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM memory_facts WHERE id = %s", (fact_id,))
    conn.commit()
    cur.close(); conn.close()
