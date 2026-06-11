"""
강의자료(lecture) 컬렉션 RAG MCP 서버

실행: python -m mcp_servers.lecture_mcp.server  (rag-lab 루트에서)
"""

from mcp_servers.common import build_server

mcp = build_server("lecture-rag", collection="lecture", label="강의자료")

if __name__ == "__main__":
    mcp.run()
