"""
공지·문서(notice) 컬렉션 RAG MCP 서버

실행: python -m mcp_servers.notice_mcp.server  (rag-lab 루트에서)
"""

from mcp_servers.common import build_server

mcp = build_server("notice-rag", collection="notice", label="공지·문서")

if __name__ == "__main__":
    mcp.run()
