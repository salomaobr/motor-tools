"""Permite: python -m motor_tools [mcp|http]"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="motor-tools",
        description="Ferramentas estatísticas motor-tools (MCP ou HTTP).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("mcp", help="Inicia o servidor MCP (stdio)")

    p_http = sub.add_parser("http", help="Inicia o servidor HTTP/OpenAPI")
    p_http.add_argument("--host", default="127.0.0.1")
    p_http.add_argument("--port", type=int, default=8000)
    p_http.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "mcp":
        from motor_tools.mcp_server import mcp
        mcp.run()
        return 0

    if args.cmd == "http":
        import uvicorn
        uvicorn.run(
            "motor_tools.http_server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
