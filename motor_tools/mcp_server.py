"""Servidor MCP (stdio). Uso: python -m motor_tools.mcp_server

Compatível com mcp 1.x (FastMCP) e 2.x (MCPServer).
"""
try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from motor_tools import api

mcp = _Server("motor-tools")

for _name, _fn in api.TOOLS.items():
    mcp.tool(name=_name, description=(_fn.__doc__ or _name).strip())(_fn)

if __name__ == "__main__":
    mcp.run()
