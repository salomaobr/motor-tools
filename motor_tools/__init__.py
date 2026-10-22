"""motor_tools: 25 ferramentas estatísticas, de monitoramento e de descoberta simbólica (MCP + HTTP).

Todas as funções em `motor_tools.api` são puras (sem estado entre chamadas),
recebem e devolvem apenas tipos JSON (dict/list/float/bool/str).
`call_tool` é fail-soft: nunca levanta exceção, devolve {"supported": False, "error": ...}.
"""
from motor_tools.api import TOOLS, call_tool  # noqa: F401

__version__ = "0.8.1"
__all__ = ["TOOLS", "call_tool", "__version__"]
