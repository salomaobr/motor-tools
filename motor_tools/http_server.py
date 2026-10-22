"""Servidor HTTP com OpenAPI automático.

Uso:
    uvicorn motor_tools.http_server:app --port 8000

Autenticação opcional:
    export MOTOR_TOOLS_API_KEY=seu_segredo
    # Depois envie header: Authorization: Bearer seu_segredo
    # (ou X-API-Key: seu_segredo)

Cada ferramenta vira POST /tools/<nome> com corpo JSON = argumentos.
O esquema em /openapi.json serve para GPT Actions, function calling etc.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import Body, Depends, FastAPI, Header, HTTPException

from motor_tools import api
from motor_tools import __version__

_API_KEY = os.environ.get("MOTOR_TOOLS_API_KEY", "").strip()

app = FastAPI(
    title="motor-tools",
    version=__version__,
    description="Stateless JSON tools for any AI or system. No agent runtime required.",
)


def _check_auth(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Se MOTOR_TOOLS_API_KEY estiver definida, exige Bearer ou X-API-Key."""
    if not _API_KEY:
        return
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    elif x_api_key:
        token = x_api_key.strip()
    if token != _API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _make(name: str, fn):
    def endpoint(
        args: Dict[str, Any] = Body(default_factory=dict),
        _: None = Depends(_check_auth),
    ) -> Dict[str, Any]:
        try:
            return fn(**args)
        except (ValueError, TypeError, RuntimeError) as e:
            raise HTTPException(status_code=422, detail=str(e))

    endpoint.__name__ = name
    app.post(
        f"/tools/{name}",
        summary=name,
        description=(fn.__doc__ or "").strip(),
        operation_id=name,
    )(endpoint)


for _n, _f in api.TOOLS.items():
    _make(_n, _f)


@app.get("/tools")
def list_tools(_: None = Depends(_check_auth)) -> Dict[str, str]:
    return {n: (f.__doc__ or "").strip().split("\n")[0] for n, f in api.TOOLS.items()}


@app.get("/health")
def health() -> Dict[str, str]:
    """Healthcheck público (não exige API key)."""
    return {"status": "ok", "version": __version__}
