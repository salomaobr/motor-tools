"""Teste de integração MCP (stdio). Requer o pacote `mcp` instalado."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = str(Path(__file__).resolve().parents[1])


@pytest.mark.mcp
def test_mcp_stdio_roundtrip():
    async def run():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "motor_tools.mcp_server"],
            cwd=ROOT,
        )
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = {t.name for t in (await s.list_tools()).tools}
                from motor_tools import TOOLS
                assert tools == set(TOOLS)
                res = await s.call_tool("sign_test", {"wins": 5, "losses": 0})
                assert "0.03125" in res.content[0].text

    asyncio.run(run())


@pytest.mark.mcp
@pytest.mark.parametrize("entry", [
    ["-m", "motor_tools.mcp_server"],
    ["-m", "motor_tools", "mcp"],
])
def test_mcp_new_tools_roundtrip(entry):
    new_tools = {"induce_structure", "closed_loop_discovery", "summarize_evidence",
                 "propose_hypotheses", "run_research_experiment"}

    async def run():
        params = StdioServerParameters(command=sys.executable, args=entry, cwd=ROOT)
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = {t.name for t in (await s.list_tools()).tools}
                assert len(tools) == 25 and new_tools <= tools
                res = await s.call_tool(
                    "propose_hypotheses",
                    {"status": "mutated", "gain": 0.25, "evidence_strength": "strong"},
                )
                assert "H1" in res.content[0].text
                res = await s.call_tool("induce_structure", {"samples": []})
                assert '"supported": false' in res.content[0].text.lower()

    asyncio.run(run())


@pytest.mark.mcp
def test_mcp_exposes_seed_and_cv_strategy():
    samples = [[[0.1 * i, 0.05 * (i % 5)], 0.0, [0.1 * i + 0.01, 0.05 * (i % 5) - 0.01]]
               for i in range(40)]

    async def run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "motor_tools.mcp_server"], cwd=ROOT,
        )
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = {t.name: t.model_dump(by_alias=True)
                         for t in (await s.list_tools()).tools}
                for name in ("induce_structure", "closed_loop_discovery"):
                    props = tools[name]["inputSchema"]["properties"]
                    assert {"seed", "cv_strategy"} <= set(props)
                res = await s.call_tool(
                    "induce_structure",
                    {"samples": samples, "seed": 3, "cv_strategy": "time_series"},
                )
                body = json.loads(res.content[0].text)
                assert body["supported"] is True
                assert (body["seed"], body["cv_strategy"]) == (3, "time_series")

    asyncio.run(run())
