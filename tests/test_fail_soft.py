"""call_tool é fail-soft: nunca levanta exceção, qualquer que seja a entrada."""
import inspect

import pytest

from motor_tools import TOOLS, call_tool


@pytest.mark.parametrize("bad", [None, 1, ["sign_test"], {"a": 1}, ""])
def test_bad_tool_name_types(bad):
    r = call_tool(bad)
    assert r["supported"] is False
    assert "error" in r


def test_unknown_tool_lists_available():
    r = call_tool("nope")
    assert r["supported"] is False
    assert set(r["available"]) == set(TOOLS)


def test_non_dict_arguments():
    r = call_tool("sign_test", ["wins"])
    assert r["supported"] is False
    assert "error" in r


def test_missing_required_argument():
    r = call_tool("sign_test", {"wins": 5})
    assert r["supported"] is False
    assert "error" in r


def test_wrong_argument_type():
    r = call_tool("sign_test", {"wins": "muitas", "losses": 1})
    assert r["supported"] is False
    assert r["tool"] == "sign_test"


@pytest.mark.parametrize("name", sorted(TOOLS))
def test_unexpected_argument(name):
    r = call_tool(name, {"__not_a_param__": 1})
    assert r["supported"] is False
    assert r["tool"] == name


# tools sem parâmetros (ex.: agi_generalization_battery) não têm o que preencher com lixo
PARAM_TOOLS = sorted(n for n, f in TOOLS.items() if inspect.signature(f).parameters)


@pytest.mark.parametrize("name", PARAM_TOOLS)
@pytest.mark.parametrize("junk", ["x", None, [], {}, -1, 1.5])
def test_junk_arguments_never_raise(name, junk):
    params = list(inspect.signature(TOOLS[name]).parameters)
    r = call_tool(name, {p: junk for p in params})
    assert isinstance(r, dict)
