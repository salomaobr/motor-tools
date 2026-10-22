"""Import smoke for package modules and packaging layout."""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "mod",
    [
        "motor_tools.motor.oos_statistics",
        "motor_tools.motor.novelty_detector",
        "motor_tools.motor.agi_transfer_robustness",
        "motor_tools.motor.agi_active_recovery",
        "motor_tools.motor.agi_generalization_battery",
        "motor_tools.motor.structure_inducer",
        "motor_tools.motor.discovery_loop",
        "motor_tools.motor.research_evidence",
        "motor_tools.motor.research_scientist",
        "motor_tools.motor.research_loop",
        "motor_tools.domains.alien_physics",
        "motor_tools.api",
    ],
)
def test_import(mod):
    importlib.import_module(mod)


def test_every_motor_module_imports():
    """Nenhum módulo mantido pode depender de código apagado."""
    import motor_tools.motor as pkg

    names = [m.name for m in pkgutil.iter_modules(pkg.__path__)]
    assert len(names) == 30  # módulo de compatibilidade closed_loop_discovery incluído
    for name in names:
        importlib.import_module(f"motor_tools.motor.{name}")


def test_domains_present():
    assert (ROOT / "motor_tools" / "domains" / "alien_physics.py").exists()


@pytest.mark.parametrize("name", ["motor", "domains", "core", "runtime"])
def test_no_top_level_leak_in_source_tree(name):
    """motor/domains/core/runtime vivem dentro de motor_tools/, nunca na raiz do repositório."""
    assert not (ROOT / name).exists()
