"""Alien Physics — benchmark mínimo de descoberta e transferência.

O agente observa (posição, velocidade, propriedade) e escolhe uma ação de força.
As regras do mundo não são expostas ao agente.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np


@dataclass(frozen=True)
class Rule:
    gravity: float
    drag: float
    nonlinear: float = 0.0
    property_bias: float = 0.0
    collision_gain: float = 1.0


RULES: Dict[str, Rule] = {
    "gravity": Rule(gravity=-0.40, drag=0.02),
    "property_gravity": Rule(gravity=-0.25, drag=0.02, property_bias=-0.30),
    "nonlinear": Rule(gravity=-0.20, drag=0.01, nonlinear=0.12),
    "collision": Rule(gravity=-0.30, drag=0.02, collision_gain=0.55),
}


class AlienPhysicsEnv:
    """Pequeno mundo contínuo com regras ocultas e intervenções controladas."""

    def __init__(self, world: str, seed: int = 0) -> None:
        if world not in RULES:
            raise ValueError(f"unknown world: {world}")
        self.world = world
        self.rule = RULES[world]
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, state: Tuple[float, float, float] | None = None) -> np.ndarray:
        if state is None:
            state = (float(self.rng.uniform(-1.0, 1.0)), float(self.rng.uniform(-0.8, 0.8)), float(self.rng.choice([-1.0, 1.0])))
        self.x, self.v, self.prop = map(float, state)
        return self.observe()

    def observe(self) -> np.ndarray:
        return np.asarray([self.x, self.v, self.prop], dtype=np.float32)

    def step(self, force: float) -> np.ndarray:
        f = float(np.clip(force, -1.0, 1.0))
        r = self.rule
        accel = r.gravity + f - r.drag * self.v
        if r.property_bias:
            accel += r.property_bias * (self.prop > 0.0)
        if r.nonlinear:
            accel += r.nonlinear * np.sign(self.v) * self.v * self.v
        self.v += accel
        self.x += self.v
        if self.world == "collision" and abs(self.x) > 2.0:
            self.x = float(np.sign(self.x) * 2.0)
            self.v = -self.v * r.collision_gain
        return self.observe()

    def transition(self, state: np.ndarray, force: float) -> np.ndarray:
        old = (self.x, self.v, self.prop)
        self.x, self.v, self.prop = map(float, np.asarray(state).reshape(-1))
        out = self.step(force).copy()
        self.x, self.v, self.prop = old
        return out
