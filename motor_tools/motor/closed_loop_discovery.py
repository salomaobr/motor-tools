"""Compatibility import for the historical ClosedLoopDiscovery module name.

The implementation lives in :mod:`motor_tools.motor.discovery_loop`.
"""
from motor_tools.motor.discovery_loop import (
    AcceptedMutation,
    ClosedLoopDiscovery,
    DiscoveryResult,
)

__all__ = ["AcceptedMutation", "ClosedLoopDiscovery", "DiscoveryResult"]
