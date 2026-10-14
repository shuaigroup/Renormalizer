"""Development and test tools for array backends.

* ``evidence``: record which adapter and device ran each contraction and which
  arrays crossed the host/device boundary (``record_execution``).
* ``strict``: the strict array API (``StrictOperations``, ``DeviceOperations``)
  reached as ``NumericalContext.ops``; it specifies adapter semantics and serves
  as the certification surface for operator providers.

Algorithms never import this package.
"""
from .evidence import ExecutionLedger, record_execution
from .strict import StrictOperations, DeviceOperations

__all__ = ["ExecutionLedger", "record_execution", "StrictOperations", "DeviceOperations"]
