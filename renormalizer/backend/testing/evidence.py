"""Execution evidence: which adapter and device ran each contraction, and which
arrays crossed the host/device boundary.

A development and test tool. ``record_execution`` activates recording for the
calls made inside it; outside it the hooks in ``execution`` do nothing.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import prod

import numpy as np

from .. import execution
from ..contracts import CapabilityError


@dataclass
class ExecutionLedger:
    adapter_id: int | None = None
    operations: list = field(default_factory=list)
    transfers: list = field(default_factory=list)

    def record_contraction(self, adapter, result):
        if self.adapter_id != id(adapter):
            raise CapabilityError('execution witness differs from requested context')
        self.operations.append(dict(operation='contraction', backend=adapter.name,
                                    device=execution._device(result), dtype=str(result.dtype),
                                    adapter_id=id(adapter), shape=tuple(result.shape)))

    def record_transfer(self, source, result, reason):
        src, dst = execution._device(source), execution._device(result)
        source_host = src.split(':')[0].lower() == 'cpu'
        target_host = dst.split(':')[0].lower() == 'cpu'
        direction = ('host_to_host' if source_host and target_host else
                     'H2D' if source_host else 'D2H' if target_host else 'device_to_device')
        itemsize = np.dtype(str(result.dtype).removeprefix('torch.')).itemsize
        self.transfers.append(dict(direction=direction, logical_bytes=prod(result.shape) * itemsize,
                                   source_device=src, target_device=dst, reason=reason,
                                   operation_id=len(self.operations)))


@contextmanager
def record_execution(context):
    """Record calls explicitly bound to context; does not select a backend."""
    ledger = ExecutionLedger(adapter_id=id(context.adapter))
    run = execution._run.get()
    token = execution._set_run(None if run is None else run.context, ledger)
    try:
        yield ledger
    finally:
        execution._run.reset(token)
