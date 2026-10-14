"""Reviewed admission uses trusted source identities; untrusted is fail-closed."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
from .protocol import read_file, MAX_FILE_BYTES, ProtocolError
from .sandbox import probe


@dataclass(frozen=True)
class ReviewRecord:
    source_hash: str
    entry: str = 'candidate'
    dependencies: tuple = ()
    compiler_options: tuple = ()


@dataclass(frozen=True)
class ExecutionPolicy:
    review: ReviewRecord | None = None
    provider_config: dict | None = None


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def source_bytes(path):
    path=Path(path)
    return read_file(path.parent,path.name,max_bytes=MAX_FILE_BYTES)


def review_source(source, *, dependencies=(), compiler_options=()):
    """Record hashes AFTER caller review. This function does not review code.

    Dependency entries are (basename, absolute source path, SHA256). Runtime
    dependency versions are independently bound by the full environment hash.
    Only a single fixed candidate(a,b) Python entry is supported by the worker.
    """
    records=[]
    names={'candidate.py','worker.py','protocol.py','scoring.py','manifest.json','limits.json'}
    for path in dependencies:
        path=Path(path).absolute()
        if path.name in names or path.suffix!='.py':
            raise ValueError('dependency name conflicts with trusted runtime')
        names.add(path.name)
        records.append((path.name,str(path),hashlib.sha256(source_bytes(path)).hexdigest()))
    return ReviewRecord(hashlib.sha256(source_bytes(source)).hexdigest(),dependencies=tuple(records),
                        compiler_options=tuple(compiler_options))


def admit(mode, device, *, candidate, policy):
    if not isinstance(policy,ExecutionPolicy):
        return Decision(False,'invalid_execution_policy')
    if mode=='untrusted':
        if device!='cpu':
            return Decision(False,'untrusted_gpu_not_admitted')
        evidence=probe(policy.provider_config)
        return Decision(evidence.verified,evidence.reason)
    if mode!='reviewed':
        return Decision(False,'unknown_mode')
    review=policy.review
    if not isinstance(review,ReviewRecord) or review.entry!='candidate':
        return Decision(False,'review_missing_or_entry_mismatch')
    try:
        if hashlib.sha256(source_bytes(candidate)).hexdigest()!=review.source_hash:
            return Decision(False,'review_hash_mismatch')
        for name,path,expected in review.dependencies:
            if Path(path).name!=name or hashlib.sha256(source_bytes(path)).hexdigest()!=expected:
                return Decision(False,'review_dependency_mismatch')
    except (ProtocolError,ValueError,TypeError):
        return Decision(False,'review_source_unreadable')
    if device!='cpu':
        return Decision(False,'reviewed_gpu_requires_trusted_harness')
    return Decision(True,'reviewed_cpu_admitted_without_security_sandbox')
