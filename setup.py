"""Compatibility shim; project metadata lives only in pyproject.toml.

Keeping a second dependency/Python-version list here made legacy and modern
build entry points disagree. Setuptools reads the same PEP 621 metadata for
both; release builds should use ``python -m build``.
"""
from setuptools import setup

setup()
