"""Source identity must not depend on the importing application's Git state."""
from pathlib import Path
import os
import subprocess
import pytest
import renormalizer.cons as cons


def test_git_identity_ignores_cwd_and_git_environment(tmp_path, monkeypatch):
    root = Path(cons.__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith('GIT_')}
    expected = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], env=env).decode().strip()
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True, env=env)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GIT_DIR', str(tmp_path / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(tmp_path))
    assert cons.get_git_commit_hash() == expected


def test_missing_git_and_installed_source_have_unknown_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('PATH', str(tmp_path))
    assert cons.get_git_commit_hash() == 'Unknown'
    monkeypatch.setattr(cons, '__file__', str(tmp_path / 'renormalizer' / 'cons.py'))
    assert cons.get_git_commit_hash() == 'Unknown'


def test_git_permission_error_is_nonfatal(monkeypatch):
    def unavailable(*args, **kwargs):
        raise PermissionError('git executable inaccessible')
    monkeypatch.setattr(cons.subprocess, 'check_output', unavailable)
    assert cons.get_git_commit_hash() == 'Unknown'


def test_installed_package_does_not_borrow_enclosing_repository(tmp_path, monkeypatch):
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    monkeypatch.setattr(cons, '__file__', str(tmp_path / 'site-packages' / 'renormalizer' / 'cons.py'))
    # No Git metadata at the package root: do not walk up into the host repo.
    assert cons.get_git_commit_hash() == 'Unknown'


def test_mismatched_git_root_is_unknown_and_interrupts_propagate(tmp_path, monkeypatch):
    monkeypatch.setattr(cons.subprocess, 'check_output', lambda *a, **k: str(tmp_path).encode())
    assert cons.get_git_commit_hash() == 'Unknown'
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(cons.subprocess, 'check_output', interrupted)
    with pytest.raises(KeyboardInterrupt):
        cons.get_git_commit_hash()


def test_sizeof_fmt_retains_historical_import_identity():
    from renormalizer.mps.backend import sizeof_fmt
    from renormalizer.utils.utils import sizeof_fmt as original
    assert sizeof_fmt is original
    assert sizeof_fmt(1024) == '1.0KiB'
    from renormalizer.mps.backend import __all__
    assert 'sizeof_fmt' in __all__
