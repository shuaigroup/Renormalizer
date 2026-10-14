"""Inventory a fixed Git tree; string matches are review inputs, not failures."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def _git(*args):
    return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True, text=True)


def _matches(commit, pattern):
    result = _git('grep', '-l', '-E', pattern, commit, '--', 'renormalizer/')
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return sorted(line.split(':', 1)[1] for line in result.stdout.splitlines())


def inventory(ref):
    resolved = _git('rev-parse', '--verify', f'{ref}^{{commit}}')
    if resolved.returncode:
        raise ValueError('reference is not a local commit')
    commit = resolved.stdout.strip()
    facade = _matches(commit, r'mps\.backend|OE_BACKEND|USE_GPU|MEMORY_ERRORS|ARRAY_TYPES|RENO_GPU|RENO_FP32')
    return {'source': ref, 'source_commit': commit, 'facade': facade,
            'matrix_helpers': _matches(commit, r'asnumpy|asxp'),
            'categories': {kind: [p for p in facade if
                          ('/tests/' in p if kind == 'tests' else
                           '/lib/' in p if kind == 'lib' else
                           '/tests/' not in p and '/lib/' not in p)]
                           for kind in ('core', 'tests', 'lib')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ref')
    print(json.dumps(inventory(parser.parse_args().ref), indent=2, sort_keys=True))
