"""Validated example launcher used by run_examples.py.

Usage: python _example_worker.py --source TREE --metadata env.json SCRIPT [ARGS]
Writes interpreter/backend provenance before running the example in its workdir.
"""
import argparse
import cProfile
from pathlib import Path
import runpy
import sys
from _common import provenance, source_tree, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=source_tree, required=True)
    ap.add_argument('--metadata', type=Path, required=True)
    ap.add_argument('--profile', type=Path)
    ap.add_argument('script')
    ap.add_argument('args', nargs=argparse.REMAINDER)
    opt = ap.parse_args()
    write_json(opt.metadata, provenance(opt.source))
    sys.argv = [opt.script] + opt.args
    sys.path.insert(0, str(Path(opt.script).resolve().parent))
    if opt.profile:
        profiler = cProfile.Profile()
        try:
            profiler.runcall(runpy.run_path, opt.script, run_name='__main__')
        finally:
            profiler.dump_stats(str(opt.profile))
    else:
        runpy.run_path(opt.script, run_name='__main__')


if __name__ == '__main__':
    main()
