"""Small trusted host fixtures independent of candidate dispatch."""
import math
import numpy as np


def matmul_fixture(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((7, 5))
    b = rng.standard_normal((5, 3))
    expected = np.array([[math.fsum(float(a[i, k]) * float(b[k, j])
                                   for k in range(5))
                          for j in range(3)] for i in range(7)], dtype=np.float64)
    return a, b, expected


def write_fixture(directory, *, seed=27):
    """Freeze a small well-conditioned control before any candidate is scored."""
    import hashlib
    import json
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, array in zip(('a', 'b', 'reference'), matmul_fixture(seed)):
        path = directory / f'{name}.npy'
        # Exclusive creation prevents silently replacing a frozen reference.
        with path.open('xb') as stream:
            np.save(stream, array, allow_pickle=False)
        arrays[name] = {'file': path.name, 'shape': list(array.shape),
                        'dtype': str(array.dtype),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {
        'schema_version': 1, 'op': 'matmul', 'seed': seed,
        'numpy_version': np.__version__, 'arrays': arrays,
        'reference_method': 'float64 products with independent math.fsum reduction',
        'tolerances': {'atol_F': 1e-12, 'rtol_F': 1e-12,
                       'atol_max': 1e-12, 'rtol_max': 1e-12},
        'tolerance_scope': 'this small float64 control only; not a universal budget',
        'verification_dtype': 'float64',
    }
    path = directory / 'manifest.json'
    with path.open('x') as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2)
        stream.write('\n')
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=27)
    args = parser.parse_args()
    print(write_fixture(args.output, seed=args.seed))
