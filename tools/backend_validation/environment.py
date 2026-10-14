"""Report actual backend selection and installed versions, not environment labels."""
import argparse
import importlib.metadata as metadata
import json
import platform


def validate_identity(env_id, numpy_version, selected_backend):
    if env_id in ('numpy1', 'numpy2'):
        if numpy_version.split('.')[0] != env_id[-1]:
            return {'ok': False, 'reason': 'numpy_major_mismatch'}
        expected = 'numpy'
    elif env_id in ('jax', 'torch', 'cupy'):
        expected = env_id
    else:
        return {'ok': False, 'reason': 'unknown_environment'}
    if selected_backend != expected:
        return {'ok': False, 'reason': 'backend_mismatch'}
    return {'ok': True, 'reason': None}


def array_evidence(array):
    """Inspect native output, never infer placement from a requested context."""
    module = type(array).__module__
    backend = next((name for name in ('numpy', 'cupy', 'torch', 'jax')
                    if module.startswith(name)), 'unknown')
    device = getattr(array, 'device', 'cpu')
    if callable(device):
        device = device()
    if backend == 'jax':
        devices = array.devices()
        if len(devices) != 1:
            raise ValueError('smoke requires a single native JAX device')
        device = next(iter(devices))
        device = 'cpu' if device.platform == 'cpu' else f'cuda:{device.id}'
    elif backend == 'cupy':
        device = f'cuda:{device.id}'
    return dict(backend=backend, device=str(device),
                dtype=str(array.dtype).removeprefix('torch.'),
                array_type=f'{module}.{type(array).__qualname__}')


def check_error(value, reference):
    """Frozen dual bounds for this tiny float64 installation control only."""
    import numpy as np
    if value.shape != reference.shape or not np.isfinite(value).all():
        return dict(ok=False, reason='nonfinite_or_shape_mismatch')
    delta = value - reference
    absolute = float(np.linalg.norm(delta))
    maximum = float(np.max(np.abs(delta)))
    bound = 1e-12 + 1e-12 * float(np.linalg.norm(reference))
    max_bound = 1e-12 + 1e-12 * float(np.max(np.abs(reference)))
    return dict(ok=bool(np.isfinite(value).all() and absolute <= bound and maximum <= max_bound),
                frobenius_error=absolute, maximum_error=maximum,
                frobenius_bound=bound, maximum_bound=max_bound)


def collect(env_id, *, device='cpu'):
    import numpy as np
    versions = {}
    for name in ('numpy', 'scipy', 'qutip', 'jax', 'jaxlib', 'torch', 'cupy',
                 'cupy-cuda12x', 'cupy-cuda13x', 'jax-cuda12-plugin',
                 'jax-cuda12-pjrt', 'jax-cuda13-plugin', 'jax-cuda13-pjrt'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    report = {
        'ok': False, 'reason': None, 'requested_device': device,
        'environment': env_id, 'python': platform.python_version(),
        'versions': versions, 'scope': 'tiny float64 matmul and SVD installation smoke',
    }
    try:
        from renormalizer.backend.context import make_context
        # Explicit construction cannot silently turn a missing GPU into CPU;
        # host transfers below are solely for the independent acceptance check.
        context = make_context('numpy' if env_id in ('numpy1', 'numpy2') else env_id,
                               device=device, real_dtype='float64', host_policy='explicit')
        ops = context.ops
        host = np.array([[1., 2.], [3., -1.], [2., 4.]])
        a = ops.from_numpy(host)
        b = ops.array([[2., 1.], [-1., 3.]], dtype='float64')
        product = ops.matmul(a, b)
        u, s, vh = ops.svd(a)
        reconstruction = ops.matmul(ops.multiply(u, s), vh)
        ops.sync()
        # Inspect every factor as well as products, so a host decomposition
        # cannot be masked by copying only the final result to the GPU.
        witnesses = {name: array_evidence(value) for name, value in
                     (('input', a), ('matmul', product), ('u', u), ('s', s),
                      ('vh', vh), ('svd_reconstruction', reconstruction))}
        report.update(witnesses=witnesses, **witnesses['matmul'])
        for witness in witnesses.values():
            identity = validate_identity(env_id, np.__version__, witness['backend'])
            if not identity['ok']:
                report.update(identity)
                return report
            if witness['device'] != device:
                report['reason'] = 'native_device_mismatch'
                return report
            if witness['dtype'] != 'float64':
                report['reason'] = 'precision_mismatch'
                return report
        if context.adapter.name == 'torch':
            import torch
            report['torch_build_cuda'] = torch.version.cuda
        checks = {
            'matmul': check_error(ops.to_numpy(product), np.array([[0., 7.], [7., 0.], [0., 14.]])),
            'svd_reconstruction': check_error(ops.to_numpy(reconstruction), host),
        }
        report.update(checks=checks, ok=all(check['ok'] for check in checks.values()))
        report['reason'] = None if report['ok'] else 'numerical_error'
    except Exception as error:
        # Emit machine-readable failure without retrying on a different device.
        report.update(ok=False, reason='smoke_failed', exception_type=type(error).__name__,
                      error=str(error))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('environment', choices=('numpy1', 'numpy2', 'jax', 'torch', 'cupy'))
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    args = parser.parse_args()
    result = collect(args.environment, device=args.device)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    raise SystemExit(0 if result['ok'] else 1)


if __name__ == '__main__':
    main()
