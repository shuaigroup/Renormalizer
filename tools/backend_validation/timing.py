"""Trusted clocks; candidate return values are never timing authorities."""
from time import perf_counter
from statistics import median
import math
import hashlib


def timed_reviewed_call(cp, function, *, stream):
    stream.synchronize()
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    with stream:
        begin.record()
        value = function()
        end.record()
    end.synchronize()
    return value, cp.cuda.get_elapsed_time(begin, end) / 1000.0


def measure_reviewed(cp, function, inputs, *, repeats=5, warmup=1):
    """Fixed host-input workload, including upload/download in e2e.

    Single stream only. Cold includes first upload, call/compile, and download;
    process/import are measured separately by the parent runner. Pool samples
    are endpoints, not continuous peaks. Inputs must be host arrays.
    """
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError('invalid repeat counts')
    stream = cp.cuda.get_current_stream()
    samples = []
    output_hashes = []
    for iteration in range(1+warmup+repeats):
        stream.synchronize()
        start = perf_counter()
        arrays = [cp.asarray(a) for a in inputs]
        value, kernel = timed_reviewed_call(cp, lambda: function(*arrays), stream=stream)
        host = value.get()
        stream.synchronize()
        output_hashes.append(hashlib.sha256(host.tobytes(order='C')).hexdigest())
        elapsed = perf_counter()-start
        sample = dict(t_e2e=elapsed,t_kernel=kernel,
                      h2d_bytes=sum(a.nbytes for a in inputs), d2h_bytes=host.nbytes,
                      pool_used=cp.get_default_memory_pool().used_bytes(),
                      pool_reserved=cp.get_default_memory_pool().total_bytes())
        if iteration == 0:
            cold = elapsed
        elif iteration > warmup:
            samples.append(sample)
        del arrays, value
    return host, dict(cold_call_seconds=cold,samples=samples,output_hashes=output_hashes,
                      summary={key:dict(median=median(s[key] for s in samples), p90=sorted(s[key] for s in samples)[math.ceil(0.9*len(samples))-1], minimum=min(s[key] for s in samples), maximum=max(s[key] for s in samples)) for key in ("t_e2e","t_kernel")},
                      cold_boundary='first transfer/call-compilation/download; excludes process/import',
                      e2e_boundary='host inputs through completed host output',
                      memory_coverage='pool endpoint samples only', warmup=warmup,repeats=repeats)
