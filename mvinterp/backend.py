"""CUDA backend (CuPy is required).

This project targets the `grammar` GPU cluster, so CuPy/CUDA is assumed to be
present. There is intentionally NO NumPy fallback for the compute path: import
`xp` (= cupy) from here. Host-side data prep (mock generation, growth ODE) still
uses NumPy/SciPy and is transferred to the device by the interpolator.
"""
import time
import contextlib

try:
    import cupy as xp  # noqa: F401
except Exception as e:  # pragma: no cover
    raise ImportError(
        "mvinterp requires CuPy (CUDA). On the GPU node install the wheel that "
        "matches the cluster toolkit, e.g. `pip install cupy-cuda12x`."
    ) from e


def backend_name():
    dev = xp.cuda.runtime.getDeviceProperties(xp.cuda.runtime.getDevice())
    return f"cupy (GPU: {dev['name'].decode()})"


def asnumpy(a):
    return xp.asnumpy(a)


def to_device(a, dtype=None):
    return xp.asarray(a, dtype=dtype)


def sync():
    xp.cuda.Stream.null.synchronize()


def free_pool():
    """Release cached device memory back to the driver (call between big stages)."""
    xp.get_default_memory_pool().free_all_blocks()


def device_mem_gb():
    free, total = xp.cuda.runtime.memGetInfo()
    return (total - free) / 1e9, total / 1e9


@contextlib.contextmanager
def timed(label, store=None):
    """Wall-clock timer that syncs the GPU first (honest timing)."""
    sync()
    t0 = time.perf_counter()
    yield
    sync()
    dt = time.perf_counter() - t0
    if store is not None:
        store[label] = dt
    print(f"[time] {label}: {dt*1e3:.1f} ms")
