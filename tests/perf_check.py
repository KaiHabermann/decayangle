"""
Performance check: helicity and Wigner angles for a 3-body decay over 100_000 momenta.

Run with:
    python tests/perf_check.py
"""

import time
import numpy as np

from jax import config as jax_cfg

jax_cfg.update("jax_enable_x64", True)

from decayangle.decay_topology import TopologyCollection
from decayangle.config import config as cfg

try:
    import decayangle_rs  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

N = 100_000

# ── helpers ──────────────────────────────────────────────────────────────────


def make_momenta(n, backend):
    """Random 3-body momenta in the mother rest frame (approx)."""
    rng = np.random.default_rng(42)
    p1 = np.stack(
        [
            rng.uniform(-0.5, 0.5, n),
            rng.uniform(-0.5, 0.5, n),
            rng.uniform(-0.9, 0.9, n),
            np.ones(n),
        ],
        axis=-1,
    )
    p2 = np.stack(
        [
            rng.uniform(-0.5, 0.5, n),
            rng.uniform(-0.5, 0.5, n),
            rng.uniform(-0.4, 0.4, n),
            np.ones(n),
        ],
        axis=-1,
    )
    p3 = np.stack(
        [
            -p1[:, 0] - p2[:, 0],
            -p1[:, 1] - p2[:, 1],
            -p1[:, 2] - p2[:, 2],
            np.ones(n),
        ],
        axis=-1,
    )

    if backend == "jax":
        import jax.numpy as jnp

        return {1: jnp.array(p1), 2: jnp.array(p2), 3: jnp.array(p3)}
    return {1: p1, 2: p2, 3: p3}


def time_it(fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    return elapsed, result


def print_table(rows):
    col_widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    header, *data = rows
    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"
    print(sep)
    print(fmt.format(*header))
    print(sep)
    for row in data:
        print(fmt.format(*row))
    print(sep)


# ── benchmark ────────────────────────────────────────────────────────────────


def run_benchmark(backend_name, use_rust=False):
    cfg.backend = backend_name
    cfg.use_rust = use_rust

    tg = TopologyCollection(0, [1, 2, 3])
    topologies = tg.topologies
    reference = topologies[0]

    momenta_raw = make_momenta(N, backend_name)

    t_rest, momenta = time_it(lambda: reference.to_rest_frame(momenta_raw))

    def all_helicity():
        for topo in topologies:
            topo.helicity_angles(momenta)

    def all_wigner():
        for topo in topologies:
            reference.relative_wigner_angles(topo, momenta)

    t_hel, _ = time_it(all_helicity)
    t_wig, _ = time_it(all_wigner)

    # JAX: second run separates JIT from execution
    if backend_name == "jax":
        t_rest2, momenta = time_it(lambda: reference.to_rest_frame(momenta_raw))
        t_hel2, _ = time_it(all_helicity)
        t_wig2, _ = time_it(all_wigner)
        return {
            "to_rest_frame (1st / JIT)": f"{t_rest:.3f}s",
            "to_rest_frame (2nd / exec)": f"{t_rest2:.3f}s",
            "helicity_angles (1st / JIT)": f"{t_hel:.3f}s",
            "helicity_angles (2nd / exec)": f"{t_hel2:.3f}s",
            "wigner_angles (1st / JIT)": f"{t_wig:.3f}s",
            "wigner_angles (2nd / exec)": f"{t_wig2:.3f}s",
        }

    return {
        "to_rest_frame": f"{t_rest:.3f}s",
        "helicity_angles": f"{t_hel:.3f}s",
        "wigner_angles": f"{t_wig:.3f}s",
    }


def main():
    print(f"\nPerformance check — 3-body decay, N = {N:,} momenta\n")

    results_np = run_benchmark("numpy", use_rust=False)
    results_jax = run_benchmark("jax", use_rust=False)

    all_ops = sorted(set(results_np) | set(results_jax))

    if RUST_AVAILABLE:
        results_rust = run_benchmark("numpy", use_rust=True)

        # Speedup vs NumPy (only for keys present in both)
        def speedup(py_str, rs_str):
            try:
                py_t = float(py_str.rstrip("s"))
                rs_t = float(rs_str.rstrip("s"))
                if rs_t == 0:
                    return "—"
                return f"{py_t / rs_t:.1f}x"
            except (ValueError, ZeroDivisionError):
                return "—"

        rows = [("Operation", "NumPy", "JAX", "Rust", "Speedup (NumPy/Rust)")]
        for op in all_ops:
            py = results_np.get(op, "—")
            jx = results_jax.get(op, "—")
            rs = results_rust.get(op, "—")
            spd = speedup(py, rs) if rs != "—" else "—"
            rows.append((op, py, jx, rs, spd))
    else:
        print(
            "(Rust extension not available — build with `cd decayangle-rs && maturin develop --release`)\n"
        )
        rows = [("Operation", "NumPy", "JAX")]
        for op in all_ops:
            rows.append((op, results_np.get(op, "—"), results_jax.get(op, "—")))

    print_table(rows)
    print()


if __name__ == "__main__":
    main()
