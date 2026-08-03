"""
Topology.helicity_angles and Topology.relative_wigner_angles delegate to the
Rust extension when cfg.use_rust is True. That delegation converts momenta to
concrete numpy arrays, which is impossible while jax is tracing (e.g. inside
jax.jit or jax.grad): the momenta are abstract tracers with no concrete value.

cfg.check_gate is used to detect this and fall back to the pure Python/JAX
implementation instead of handing tracers to the Rust extension. These tests
verify that the fallback is actually taken while tracing, and that its result
agrees with the plain (non-rust) computation.

Note: kinematics.py/decay_topology.py/lorentz.py snapshot cfg.backend into a
module-level `cb` at import time, so cfg.backend must be set to "jax" before
decayangle.decay_topology is first imported in the process. This is done in
a subprocess to avoid depending on/interfering with import order in the rest
of the test suite.
"""

import json
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import json
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    from decayangle.config import config as cfg

    # Must happen before decay_topology/kinematics are imported: those
    # modules snapshot cfg.backend into a module-level `cb` at import time.
    cfg.backend = "jax"
    cfg.sorting = "off"

    from decayangle.decay_topology import TopologyCollection


    def make_momenta():
        p1 = jnp.array([0.15, 0.05, 0.3, 1.0])
        p2 = jnp.array([-0.2, 0.1, -0.1, 1.0])
        p3 = jnp.array([0.05, -0.15, -0.2, 1.0])
        return {1: p1, 2: p2, 3: p3}


    tc = TopologyCollection(0, [1, 2, 3])
    topo_a, topo_b = tc.topologies[0], tc.topologies[1]

    # Reference result computed without Rust, so it works whether or not the
    # decayangle_rs extension is even built in this environment.
    cfg.use_rust = False
    momenta = topo_a.to_rest_frame(make_momenta())
    ref_h = topo_a.helicity_angles(momenta)
    ref_w = topo_a.relative_wigner_angles(topo_b, momenta)

    # Now turn Rust on. If the tracer guard is missing, tracing either of the
    # functions below will try to convert jax tracers to concrete numpy
    # arrays for the Rust extension and blow up.
    cfg.use_rust = True


    @jax.jit
    def traced_helicity(momenta):
        angles = topo_a.helicity_angles(momenta)
        key = next(iter(angles))
        return angles[key].phi_rf, angles[key].theta_rf


    @jax.jit
    def traced_wigner(momenta):
        angles = topo_a.relative_wigner_angles(topo_b, momenta)
        key = next(iter(angles))
        return angles[key].phi_rf, angles[key].theta_rf, angles[key].psi_rf


    h_phi, h_theta = traced_helicity(momenta)
    w_phi, w_theta, w_psi = traced_wigner(momenta)

    h_key = next(iter(ref_h))
    w_key = next(iter(ref_w))

    result = {
        "h_phi": float(h_phi),
        "h_theta": float(h_theta),
        "w_phi": float(w_phi),
        "w_theta": float(w_theta),
        "w_psi": float(w_psi),
        "ref_h_phi": float(ref_h[h_key].phi_rf),
        "ref_h_theta": float(ref_h[h_key].theta_rf),
        "ref_w_phi": float(ref_w[w_key].phi_rf),
        "ref_w_theta": float(ref_w[w_key].theta_rf),
        "ref_w_psi": float(ref_w[w_key].psi_rf),
    }
    print(json.dumps(result))
    """
)


def test_rust_dispatch_skipped_while_jax_tracing():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result["h_phi"] == pytest.approx(result["ref_h_phi"], abs=1e-8)
    assert result["h_theta"] == pytest.approx(result["ref_h_theta"], abs=1e-8)
    assert result["w_phi"] == pytest.approx(result["ref_w_phi"], abs=1e-8)
    assert result["w_theta"] == pytest.approx(result["ref_w_theta"], abs=1e-8)
    assert result["w_psi"] == pytest.approx(result["ref_w_psi"], abs=1e-8)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
