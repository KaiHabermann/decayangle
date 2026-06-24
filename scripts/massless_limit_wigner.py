"""
Plot relative Wigner theta angle vs particle mass approaching zero,
for 3-, 4-, and 5-body decays. One kinematic point each, two topology
pairs compared, particle 1 driven massless.

Usage:
    python scripts/massless_limit_wigner.py
"""

import numpy as np
import matplotlib.pyplot as plt
from decayangle.decay_topology import TopologyCollection
from decayangle.config import config as cfg

cfg.sorting = "off"
cfg.use_rust = False


# ── Kinematics helpers ────────────────────────────────────────────────────────


def _isotropic_unit(rng):
    phi = rng.uniform(0, 2 * np.pi)
    cos_th = rng.uniform(-1, 1)
    sin_th = np.sqrt(1 - cos_th**2)
    return np.array([sin_th * np.cos(phi), sin_th * np.sin(phi), cos_th])


def make_3body(m0, m1, m2, m3, seed=0):
    """One kinematic point for a 3-body decay in the mother rest frame."""
    rng = np.random.default_rng(seed)
    p1_mag = rng.uniform(0.05, 0.35) * m0
    p2_mag = rng.uniform(0.05, 0.35) * m0
    p1 = p1_mag * _isotropic_unit(rng)
    p2 = p2_mag * _isotropic_unit(rng)
    p3 = -(p1 + p2)
    E1 = np.sqrt(m1**2 + np.dot(p1, p1))
    E2 = np.sqrt(m2**2 + np.dot(p2, p2))
    E3 = np.sqrt(m3**2 + np.dot(p3, p3))
    return {
        1: np.array([*p1, E1]),
        2: np.array([*p2, E2]),
        3: np.array([*p3, E3]),
    }


def make_4body(m0, m1, m2, m3, m4, seed=0):
    """One kinematic point for a 4-body decay in the mother rest frame."""
    rng = np.random.default_rng(seed)
    p1_mag = rng.uniform(0.05, 0.25) * m0
    p2_mag = rng.uniform(0.05, 0.25) * m0
    p3_mag = rng.uniform(0.05, 0.25) * m0
    p1 = p1_mag * _isotropic_unit(rng)
    p2 = p2_mag * _isotropic_unit(rng)
    p3 = p3_mag * _isotropic_unit(rng)
    p4 = -(p1 + p2 + p3)
    E1 = np.sqrt(m1**2 + np.dot(p1, p1))
    E2 = np.sqrt(m2**2 + np.dot(p2, p2))
    E3 = np.sqrt(m3**2 + np.dot(p3, p3))
    E4 = np.sqrt(m4**2 + np.dot(p4, p4))
    return {
        1: np.array([*p1, E1]),
        2: np.array([*p2, E2]),
        3: np.array([*p3, E3]),
        4: np.array([*p4, E4]),
    }


def make_5body(m0, m1, m2, m3, m4, m5, seed=0):
    """One kinematic point for a 5-body decay in the mother rest frame."""
    rng = np.random.default_rng(seed)
    mags = [rng.uniform(0.05, 0.2) * m0 for _ in range(4)]
    ps = [mag * _isotropic_unit(rng) for mag in mags]
    p5 = -sum(ps)
    masses = [m1, m2, m3, m4, m5]
    pvecs = ps + [p5]
    return {
        i + 1: np.array([*p, np.sqrt(m**2 + np.dot(p, p))])
        for i, (p, m) in enumerate(zip(pvecs, masses))
    }


def add_batch_dim(momenta):
    """Add a batch dimension so momenta have shape (1, 4)."""
    return {k: v[np.newaxis, :] for k, v in momenta.items()}


# ── Wigner theta scan ─────────────────────────────────────────────────────────


def scan_wigner_angles(
    make_momenta_fn,
    topo_collection,
    topo_idx_a,
    topo_idx_b,
    massless_particle,
    mass_values,
    convention="helicity",
):
    """
    For each value in mass_values (applied to massless_particle), compute the
    relative Wigner angles between topo_idx_a and topo_idx_b.
    Returns (thetas, phi_plus_psi) arrays.
    """
    thetas = []
    phi_plus_psis = []
    topos = topo_collection.topologies
    topo_a = topos[topo_idx_a]
    topo_b = topos[topo_idx_b]

    for m in mass_values:
        momenta = add_batch_dim(make_momenta_fn(m))
        momenta_rest = topo_a.to_rest_frame(momenta)
        massless = [massless_particle] if m < 1e-3 else []
        wa = topo_a.relative_wigner_angles(
            topo_b, momenta_rest, convention=convention, massless=massless
        )
        w = wa[massless_particle]
        thetas.append(float(np.array(w.theta_rf).flat[0]))
        phi_plus_psis.append(
            float(np.array(w.phi_rf).flat[0]) + float(np.array(w.psi_rf).flat[0])
        )

    # Exact massless limit: m=0, massless flag set
    momenta = add_batch_dim(make_momenta_fn(0.0))
    momenta_rest = topo_a.to_rest_frame(momenta)
    wa = topo_a.relative_wigner_angles(
        topo_b, momenta_rest, convention=convention, massless=[massless_particle]
    )
    w = wa[massless_particle]
    theta_exact = float(np.array(w.theta_rf).flat[0])
    pps_exact = float(np.array(w.phi_rf).flat[0]) + float(np.array(w.psi_rf).flat[0])

    return np.array(thetas), np.array(phi_plus_psis), theta_exact, pps_exact


def scan_all_conventions(
    make_momenta_fn,
    topo_collection,
    topo_idx_a,
    topo_idx_b,
    massless_particle,
    mass_values,
):
    helicity = scan_wigner_angles(
        make_momenta_fn,
        topo_collection,
        topo_idx_a,
        topo_idx_b,
        massless_particle,
        mass_values,
        convention="helicity",
    )
    return (helicity,)  # tuple of (thetas, pps, theta_exact, pps_exact)


# ── Setup decays ──────────────────────────────────────────────────────────────

M0_3 = 5.0
M0_4 = 6.0
M0_5 = 7.0

# Masses for non-scanned particles
M2_3, M3_3 = 0.5, 1.0
M2_4, M3_4, M4_4 = 0.5, 0.8, 1.0
M2_5, M3_5, M4_5, M5_5 = 0.4, 0.6, 0.8, 1.0

# Mass range for particle 1 (driven toward zero), log-spaced for uniform density
mass_values = np.logspace(0, -5, 100)

tg3 = TopologyCollection(0, [1, 2, 3])
tg4 = TopologyCollection(0, [1, 2, 3, 4])
tg5 = TopologyCollection(0, [1, 2, 3, 4, 5])


def make3(m1):
    return make_3body(M0_3, m1, M2_3, M3_3)


def make4(m1):
    return make_4body(M0_4, m1, M2_4, M3_4, M4_4)


def make5(m1):
    return make_5body(M0_5, m1, M2_5, M3_5, M4_5, M5_5)


# ── Compute ───────────────────────────────────────────────────────────────────

print("Computing 3-body...")
((theta_3, pps_3, theta_3_exact, pps_3_exact),) = scan_all_conventions(
    make3, tg3, 0, 1, massless_particle=1, mass_values=mass_values
)

print("Computing 4-body...")
((theta_4, pps_4, theta_4_exact, pps_4_exact),) = scan_all_conventions(
    make4, tg4, 3, 6, massless_particle=1, mass_values=mass_values
)

print("Computing 5-body...")
((theta_5, pps_5, theta_5_exact, pps_5_exact),) = scan_all_conventions(
    make5, tg5, 18, 30, massless_particle=1, mass_values=mass_values
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def topo_latex(topo):
    """Convert a Topology repr to a LaTeX string."""
    s = repr(topo).replace("Topology: ", "").replace("->", r"\to").replace(",", "")
    return f"${s}$"


# ── Plot ──────────────────────────────────────────────────────────────────────

configs = [
    (
        3,
        M0_3,
        tg3.topologies[0],
        tg3.topologies[1],
        theta_3,
        pps_3,
        theta_3_exact,
        pps_3_exact,
        "3body",
    ),
    (
        4,
        M0_4,
        tg4.topologies[3],
        tg4.topologies[6],
        theta_4,
        pps_4,
        theta_4_exact,
        pps_4_exact,
        "4body",
    ),
    (
        5,
        M0_5,
        tg5.topologies[18],
        tg5.topologies[30],
        theta_5,
        pps_5,
        theta_5_exact,
        pps_5_exact,
        "5body",
    ),
]

for n, m0, ta, tb, thetas, pps, theta_exact, pps_exact, tag in configs:
    title = f"{topo_latex(ta)} vs {topo_latex(tb)}"

    for data, exact, ylabel, subtag in [
        (
            thetas / np.pi,
            theta_exact / np.pi,
            r"$\theta_{\rm Wigner}$ $[\pi]$",
            "theta",
        ),
        (pps / np.pi, pps_exact / np.pi, r"$(\phi + \psi)$ $[\pi]$", "phi_plus_psi"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(mass_values, data, "o-", ms=3, lw=1)
        ax.axhline(exact, color="C1", lw=1.5, ls="--", label=r"$m_1 = 0$ (exact)")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("$m_1$ [arbitrary units]", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = f"scripts/massless_limit_wigner_{tag}_{subtag}.png"
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
        plt.show()
