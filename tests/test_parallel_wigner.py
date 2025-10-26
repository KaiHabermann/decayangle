"""
Test file for parallel functionality in relative_wigner_angles.

Simple runtime comparison for different parallelization strategies.
"""

import time
import numpy as np
from decayangle.decay_topology import TopologyCollection
from decayangle.config import config as cfg
from jax import config as jax_cfg

# Enable 64-bit precision for JAX
jax_cfg.update("jax_enable_x64", True)

cb = cfg.backend


def generate_random_momenta_vectorized(m0, masses, n_events=100_000, seed=None):
    """Generate random 4-momenta for an n-body decay in the center-of-mass frame."""
    if seed is not None:
        np.random.seed(seed)

    n_particles = len(masses)

    # Generate random directions for isotropic distribution (vectorized)
    phi_angles = np.random.uniform(0, 2 * np.pi, (n_events, n_particles))
    cos_theta_angles = np.random.uniform(-1, 1, (n_events, n_particles))
    theta_angles = np.arccos(cos_theta_angles)

    # Random momentum magnitudes (scale with parent mass)
    p_magnitudes = np.random.uniform(0.1, 0.3, (n_events, n_particles)) * m0

    # Convert to Cartesian coordinates (vectorized)
    px = p_magnitudes * np.sin(theta_angles) * np.cos(phi_angles)
    py = p_magnitudes * np.sin(theta_angles) * np.sin(phi_angles)
    pz = p_magnitudes * np.cos(theta_angles)

    # Calculate energies from mass-shell condition (vectorized)
    p_squared = px**2 + py**2 + pz**2
    energies = np.sqrt(np.array(masses) ** 2 + p_squared)

    # Create 4-momenta (px, py, pz, E) - shape: (n_events, n_particles, 4)
    momenta_4d = np.stack([px, py, pz, energies], axis=-1)

    # Check momentum conservation and adjust (vectorized)
    total_px = np.sum(px, axis=1, keepdims=True)
    total_py = np.sum(py, axis=1, keepdims=True)
    total_pz = np.sum(pz, axis=1, keepdims=True)

    correction_x = -total_px / n_particles
    correction_y = -total_py / n_particles
    correction_z = -total_pz / n_particles

    # Apply correction
    momenta_4d[:, :, 0] += correction_x
    momenta_4d[:, :, 1] += correction_y
    momenta_4d[:, :, 2] += correction_z

    # Recalculate energies with corrected momenta
    p_squared_corrected = (
        momenta_4d[:, :, 0] ** 2 + momenta_4d[:, :, 1] ** 2 + momenta_4d[:, :, 2] ** 2
    )
    momenta_4d[:, :, 3] = np.sqrt(np.array(masses) ** 2 + p_squared_corrected)

    return {i + 1: cb.array(momenta_4d[:, i, :]) for i in range(n_particles)}


def test_runtime_comparison():
    """Simple runtime comparison for different parallelization strategies."""

    # Test configurations: (n_events, masses, description)
    test_configs = [
        (10_000, [200, 500, 800], "Small dataset (10k events, 3 particles)"),
        (100_000, [200, 500, 800, 1000], "Large dataset (100k events, 4 particles)"),
        (1_000_000, [200, 500, 800], "Very large dataset (1M events, 3 particles)"),
    ]

    m0 = 5000  # Parent particle mass

    for n_events, masses, description in test_configs:
        print(f"\n{description}")
        print("=" * len(description))

        # Generate momenta
        momenta = generate_random_momenta_vectorized(
            m0, masses, n_events=n_events, seed=42
        )

        # Create topologies
        tg = TopologyCollection(0, list(range(1, len(masses) + 1)))
        topology1 = tg.topologies[0]
        topology2 = tg.topologies[1]

        # Transform to rest frame
        momenta_rest = topology1.to_rest_frame(momenta)

        # Test different strategies
        strategies = [
            ("Sequential", None, None),
            ("2 cores", 2, None),
            ("4 cores", 4, None),
            ("8 cores", 8, None),
            ("4 cores + 25k chunks", 4, 25_000),
            ("8 cores + 10k chunks", 8, 10_000),
        ]

        wigner_results = []
        helicity_results = []

        for strategy_name, cores, chunk_size in strategies:
            # Test Wigner angles
            start_time = time.time()
            topology1.relative_wigner_angles(
                topology2, momenta_rest, parallel_cores=cores, chunk_size=chunk_size
            )
            wigner_time = time.time() - start_time
            wigner_results.append((strategy_name, wigner_time))

            # Test helicity angles
            start_time = time.time()
            topology1.helicity_angles(
                momenta_rest, parallel_cores=cores, chunk_size=chunk_size
            )
            helicity_time = time.time() - start_time
            helicity_results.append((strategy_name, helicity_time))

        # Print results
        print("Wigner Angles:")
        sequential_wigner_time = wigner_results[0][1]
        print(f"  Sequential: {sequential_wigner_time:.3f}s")

        for strategy_name, elapsed_time in wigner_results[1:]:
            speedup = sequential_wigner_time / elapsed_time
            print(f"  {strategy_name}: {elapsed_time:.3f}s (speedup: {speedup:.2f}x)")

        print("\nHelicity Angles:")
        sequential_helicity_time = helicity_results[0][1]
        print(f"  Sequential: {sequential_helicity_time:.3f}s")

        for strategy_name, elapsed_time in helicity_results[1:]:
            speedup = sequential_helicity_time / elapsed_time
            print(f"  {strategy_name}: {elapsed_time:.3f}s (speedup: {speedup:.2f}x)")


if __name__ == "__main__":
    test_runtime_comparison()
