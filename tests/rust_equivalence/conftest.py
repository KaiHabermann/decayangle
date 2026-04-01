import numpy as np
import pytest

try:
    import decayangle_rs  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

skip_no_rust = pytest.mark.skipif(
    not RUST_AVAILABLE,
    reason="decayangle_rs extension not built — run `maturin develop --release` in decayangle-rs/",
)

N = 1_000


@pytest.fixture(scope="session")
def momenta_3body():
    """Random 3-body momenta in (approximately) the mother rest frame, shape (N, 4)."""
    rng = np.random.default_rng(0)
    p1 = np.column_stack(
        [
            rng.uniform(-0.5, 0.5, N),
            rng.uniform(-0.5, 0.5, N),
            rng.uniform(-0.9, 0.9, N),
            np.ones(N),
        ]
    )
    p2 = np.column_stack(
        [
            rng.uniform(-0.5, 0.5, N),
            rng.uniform(-0.5, 0.5, N),
            rng.uniform(-0.4, 0.4, N),
            np.ones(N),
        ]
    )
    p3 = np.column_stack(
        [
            -p1[:, 0] - p2[:, 0],
            -p1[:, 1] - p2[:, 1],
            -p1[:, 2] - p2[:, 2],
            np.ones(N),
        ]
    )
    return {1: p1, 2: p2, 3: p3}
