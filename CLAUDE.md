# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install for development:**

```bash
pip install -e .
# with notebook extras:
pip install -e ".[notebooks]"
```

**Run tests:**

```bash
pytest
```

**Run a single test file:**

```bash
pytest tests/test_decay_topology.py
```

**Run with coverage:**

```bash
pytest --cov-report=term-missing --cov-report=xml --cov-config=pyproject.toml --cov --cov=tests
```

No linting or formatting tools are configured in this project.

**Build the Rust extension** (requires `maturin` and a Rust toolchain):

```bash
cd decayangle-rs && maturin develop --release
```

**Run Rust equivalence tests** (after building the extension):

```bash
pytest tests/rust_equivalence/
```

## Architecture

All source code lives in `src/decayangle/`. The package is flat (no sub-packages).

### Module Responsibilities

- **`config.py`** — Global Borg-pattern singleton `cfg` controlling backend (`"numpy"` or `"jax"`), sorting behavior, numerical safety checks, and tolerances. State changes are global and persist across calls.
- **`backend.py`** — Thin shim exposing `jax_backend` and `numpy_backend`; modules import `cfg.backend` (aliased as `cb`) and call all math through it.
- **`kinematics.py`** — Low-level special-relativistic math: 4-vector operations, boost and rotation matrices in both 4×4 O(3,1) and 2×2 SU(2) representations, `boost_to_rest`.
- **`lorentz.py`** — `LorentzTrafo` class wrapping a Lorentz transformation parameterized by 6 angles (φ, θ, ξ, φ_rf, θ_rf, ψ_rf). Supports `@` composition, `.inverse()`, `.decode()`, `.wigner_angles()`. Also defines `WignerAngles` namedtuple.
- **`decay_topology.py`** — Main user-facing module with `Node` (binary tree node), `Topology` (full decay tree), and `TopologyCollection` (all topologies for an N-body decay).
- **`numerics_helpers.py`** — `save_arccos` and batched `matrix_vector_product`.

### Data Flow

1. User provides momenta as `Dict[int, np.ndarray(..., 4)]` — **energy/time is the last component (index 3)**, not first.
2. `topology.to_rest_frame(momenta)` boosts all momenta to the mother rest frame.
3. `topology.helicity_angles(momenta)` walks the decay tree, boosting to each intermediate rest frame and extracting angles at each vertex.
4. `topology1.relative_wigner_angles(topology2, momenta)` computes the product of `LorentzTrafo` objects along each topology's path to the same final-state particle, then decodes the three Wigner angles (φ, θ, ψ).

### Key Conventions

- **4-vector layout:** `[px, py, pz, E]` — energy is index 3 (non-standard for HEP).
- **Particle labels:** Final-state particles are non-negative integers ≤ 10000. Intermediate states are automatically named as sorted tuples of their constituent integers (e.g., `(1, 2)`).
- **Three boost conventions:** `"helicity"`, `"minus_phi"`, `"canonical"` — affects how the Wigner rotation is defined and must be consistent throughout a calculation.
- **Sorting:** `cfg.sorting = "value"` (default) sorts daughters with the longest tuple first. Set `cfg.sorting = "off"` to preserve input order and match DPD paper analytic conventions.
- **JAX 64-bit precision:** Tests requiring numerical accuracy must call `jax.config.update("jax_enable_x64", True)` before importing math functions. JAX defaults to 32-bit.
- **Config is global state:** The Borg-pattern config persists across function calls. Tests that change `cfg` must save and restore old values explicitly.
- **`test_equivalence.py`** pip-installs `sympy` at runtime and validates against symbolic Wigner D-matrix results — it is the most demanding test.
