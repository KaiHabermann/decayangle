from decayangle.kinematics import (
    build_4_4,
    build_2_2,
    from_mass_and_momentum,
    mass_squared,
)
from decayangle.lorentz import LorentzTrafo
from decayangle.decay_topology import TopologyCollection, Topology
from decayangle.config import config as cfg
from jax import numpy as jnp
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def boost_definitions():
    np.random.seed(123456)
    definitions = []
    for i in range(20):
        args = np.random.rand(6) * np.pi
        definitions.append(args.copy())
        args[-3] = args[-3] + 2 * np.pi
        definitions.append(args)
    return definitions


def test_lotentz(boost_definitions):
    def single_test(psi, theta, xi, theta_rf, phi_rf, psi_rf):
        M = build_4_4(psi, theta, xi, theta_rf, phi_rf, psi_rf)
        matrix_2x2 = build_2_2(psi, theta, xi, theta_rf, phi_rf, psi_rf)
        trafo = LorentzTrafo(psi, theta, xi, theta_rf, phi_rf, psi_rf)
        psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_ = trafo.decode()
        assert np.allclose(M, build_4_4(psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_))
        assert np.allclose(psi, psi_)
        assert np.allclose(theta, theta_)
        assert np.allclose(xi, xi_)
        assert np.allclose(phi_rf, phi_rf_)
        assert np.allclose(theta_rf, theta_rf_)
        assert np.allclose(psi_rf, psi_rf_)

    for boost_definition in boost_definitions:
        single_test(*boost_definition)


def test_lotentz2(boost_definitions):

    def test_single(definition1, definition2):
        trafo = LorentzTrafo(*definition1) @ LorentzTrafo(*definition2)
        psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_ = trafo.decode()
        assert np.isfinite(psi_)
        assert np.isfinite(theta_)
        assert np.isfinite(xi_)
        assert np.isfinite(phi_rf_)
        assert np.isfinite(theta_rf_)
        assert np.isfinite(psi_rf_)

        trafo = LorentzTrafo(*definition1) @ LorentzTrafo(*definition2).inverse()
        psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_ = trafo.decode()
        assert np.isfinite(psi_)
        assert np.isfinite(theta_)
        assert np.isfinite(xi_)
        assert np.isfinite(phi_rf_)
        assert np.isfinite(theta_rf_)
        assert np.isfinite(psi_rf_)

        assert np.allclose(
            (LorentzTrafo(*definition1) @ LorentzTrafo(*definition2))
            .inverse()
            .matrix_4x4,
            (
                LorentzTrafo(*definition2).inverse()
                @ LorentzTrafo(*definition1).inverse()
            ).matrix_4x4,
        )

    for i in range(len(boost_definitions) - 1):
        test_single(boost_definitions[i], boost_definitions[i + 1])


def test_lotentz_jax(boost_definitions):

    def test_single(definition1, definition2):
        trafo = LorentzTrafo(*definition1) @ LorentzTrafo(*definition2)
        psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_ = trafo.decode()
        assert np.isfinite(psi_)
        assert np.isfinite(theta_)
        assert np.isfinite(xi_)
        assert np.isfinite(phi_rf_)
        assert np.isfinite(theta_rf_)
        assert np.isfinite(psi_rf_)

        trafo = LorentzTrafo(*definition1) @ LorentzTrafo(*definition2).inverse()
        psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_ = trafo.decode()
        assert np.isfinite(psi_)
        assert np.isfinite(theta_)
        assert np.isfinite(xi_)
        assert np.isfinite(phi_rf_)
        assert np.isfinite(theta_rf_)
        assert np.isfinite(psi_rf_)

        assert np.allclose(
            (LorentzTrafo(*definition1) @ LorentzTrafo(*definition2))
            .inverse()
            .matrix_4x4,
            (
                LorentzTrafo(*definition2).inverse()
                @ LorentzTrafo(*definition1).inverse()
            ).matrix_4x4,
        )

    cfg.backend = "jax"
    for i in range(len(boost_definitions) - 1):
        test_single(boost_definitions[i], boost_definitions[i + 1])
    cfg.backend = "numpy"


@pytest.mark.parametrize(
    "momenta", [np.random.rand(3, 3)] + [np.random.rand(3, 100, 3) for _ in range(3)]
)
def test_daltiz_plot_decomposition(momenta):

    def Kallen(x, y, z):
        return x**2 + y**2 + z**2 - 2 * (x * y + x * z + y * z)

    def ijk(k):
        """Returns the indices based on k, adapted for Python's 0-based indexing."""
        return [(k + 1) % 3, (k + 2) % 3, k]

    def cos_theta_12(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return (
            2 * sigma3 * (sigma2 - m3**2 - m1**2)
            - (sigma3 + m1**2 - m2**2) * (M**2 - sigma3 - m3**2)
        ) / (Kallen(M**2, m3**2, sigma3) * Kallen(sigma3, m1**2, m2**2)) ** 0.5

    def cos_theta_23(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return cos_theta_12(M, m2, m3, m1, sigma2, sigma3, sigma1)

    def cos_theta_31(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return cos_theta_12(M, m3, m1, m2, sigma3, sigma1, sigma2)

    def cos_zeta_31_for2(msq, sigmas):
        """
        Calculate the cosine of the ζ angle for the case where k=2.

        rotates topology 3 into topology 1 for particle 2

        :param msq: List containing squared masses, with msq[-1] being m02
        :param sigmas: List containing sigma values, adjusted for Python.
        """
        # Adjusted indices for k=2, directly applicable without further modification
        i, j, k = ijk(2)

        s = msq[0]  # s is the first element, acting as a placeholder in this context
        EE4m1sq = (sigmas[i] - msq[j] - msq[k]) * (sigmas[j] - msq[k] - msq[i])
        pp4m1sq = (
            Kallen(sigmas[i], msq[j], msq[k]) * Kallen(sigmas[j], msq[k], msq[i])
        ) ** 0.5
        rest = msq[i] + msq[j] - sigmas[k]

        return (2 * msq[k] * rest + EE4m1sq) / pp4m1sq

    def cos_zeta_1_aligned_3_in_tree_1(M, m1, m2, m3, sigma1, sigma2, sigma3):
        """
        Calculate the cosine of the ζ angle for the case where k=1.
        The aligned topology is topology 3, the reference topology is topology 1.
        i.e. we rotate topology 3 into topology 1 for particle 1.
        """
        return (
            2 * m1**2 * (sigma2 - M**2 - m2**2)
            + (M**2 + m1**2 - sigma1) * (sigma3 - m1**2 - m2**2)
        ) / (Kallen(M**2, m1**2, sigma1) * Kallen(sigma3, m1**2, m2**2)) ** 0.5

    def cos_zeta_1_aligned_1_in_tree_2(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return cos_zeta_1_aligned_3_in_tree_1(M, m1, m3, m2, sigma1, sigma3, sigma2)

    def cos_zeta_2_aligned_1_in_tree_2(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return cos_zeta_1_aligned_3_in_tree_1(M, m2, m3, m1, sigma2, sigma3, sigma1)

    def cos_zeta_2_aligned_2_in_tree_3(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return cos_zeta_1_aligned_3_in_tree_1(M, m2, m1, m3, sigma2, sigma1, sigma3)

    def cos_zeta_3_aligned_2_in_tree_3(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return cos_zeta_1_aligned_3_in_tree_1(M, m3, m1, m2, sigma3, sigma1, sigma2)

    def cos_zeta_3_aligned_3_in_tree_1(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return cos_zeta_1_aligned_3_in_tree_1(M, m3, m2, m1, sigma3, sigma2, sigma1)

    def cos_theta_hat_3_canonical_1(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return (
            (M**2 + m3**2 - sigma3) * (M**2 + m1**2 - sigma1)
            - 2 * M**2 * (sigma2 - m3**2 - m1**2)
        ) / (Kallen(M**2, m1**2, sigma1) * Kallen(M**2, sigma3, m3**2)) ** 0.5

    def cos_theta_hat_1_canonical_1(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return 1

    def cos_theta_hat_2_canonical_2(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return 1

    def cos_theta_hat_3_canonical_3(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return 1

    def cos_theta_hat_1_canonical_2(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return cos_theta_hat_3_canonical_1(M, m2, m3, m1, sigma2, sigma3, sigma1)

    def cos_theta_hat_2_canonical_3(M, m1, m2, m3, sigma1, sigma2, sigma3):
        return cos_theta_hat_3_canonical_1(M, m3, m1, m2, sigma3, sigma1, sigma2)

    np.random.seed(0)
    # momenta = np.random.rand(3, 2, 3)
    masses = np.array([1, 2, 3])
    momenta = {
        i + 1: from_mass_and_momentum(mass, momentum)
        for i, (mass, momentum) in enumerate(zip(masses, momenta))
    }

    sigmas = [
        mass_squared(momenta[2] + momenta[3]),
        mass_squared(momenta[1] + momenta[3]),
        mass_squared(momenta[1] + momenta[2]),
    ]
    mothermass2 = mass_squared(momenta[1] + momenta[2] + momenta[3])
    assert np.all(abs(sum(sigmas) - sum(masses**2) - mothermass2) < 1e-10)
    tg = TopologyCollection(0, [1, 2, 3])
    momenta = tg.topologies[0].to_rest_frame(momenta)

    # topology 1 is the reference topology
    (reference_tree,) = tg.filter((2, 3))
    isobars = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    for k, isobar in isobars.items():
        (topology,) = tg.filter(isobar)
        # first simple test to check, that we can compute everything without exception
        args = reference_tree.relative_wigner_angles(topology, momenta)

    # we can simply filter for the isobars to get the topology we want
    (tree1,) = tg.filter((2, 3))
    (tree2,) = tg.filter((1, 3))
    (tree3,) = tg.filter((1, 2))
    phi_rf, theta_rf, psi_rf = tree2.relative_wigner_angles(tree1, momenta)[3]
    dpd_value = cos_zeta_31_for2([m**2 for m in masses] + [mothermass2], sigmas)
    assert np.allclose(np.cos(theta_rf), dpd_value)

    dpd_value = cos_zeta_1_aligned_3_in_tree_1(mothermass2**0.5, *masses, *sigmas)
    phi_rf, theta_rf, psi_rf = tree1.relative_wigner_angles(tree3, momenta)[1]
    assert np.allclose(np.cos(theta_rf), dpd_value)

    dpd_value = cos_zeta_1_aligned_1_in_tree_2(mothermass2**0.5, *masses, *sigmas)
    phi_rf, theta_rf, psi_rf = tree2.relative_wigner_angles(tree1, momenta)[1]
    assert np.allclose(np.cos(theta_rf), dpd_value)

    dpd_value = cos_zeta_2_aligned_1_in_tree_2(mothermass2**0.5, *masses, *sigmas)
    phi_rf, theta_rf, psi_rf = tree2.relative_wigner_angles(tree1, momenta)[2]
    assert np.allclose(np.cos(theta_rf), dpd_value)

    dpd_value = cos_zeta_2_aligned_2_in_tree_3(mothermass2**0.5, *masses, *sigmas)
    phi_rf, theta_rf, psi_rf = tree3.relative_wigner_angles(tree2, momenta)[2]
    assert np.allclose(np.cos(theta_rf), dpd_value)

    dpd_value = cos_zeta_3_aligned_2_in_tree_3(mothermass2**0.5, *masses, *sigmas)
    phi_rf, theta_rf, psi_rf = tree3.relative_wigner_angles(tree2, momenta)[3]
    assert np.allclose(np.cos(theta_rf), dpd_value)

    dpd_value = cos_zeta_3_aligned_3_in_tree_1(mothermass2**0.5, *masses, *sigmas)
    phi_rf, theta_rf, psi_rf = tree1.relative_wigner_angles(tree3, momenta)[3]
    phi_rf_, theta_rf_, psi_rf_ = tree3.relative_wigner_angles(tree1, momenta)[3]
    assert np.allclose(theta_rf, theta_rf_)
    assert np.allclose(np.cos(theta_rf), dpd_value)

    dpd_helicity_2 = cos_theta_12(mothermass2**0.5, *masses, *sigmas)
    hel_angles = tree3.helicity_angles(momenta)
    theta_rf, psi_rf = hel_angles[(1, 2)]
    assert np.allclose(dpd_helicity_2, np.cos(theta_rf))

    dpd_helicity_2 = cos_theta_31(mothermass2**0.5, *masses, *sigmas)
    hel_angles = tree2.helicity_angles(momenta)
    theta_rf, psi_rf = hel_angles[(1, 3)]
    # dpd defines the angle to particle 3, but we chose the angle to particle 1
    # so we need to invert the angle
    assert np.allclose(dpd_helicity_2, np.cos(np.pi - theta_rf))

    dpd_helicity_2 = cos_theta_23(mothermass2**0.5, *masses, *sigmas)
    hel_angles = tree1.helicity_angles(momenta)
    theta_rf, psi_rf = hel_angles[(2, 3)]
    assert np.allclose(dpd_helicity_2, np.cos(theta_rf))

    # we will now test the theta hat angles from dpd
    # the issue here is, that we will need specific aligned topologies for that
    # dpd aligns to the reference topology along the negative axis of the k-th particle
    # for tree1 this is particle 1 and the negative direction is the combined momentum of particle 2 and 3
    tree1_aligned_momenta = tree1.align_with_daughter(momenta, (2, 3))
    dpd_value = cos_theta_hat_3_canonical_1(mothermass2**0.5, *masses, *sigmas)
    theta_rf, psi_rf = tree3.helicity_angles(tree1_aligned_momenta)[((1, 2), 3)]
    assert np.allclose(dpd_value, np.cos(theta_rf))


def test_helicity_angles():
    import numpy as np

    # Make sure, the sorting is turned off
    cfg.sorting = "off"
    chain_vars = {
        "Kpi": {
            "mkpisq": 1.3743747462964881,
            "theta_Kst": 1.0468159811504423,
            "phi_Kst": 1.3921357860994747,
            "theta_K": 1.692234518478623,
            "phi_K": 0.5466265769529981,
        },
        "pK": {
            "mkpsq": 2.756020646168232,
            "theta_L": 2.6621627135924624,
            "phi_L": 3.010596711405015,
            "Ltheta_p": 1.861461784272743,
            "Lphi_p": 1.3499280354237881,
        },
        "pip": {
            "mppisq": 2.2410542593352796,
            "theta_D": 1.05685191949046,
            "phi_D": -1.1654157065810633,
            "theta_pi": 1.1669778470524175,
            "phi_pi": 2.5984404692152796,
        },
    }

    # Given values
    # Lc -> p K pi
    m1, m2, m3, m0 = 0.93827, 0.493677, 0.139570, 2.28646
    m12 = 2.756020646168232**0.5
    m23 = 1.3743747462964881**0.5
    m31 = 2.2410542593352796**0.5

    # Squared masses
    m0sq, m1sq, m2sq, m3sq, m12sq, m23sq = [x**2 for x in [m0, m1, m2, m3, m12, m23]]

    # Källén function
    def Kallen(x, y, z):
        return x**2 + y**2 + z**2 - 2 * (x * y + x * z + y * z)

    # Calculating missing mass squared using momentum conservation
    m31sq = m0sq + m1sq + m2sq + m3sq - m12sq - m23sq

    # Momenta magnitudes
    p1a = np.sqrt(Kallen(m23sq, m1sq, m0sq)) / (2 * m0)
    p2a = np.sqrt(Kallen(m31sq, m2sq, m0sq)) / (2 * m0)
    p3a = np.sqrt(Kallen(m12sq, m3sq, m0sq)) / (2 * m0)

    # Directions and components
    cos_zeta_12_for0_numerator = (m0sq + m1sq - m23sq) * (
        m0sq + m2sq - m31sq
    ) - 2 * m0sq * (m12sq - m1sq - m2sq)
    cos_zeta_12_for0_denominator = np.sqrt(Kallen(m0sq, m2sq, m31sq)) * np.sqrt(
        Kallen(m0sq, m23sq, m1sq)
    )
    cos_zeta_12_for0 = cos_zeta_12_for0_numerator / cos_zeta_12_for0_denominator

    p1z = -p1a
    p2z = -p2a * cos_zeta_12_for0
    p2x = np.sqrt(p2a**2 - p2z**2)
    p3z = -p2z - p1z
    p3x = -p2x

    # Energy calculations based on the relativistic energy-momentum relation
    E1 = np.sqrt(p1z**2 + m1sq)
    E2 = np.sqrt(p2z**2 + p2x**2 + m2sq)
    E3 = np.sqrt(p3z**2 + p3x**2 + m3sq)

    # Vectors
    p1 = np.array([0, 0, p1z, E1])
    p2 = np.array([p2x, 0, p2z, E2])
    p3 = np.array([p3x, 0, p3z, E3])

    # Mass squared function for a four-vector
    def masssq(p):
        return p[3] ** 2 - (p[0] ** 2 + p[1] ** 2 + p[2] ** 2)

    # Assertions to check calculations
    assert round(masssq(p1), 5) == round(m1sq, 5)
    assert round(masssq(p2), 5) == round(m2sq, 5)
    assert round(masssq(p3), 5) == round(m3sq, 5)
    assert round(masssq(p1 + p2), 5) == round(m12sq, 5)
    assert round(masssq(p2 + p3), 5) == round(m23sq, 5)
    assert round(masssq(p3 + p1), 5) == round(m31sq, 5)

    # Lorentz transformation
    momenta = {i: p for i, p in zip([1, 2, 3], [p1, p2, p3])}
    tree1 = Topology(root=0, decay_topology=((2, 3), 1))
    tree2 = Topology(root=0, decay_topology=((3, 1), 2))
    tree3 = Topology(root=0, decay_topology=((1, 2), 3))
    # momenta = tc.topologies[0].to_rest_frame(momenta)
    momenta = tree1.align_with_daughter(momenta, (2, 3))
    phi_rf = chain_vars["Kpi"]["phi_Kst"]
    theta_rf = chain_vars["Kpi"]["theta_Kst"]
    psi_rf = chain_vars["Kpi"]["phi_K"]

    rotation = LorentzTrafo(0, 0, 0, phi_rf, theta_rf, psi_rf)
    # direct outside rotations are not really supported, but possible via direct matrix multiplication or via the root node of a tree
    momenta = tree1.to_rest_frame(momenta)
    momenta_23_rotated = tree1.root.transform(rotation, momenta)
    assert np.allclose(
        np.cos(tree1.helicity_angles(momenta_23_rotated)[((2, 3), 1)].theta_rf),
        np.cos(chain_vars["Kpi"]["theta_Kst"]),
    )
    assert np.allclose(
        np.cos(tree1.helicity_angles(momenta_23_rotated)[((2, 3), 1)].psi_rf),
        np.cos(chain_vars["Kpi"]["phi_Kst"]),
    )
    assert np.allclose(
        np.cos(tree1.helicity_angles(momenta_23_rotated)[(2, 3)].psi_rf),
        np.cos(chain_vars["Kpi"]["phi_K"]),
    )
    assert np.allclose(
        np.cos(tree1.helicity_angles(momenta_23_rotated)[(2, 3)].theta_rf),
        np.cos(chain_vars["Kpi"]["theta_K"]),
        1e-4,
    )

    assert np.allclose(
        np.cos(tree2.helicity_angles(momenta_23_rotated)[((3, 1), 2)].theta_rf),
        np.cos(chain_vars["pip"]["theta_D"]),
    )
    assert np.allclose(
        np.cos(tree2.helicity_angles(momenta_23_rotated)[((3, 1), 2)].psi_rf),
        np.cos(chain_vars["pip"]["phi_D"]),
    )

    assert np.allclose(
        np.cos(tree2.helicity_angles(momenta_23_rotated)[(3, 1)].psi_rf),
        np.cos(chain_vars["pip"]["phi_pi"]),
    )
    assert np.allclose(
        np.cos(tree2.helicity_angles(momenta_23_rotated)[(3, 1)].theta_rf),
        np.cos(chain_vars["pip"]["theta_pi"]),
    )

    assert np.allclose(
        np.cos(tree3.helicity_angles(momenta_23_rotated)[((1, 2), 3)].theta_rf),
        np.cos(chain_vars["pK"]["theta_L"]),
    )
    assert np.allclose(
        np.cos(tree3.helicity_angles(momenta_23_rotated)[((1, 2), 3)].psi_rf),
        np.cos(chain_vars["pK"]["phi_L"]),
    )
    assert np.allclose(
        np.cos(tree3.helicity_angles(momenta_23_rotated)[(1, 2)].psi_rf),
        np.cos(chain_vars["pK"]["Lphi_p"]),
        1e-4,
    )
    assert np.allclose(
        np.cos(tree3.helicity_angles(momenta_23_rotated)[(1, 2)].theta_rf),
        np.cos(chain_vars["pK"]["Ltheta_p"]),
        1e-4,
    )


@pytest.mark.parametrize("momenta", [np.random.rand(4, 3)])
def test_conventions(momenta):
    tg = TopologyCollection(0, [1, 2, 3, 4])
    masses = np.array([1, 2, 3, 4])
    momenta = {
        i + 1: from_mass_and_momentum(mass, momentum)
        for i, (mass, momentum) in enumerate(zip(masses, momenta))
    }
    momenta = tg.topologies[0].to_rest_frame(momenta)
    reference_tree = tg.topologies[0]

    # topology 1 is the reference topology
    for topology in tg.topologies[1:]:
        # first simple test to check, that we can compute everything without exception
        args = reference_tree.relative_wigner_angles(topology, momenta)
        args_canonical = reference_tree.relative_wigner_angles(
            topology, momenta, convention="canonical"
        )
        args_minus_phi = reference_tree.relative_wigner_angles(
            topology, momenta, convention="minus_phi"
        )

        for (i, helicity), canonical, minus_phi in zip(
            args.items(), args_canonical.values(), args_minus_phi.values()
        ):
            if reference_tree.path_to(i)[0] == topology.path_to(i)[0]:
                # We may have different trees, but the relevant subtree is the same
                continue
            assert (
                not np.allclose(helicity.phi_rf, canonical.phi_rf)
                or not np.allclose(helicity.phi_rf, minus_phi.phi_rf)
                or not np.allclose(canonical.phi_rf, minus_phi.phi_rf)
                or not np.allclose(helicity.psi_rf, canonical.psi_rf)
                or not np.allclose(helicity.psi_rf, minus_phi.psi_rf)
                or not np.allclose(canonical.psi_rf, minus_phi.psi_rf)
                or not np.allclose(helicity.theta_rf, canonical.theta_rf)
                or not np.allclose(helicity.theta_rf, minus_phi.theta_rf)
                or not np.allclose(canonical.theta_rf, minus_phi.theta_rf)
            )


if __name__ == "__main__":
    # test_lotentz(boost_definitions())
    # test_lotentz2(boost_definitions())
    # test_helicity_angles()
    test_conventions(np.random.rand(4, 3))
