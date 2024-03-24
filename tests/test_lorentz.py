from decayangle.kinematics import build_4_4,  build_2_2, from_mass_and_momentum, mass_squared
from decayangle.lorentz import LorentzTrafo
from decayangle.DecayTopology import TopologyGroup
from jax import numpy as jnp
import jax
import numpy as np
import pytest
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def boost_definitions():
    definitions = []
    for i in range(20):
        args = np.random.rand(6) * np.pi
        args[-1] = args[-1] + 2 * np.pi
        definitions.append(args)
    return definitions

def test_lotentz(boost_definitions):
    def single_test(psi, theta, xi, theta_rf, phi_rf, psi_rf):
        M = build_4_4(psi, theta, xi, theta_rf, phi_rf, psi_rf)
        M2 = build_2_2(psi, theta, xi, theta_rf, phi_rf, psi_rf)
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
            (LorentzTrafo(*definition1) @ LorentzTrafo(*definition2)).inverse().M4,
            (LorentzTrafo(*definition2).inverse() @ LorentzTrafo(*definition1).inverse()).M4
        )

    for i in range(len(boost_definitions) - 1):
        test_single(boost_definitions[i], boost_definitions[i+1])

def test_lorentz_threeBody():
    def Kallen(x, y, z):
        return x**2 + y**2 + z**2 - 2*(x*y + x*z + y*z)
    def ijk(k):
        """Returns the indices based on k, adapted for Python's 0-based indexing."""
        return [(k + 1) % 3, (k + 2) % 3, k]

    def cos_zeta_31_for2(msq, sigmas):
        """
        Calculate the cosine of the ζ angle for the case where k=2.
        
        rotates frame 3 into frame 1 for particle 2

        :param msq: List containing squared masses, with msq[-1] being m02
        :param sigmas: List containing sigma values, adjusted for Python.
        """
        # Adjusted indices for k=2, directly applicable without further modification
        i, j, k = ijk(2)
        
        s = msq[0]  # s is the first element, acting as a placeholder in this context
        EE4m1sq = (sigmas[i] - msq[j] - msq[k]) * (sigmas[j] - msq[k] - msq[i])
        pp4m1sq = (Kallen(sigmas[i], msq[j], msq[k]) * Kallen(sigmas[j], msq[k], msq[i]))**0.5
        rest = msq[i] + msq[j] - sigmas[k]
    
        return (2*msq[k] * rest + EE4m1sq) / pp4m1sq

    momenta = np.random.rand(3, 3)
    masses = np.array([1, 2, 3])
    momenta = {
        i+1: from_mass_and_momentum(mass, momentum) 
        for i, (mass, momentum) in enumerate(zip(masses, momenta))
    }

    sigmas = [
        mass_squared(momenta[2] + momenta[3]),
        mass_squared(momenta[1] + momenta[3]),
        mass_squared(momenta[1] + momenta[2])
    ]
    mothermass2 = mass_squared(momenta[1] + momenta[2] + momenta[3])
    assert abs(sum(sigmas) - sum(masses**2) - mothermass2) < 1e-10
    tg = TopologyGroup(0, [1,2,3])
    momenta = tg.trees[0].to_rest_frame(momenta)
    reference_frame, = tg.filter((2,3))
    isobars = {
        1: (2,3),
        2: (1,3),
        3: (1,2)
    }
    for k, isobar in isobars.items():
        frame, = tg.filter(isobar)
        print(f"Rotation of {frame} into {reference_frame}")
        for node in [1, 2, 3]:
            theta, phi = reference_frame.relative_wigner_angles(frame, node, momenta)

    frame3, = tg.filter((1,2))
    theta, phi = reference_frame.relative_wigner_angles(frame3, 2, momenta)
    print(np.cos(phi), np.cos(theta))
    print(cos_zeta_31_for2([m**2 for m in masses] + [mothermass2] , sigmas))

if __name__ == "__main__":
    test_lorentz_threeBody()