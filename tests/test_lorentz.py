from decayangle.kinematics import build_4_4,  build_2_2
from decayangle.lorentz import LorentzTrafo
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

        assert np.allclose(
            (LorentzTrafo(*definition1) @ LorentzTrafo(*definition2)).inverse().M4,
            (LorentzTrafo(*definition2).inverse() @ LorentzTrafo(*definition1).inverse()).M4
        )

    for i in range(len(boost_definitions) - 1):
        test_single(boost_definitions[i], boost_definitions[i+1])

if __name__ == "__main__":
    pass