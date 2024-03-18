from decayangle.kinematics import build_4_4,  build_2_2
from decayangle.lorentz import LorentzTrafo
from jax import numpy as jnp
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
def test_lotentz( psi, theta, xi, theta_rf, phi_rf, psi_rf):
    M = build_4_4(psi, theta, xi, theta_rf, phi_rf, psi_rf)

    M2 = build_2_2(psi, theta, xi, theta_rf, phi_rf, psi_rf)
    trafo = LorentzTrafo(psi, theta, xi, theta_rf, phi_rf, psi_rf)
    psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_ = trafo.decode(False)
    print(psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_)

    psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_ = trafo.decode()

    assert np.allclose(M, build_4_4(psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_))

    assert np.allclose(psi, psi_)
    assert np.allclose(theta, theta_)
    assert np.allclose(xi, xi_)
    assert np.allclose(phi_rf, phi_rf_)
    assert np.allclose(theta_rf, theta_rf_)
    assert np.allclose(psi_rf, psi_rf_)

def test_lotentz2( psi, theta, xi, theta_rf, phi_rf, psi_rf):
    M = build_4_4(psi, theta, xi, theta_rf, phi_rf, psi_rf)

    # trafo = LorentzTrafo(psi, theta, xi, theta_rf, phi_rf, psi_rf) @ LorentzTrafo(psi, theta, xi, theta_rf, phi_rf, psi_rf).inverse()
    # psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_ = trafo.decode()
    trafo = LorentzTrafo(psi, theta, xi, theta_rf, phi_rf, psi_rf) @ LorentzTrafo(psi, theta, xi, theta_rf, phi_rf, psi_rf)
    psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_ = trafo.decode()




if __name__ == "__main__":
    for i in range(20):
        args = np.random.rand(6) * np.pi
        args[-1] = args[-1] + 2 * np.pi
        print(args)
        test_lotentz(*args)
        test_lotentz2(*args)
    test_lotentz(-1.8, 1.3, 1.4, 1.5, 2.6, -2.1 + 2* np.pi)