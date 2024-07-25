from decayangle.decay_topology import Topology, Node
from decayangle.lorentz import LorentzTrafo
from decayangle.kinematics import boost_to_rest
from decayangle.config import config as cfg
import numpy as np


def test_particle2():
    cfg.sorting = "off"
    topo1 = Topology(0, ((2, 3), 1))
    topo2 = Topology(0, ((3, 2), 1))
    print("Topology 1:", topo1)
    print("Topology 2:", topo2)

    momenta = {
        1: np.array([0.0, 0.0, -0.4373593403089831, 1.035197462060021]),
        2: np.array([0.466794284860449, 0.0, 0.1935604618890383, 0.7064556158132482]),
        3: np.array([-0.466794284860449, 0.0, 0.2437988784199448, 0.5448069221267302]),
    }
    x, y = np.linspace(-np.pi + 1e-5, np.pi - 1e-5, 30), np.linspace(
        -np.pi + 1e-5, np.pi - 1e-5, 30
    )
    X, Y = np.meshgrid(x, y)
    rotation = LorentzTrafo(0, 0, 0, X, 1.0, Y)
    momenta = {i: rotation.matrix_4x4 @ p for i, p in momenta.items()}
    wigner_angles = topo1.relative_wigner_angles(topo2, momenta)

    hel1 = topo1.helicity_angles(momenta)
    hel2 = topo2.helicity_angles(momenta)

    assert np.allclose(hel1[(2, 3)].theta_rf, (np.pi - hel2[(3, 2)].theta_rf))

    assert np.allclose(
        hel1[(2, 3)].psi_rf - hel2[(3, 2)].psi_rf + wigner_angles[2].phi_rf, np.pi
    )

    cfg.sorting = "value"


if __name__ == "__main__":
    test_particle2()
