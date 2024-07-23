from decayangle.decay_topology import Topology, Node
from decayangle.lorentz import LorentzTrafo
from decayangle.kinematics import boost_to_rest
from decayangle.config import config as cfg
import numpy as np


def test_particle2():
    cfg.sorting = "off"
    root = 0
    topo1 = Topology(root, (1, (2, 3)))
    topo2 = Topology(root, (1, (3, 2)))
    print("Topology 1:", topo1)
    print("Topology 2:", topo2)

    momenta = {
        1: np.array([0.0, 0.0, -0.4373593403089831, 1.035197462060021]),
        2: np.array([0.466794284860449, 0.0, 0.1935604618890383, 0.7064556158132482]),
        3: np.array([-0.466794284860449, 0.0, 0.2437988784199448, 0.5448069221267302]),
    }

    rotation = LorentzTrafo(0, 0, 0, 1.0, 1.0, 1.0)
    momenta = {i: rotation.matrix_4x4 @ p for i, p in momenta.items()}
    wigner_angles = topo1.relative_wigner_angles(topo2, momenta)

    for particle in [1, 2, 3]:
        # this will always be 0, since the chain of boosts and rotations into the particle i rest frame is order independent
        # this is, because the final state frame is determined by the boost axis from the second to last frame into the last frame. This is not ordering dependent.
        print(f"Particle {particle}:", wigner_angles[particle])
        assert np.allclose(wigner_angles[particle].theta_rf, 0)
        assert np.allclose(wigner_angles[particle].psi_rf, 0)

    hel1 = topo1.helicity_angles(momenta)
    hel2 = topo2.helicity_angles(momenta)

    hel1_m_phi = topo1.helicity_angles(momenta, convention="minus_phi")
    hel2_m_phi = topo2.helicity_angles(momenta, convention="minus_phi")

    print(hel1[(2, 3)], hel2[(3, 2)])
    print(hel1_m_phi[(2, 3)], hel2_m_phi[(3, 2)])

    cfg.sorting = "value"


if __name__ == "__main__":
    test_particle2()
