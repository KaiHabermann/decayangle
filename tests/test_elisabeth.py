from decayangle.decay_topology import Topology, TopologyCollection, HelicityAngles
from decayangle.lorentz import LorentzTrafo
from decayangle.config import config as decayangle_config
import numpy as np


def make_four_vectors_from_dict(mkpisq, mkpsq, mppisq, phip, thetap, chi):
    import numpy as np

    # Make sure, the sorting is turned off

    # Given values
    # Lc -> p K pi
    m0 = 2.286
    m12 = mkpsq**0.5
    m23 = mkpisq**0.5
    m13 = mppisq**0.5
    m1, m2, m3 = 0.938, 0.495, 0.139
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

    # Vectors such that we align with proton momentum
    p1 = np.array([0, 0, p1z, E1])
    p2 = np.array([p2x, 0, p2z, E2])
    p3 = np.array([p3x, 0, p3z, E3])

    # Lorentz transformation
    momenta = {i: p for i, p in zip([1, 2, 3], [p1, p2, p3])}
    tree1 = Topology(root=0, decay_topology=((2, 3), 1))

    rotation = LorentzTrafo(0, 0, 0, -phip, -thetap, -chi)

    momenta_23_rotated = tree1.root.transform(rotation, momenta)
    return momenta_23_rotated


decayangle_config.sorting = "off"

tg = TopologyCollection(
    0,
    topologies=[
        Topology(0, decay_topology=((2, 3), 1)),
        Topology(0, decay_topology=((3, 1), 2)),
        Topology(0, decay_topology=((1, 2), 3)),
    ],
)


def read_helicity_angles_from_dict(dtc):
    mappings = {
        ((2, 3), 1): ("Kpi", "theta_Kst", "phi_Kst", "theta_K", "phi_K"),
        ((3, 1), 2): ("pip", "theta_D", "phi_D", "theta_pi", "phi_pi"),
        ((1, 2), 3): ("pK", "theta_L", "phi_L", "theta_p", "phi_p"),
    }

    topos = {}

    for tpl, (name, theta_hat, phi_hat, theta, phi) in mappings.items():
        topos[tpl] = {
            tpl: HelicityAngles(
                dtc[name][theta_hat],
                dtc[name][phi_hat],
            ),
            tpl[0]: HelicityAngles(
                dtc[name][theta],
                dtc[name][phi],
            ),
        }
    return topos


def test_elisabeth():
    import json

    path = "tests/test_data/Parsed_ccp_kinematics_100events.json"
    with open(path, "r") as f:
        data = json.load(f)
    for k, dtc in data.items():
        momenta = make_four_vectors_from_dict(**dtc["kinematic"])
        angles_from_json = read_helicity_angles_from_dict(dtc["chain_variables"])
        for topo_tuple, read_hel_angles in angles_from_json.items():
            topology = Topology(0, decay_topology=topo_tuple)
            helicity_angles = topology.helicity_angles(momenta=momenta)
            for decay in helicity_angles:
                print(helicity_angles[decay].theta_rf, read_hel_angles[decay].theta_rf)
                print(helicity_angles[decay].theta_rf - read_hel_angles[decay].theta_rf)
                # assert np.isclose(
                #     helicity_angles[decay].theta_rf, read_hel_angles[decay].theta_rf
                # )
                # assert np.isclose(
                #     helicity_angles[decay].phi_rf, read_hel_angles[decay].phi_rf
                # )


if __name__ == "__main__":
    test_elisabeth()
