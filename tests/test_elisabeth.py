from decayangle.decay_topology import Topology, TopologyCollection, HelicityAngles
from decayangle.lorentz import LorentzTrafo
from decayangle.config import config as decayangle_config
import numpy as np

def make_numpy(f):
    def wrapper(*args, **kwargs):
        args = [np.array(arg) for arg in args]
        kwargs = {k: np.array(v) for k, v in kwargs.items()}
        return f(*args, **kwargs)
    return wrapper

@make_numpy
def make_four_vectors_from_dict(mkpisq, mkpsq, mppisq, phip, thetap, chi, phi_Kst = None, theta_Kst=None, phi_K=None, theta_K=None):
    import numpy as np


    # Make sure, the sorting is turned off

    # Given values
    # Lc -> p K pi
    # 0 -> 1 2 3
    m12 = mkpsq**0.5
    m23 = mkpisq**0.5
    m13 = mppisq**0.5
    m1, m2, m3 = 0.938272, 0.493677, 0.1395704
    m0 = ((mkpisq + mkpsq + mppisq) - m1**2  - m2**2 - m3**2)**0.5

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
    # momenta are now in x-z plane

    phip = -np.pi + phip 

    thetap = np.pi - thetap
    chi = -np.pi + chi
    # rotation = LorentzTrafo(0, 0, 0, phip, thetap, chi)
    rotation = LorentzTrafo(0, 0, 0, phi_Kst, theta_Kst, phi_K)

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


# Lc -> p K pi
# 0 -> 1 2 3

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
                dtc[name][phi_hat],
                dtc[name][theta_hat],
            ),
            tpl[0]: HelicityAngles(
                dtc[name][phi],
                dtc[name][theta],
            ),
        }
    return topos

def report(a, b, name=None):
    print(f"{name}: decayangle {(a):.3f} Elisabeth {(b):.3f} Diff {(a - b):.3f} Sum {(a + b):.3f} DTypes {type(a)} {type(b)}")


def test_elisabeth():
    import json

    path = "tests/test_data/Parsed_ccp_kinematics_100events.json"
    with open(path, "r") as f:
        data = json.load(f)
    for k, dtc in data.items():
        kwargs = {k: v for k, v in dtc["kinematic"].items() if k != "mkpisq" }
        momenta = make_four_vectors_from_dict(**dtc["chain_variables"]["Kpi"], **kwargs)
        angles_from_json = read_helicity_angles_from_dict(dtc["chain_variables"])
        for topo_tuple, read_hel_angles in angles_from_json.items():
            topology = Topology(0, decay_topology=topo_tuple)
            helicity_angles = topology.helicity_angles(momenta=momenta)
            for decay in helicity_angles:
                # print("---", decay)
                # report(helicity_angles[decay].theta_rf, read_hel_angles[decay].theta_rf, "Theta")
                # report(helicity_angles[decay].phi_rf, read_hel_angles[decay].phi_rf, "Phi")
                assert np.isclose(
                    helicity_angles[decay].theta_rf, read_hel_angles[decay].theta_rf, atol=1e-3
                )
                assert np.isclose(
                    helicity_angles[decay].phi_rf, read_hel_angles[decay].phi_rf, atol=1e-3
                )
        # exit(0)
    
    # phi_Kst = np.linspace(1e-5, 2 * np.pi - 1e-5, 30)
    # theta_Kst = np.linspace(1e-5,  np.pi - 1e-5, 30)

    # theta_Kst, phi_Kst = np.meshgrid(theta_Kst, phi_Kst)
    np.random.seed(0)
    theta_Kst = np.random.uniform(0, np.pi, 10000)
    phi_Kst = np.random.uniform(0, 2 * np.pi, 10000)
    phi_K = np.random.uniform(0, 2 * np.pi, 10000)
    grid = {
    "kinematic": {
      "mkpisq": 0.5561674682091109,
      "mkpsq": 3.0833222564754488,
      "mppisq": 2.7319599271154402,
      "phip": 1.3378497917393044,
      "chi": -0.6240548021534527,
      "thetap": 1.353485369031088
    },
    "chain_variables": {
      "Kpi": {
        "mkpisq": 0.5561674682091109,
        "theta_Kst":theta_Kst,
        "phi_Kst": phi_Kst,
        "theta_K": 2.5975969431839845,
        "phi_K": phi_K
      },
      "pip": {
        "mppisq": 2.7319599271154402,
        "theta_D": 1.14023558672871,
        "phi_D": 1.5086313414356933,
        "theta_pi": 2.761208621409753,
        "phi_pi": 2.462748025368235
      },
      "pK": {
        "mkpsq": 3.0833222564754488,
        "theta_L": 1.5166647025202817,
        "phi_L": 1.2214893641013873,
        "theta_p": 0.31622082220551684,
        "phi_p": 2.5333520669181233
      }
    }
  }
    kwargs = {k: v for k, v in grid["kinematic"].items() if k != "mkpisq" }
    momenta = make_four_vectors_from_dict(**grid["chain_variables"]["Kpi"], **kwargs)
    reference = Topology(0, decay_topology=((2, 3), 1))
    wigner_rotation = reference.relative_wigner_angles(
        Topology(0, decay_topology=((3, 1), 2)), momenta=momenta
    )
    # Lc -> p K pi
    # 0  -> 1 2 3
    import matplotlib.pyplot as plt
    phi_p = Topology(0, decay_topology=((2, 3), 1)).helicity_angles(momenta=momenta)[((2, 3), 1)].phi_rf 
    phi_k = Topology(0, decay_topology=((3, 1), 2)).helicity_angles(momenta=momenta)[((3, 1), 2)].phi_rf
    rotation = wigner_rotation[1]
    color = np.fmod(rotation.phi_rf + rotation.psi_rf + 4 * np.pi, 4*np.pi)/np.pi
    color_restricted = np.fmod(color.copy(), 4)
    color_restricted[phi_p - phi_k > np.pi] = color_restricted[phi_p - phi_k > np.pi] - 2
    color_restricted[phi_p - phi_k < -np.pi] = color_restricted[phi_p - phi_k < -np.pi] - 2
    color_restricted = color_restricted % 4

    plt.scatter(phi_k, phi_p, c=color_restricted, cmap="viridis", s=1)
    # phi_p - phi_k = pi
    y = np.pi + np.linspace(-np.pi, np.pi, 1000)
    y[y > np.pi] = y[y > np.pi] - 2 * np.pi
    plt.plot(np.linspace(-np.pi, np.pi, 1000), y, c="red")
    plt.xlabel(r"$\phi_{K}$")
    plt.ylabel(r"$\phi_{p}$")
    plt.colorbar()
    plt.savefig("test.png")
    exit()
    for particle, rotation in wigner_rotation.items():
        # R_y(phi) R_z(theta) R_y(psi)
        # plt.imshow(np.fmod(rotation.phi_rf + rotation.psi_rf + 4 * np.pi, 4*np.pi), extent=[0, 2 * np.pi, 0, np.pi], origin="lower")
        # plt.scatter(rotation.phi_rf, rotation.theta_rf, c=np.fmod(rotation.phi_rf + rotation.psi_rf + 4 * np.pi, 4*np.pi), cmap="viridis")
        plt.ylabel(r"$\theta_{p}$")
        plt.xlabel(r"$\phi_{K^*}$")
        plt.colorbar()
        plt.savefig(f"tests/test_data/phi_rf_{particle}.png")
        plt.close()



if __name__ == "__main__":
    test_elisabeth()
