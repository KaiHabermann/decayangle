from typing import NamedTuple
import numpy as np
import subprocess
import sys
from functools import lru_cache
from decayangle.config import config as cfg
from decayangle.lorentz import LorentzTrafo
from decayangle.decay_topology import Topology, TopologyCollection

subprocess.check_call([sys.executable, "-m", "pip", "install", "sympy"])
from sympy import Rational
from sympy.abc import x
from sympy.utilities.lambdify import lambdify
from sympy.physics.quantum.spin import Rotation as Wigner
from sympy.physics.quantum.cg import CG
from sympy import Rational

cfg.sorting = "off"

cache = lru_cache(maxsize=None)


class J:
    """
    Helper class to ensure correctness of angular variable treatment
    """

    def __init__(self, j_times_2):
        self.j_times_2 = j_times_2

    @property
    def index(self):
        return self.j_times_2

    @property
    def value(self):
        return self.j_times_2 / 2

    @property
    def sympy(self):
        return Rational(self.j_times_2, 2)


def make_four_vectors(phi_rf, theta_rf, psi_rf):
    import numpy as np

    # Make sure, the sorting is turned off

    # Given values
    # Lc -> p K pi
    m0 = 6.32397
    m12 = 9.55283383**0.5
    m23 = 26.57159046**0.5
    m13 = 17.86811729**0.5
    m1, m2, m3 = 1, 2, 3
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

    # Lorentz transformation
    momenta = {i: p for i, p in zip([1, 2, 3], [p1, p2, p3])}
    tree1 = Topology(root=0, decay_topology=((2, 3), 1))

    # momenta = Topology(root=0, decay_topology=((1, 2), 3)).align_with_daughter(momenta, 3)
    # momenta = tree1.root.transform(LorentzTrafo(0, 0, 0, 0, -np.pi, 0), momenta)
    # print(momenta)
    rotation = LorentzTrafo(0, 0, 0, phi_rf, theta_rf, psi_rf)

    momenta_23_rotated = tree1.root.transform(rotation, momenta)
    return momenta_23_rotated


@cache
def get_wigner_function(j: J, m1: J, m2: J):
    j, m1, m2 = int(j), int(m1), int(m2)
    d = Wigner.d(Rational(j, 2), Rational(m1, 2), Rational(m2, 2), x).doit().evalf()
    d = lambdify(x, d, "numpy")
    return d


def wigner_small_d(theta, j, m1, m2):
    """Calculate Wigner small-d function. Needs sympy.
      theta : angle
      j : spin (in units of 1/2, e.g. 1 for spin=1/2)
      m1 and m2 : spin projections (in units of 1/2)

    :param theta:
    :param j:
    :param m1: before rotation
    :param m2: after rotation

    """
    d_func = get_wigner_function(j, m1, m2)
    d = d_func(theta)
    # d = np.array(d)
    # d[np.isnan(d)] = 0
    d = np.nan_to_num(d, copy=True, nan=0.0)
    return d


def wigner_capital_d(phi, theta, psi, j, m1, m2):
    return (
        np.exp(-1j * phi * m1 / 2)
        * wigner_small_d(theta, j, m1, m2)
        * np.exp(-1j * psi * m2 / 2)
    )


def BWResonance(spin, mass, width):
    """Create a Breit-Wigner resonance function for a given spin.
    Args:
        spin (int): spin quantum number multiplied by 2
    """

    def f(s, L):
        return 1
        return np.sqrt(s) / (s - mass**2 + 1j * mass * width)

    return f


class resonance:
    def __init__(self, spin, s0, si, sj, sk, LSin, LSout):
        self.spin = spin
        self.s0 = s0
        self.si = si
        self.sj = sj
        self.sk = sk
        self.LSin = LSin
        self.LSout = LSout
        self.lineshape = lambda *args: 1.0

    @property
    def possible_helicities(self):
        return list(range(-self.spin, self.spin + 1, 2))

    def LS_couplings_mother_decay(self):
        LS = NamedTuple("LS", [("L", int), ("S", int), ("coupling", complex)])
        return [LS(L, S, 1 + 0j) for L, S in self.LSin]

    def LS_coupling_resonance_decay(self):
        LS = NamedTuple("LS", [("L", int), ("S", int), ("coupling", complex)])
        return [LS(L, S, 1 + 0j) for L, S in self.LSout]

    @cache
    def clebsch_gordan(self, j1, m1, j2, m2, J, M):
        """
        Return clebsch-Gordan coefficient. Note that all arguments should be multiplied by 2
        (e.g. 1 for spin 1/2, 2 for spin 1 etc.). Needs sympy.
        """

        cg = (
            CG(
                Rational(j1, 2),
                Rational(m1, 2),
                Rational(j2, 2),
                Rational(m2, 2),
                Rational(J, 2),
                Rational(M, 2),
            )
            .doit()
            .evalf()
        )
        cg = float(cg)
        if str(cg) == "nan":
            raise ValueError(
                f"CG({j1/2},{m1/2},{j2/2},{m2/2},{J/2},{M/2}) is not a number"
            )
        return cg

    def helicity_coupling_times_lineshape(self, s, hi_, hj_, convention="helicity"):
        ls_resonance_decay = self.LS_coupling_resonance_decay()
        h = sum(
            ls.coupling
            * self.lineshape(s, ls.L)
            * self.clebsch_gordan(self.si, hi_, self.sj, -hj_, ls.S, hi_ - hj_)
            * self.clebsch_gordan(ls.L, 0, ls.S, hi_ - hj_, self.spin, hi_ - hj_)
            * (ls.L + 1) ** 0.5
            / (self.spin + 1) ** 0.5
            for ls in ls_resonance_decay
        )
        if convention == "minus_phi":
            # TODO insert phase difference here
            pass
        return h * (-1) ** ((self.sj - hj_) / 2)

    def h_mother(self, hk_, hiso_, convention="helicity") -> float:
        """
        Calculate the helicity amplitude for the mother particle decay.
        The phase is (-1)**(pk.spin - hk_) comes as a consequence of the definition of the two particle state.
        H = (-1)**(pk.spin - hk_) H_tilde
        H_tilde = sum_{ls} coupling_ls * CG(pi.spin, hi_, pj.spin, -hj_, ls.S, hi_ - hj_) * CG(ls.S, hi_ - hj_, ls.L, 0, J, hi_ - hj_)
        H_tilde =
        """
        mother_decay = self.LS_couplings_mother_decay()
        h = sum(
            ls.coupling
            * self.clebsch_gordan(self.spin, hiso_, self.sk, -hk_, ls.S, hiso_ - hk_)
            * self.clebsch_gordan(ls.L, 0, ls.S, hiso_ - hk_, self.s0, hiso_ - hk_)
            * (ls.L + 1) ** 0.5
            / (self.s0 + 1) ** 0.5
            for ls in mother_decay
        )

        if convention == "minus_phi":
            # TODO insert phase difference here
            pass

        return h * (-1) ** ((self.sk - hk_) / 2)


# particle 1

spin0 = 1
spin1 = 1
spin2 = 2
spin3 = 0

helicities = {
    0: list(range(-spin0, spin0 + 1, 2)),
    1: list(range(-spin1, spin1 + 1, 2)),
    2: list(range(-spin2, spin2 + 1, 2)),
    3: list(range(-spin3, spin3 + 1, 2)),
}

tg = TopologyCollection(
    0,
    topologies=[
        Topology(0, decay_topology=((2, 3), 1)),
        Topology(0, decay_topology=((3, 1), 2)),
        Topology(0, decay_topology=((1, 2), 3)),
    ],
)

reference_topology = tg.topologies[0]

theta, psi = np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 40)

THETA, PSI = np.meshgrid(theta, psi)

momenta = make_four_vectors(PSI, THETA, 0)
specific_point = make_four_vectors(0.3, np.arccos(0.4), 0.5)
momenta = {
    i: np.concatenate(
        [momenta[i].reshape((40 * 40, 4)), specific_point[i].reshape((1, 4))], axis=0
    )
    for i in range(1, 4)
}


@cache
def angles(convention):
    final_state_rotations = {
        topology.tuple: reference_topology.relative_wigner_angles(
            topology, momenta, convention=convention
        )
        for topology in tg.topologies
    }

    helicity_angles = {
        topology.tuple: topology.helicity_angles(momenta, convention=convention)
        for topology in tg.topologies
    }
    return final_state_rotations, helicity_angles


def internal_rotation(convention):
    if convention in ["helicity", "minus_phi"]:
        return wigner_capital_d
    elif convention in ["canonical"]:

        def gamma_lm(phi, theta, psi, l, m, m_):
            # m_ is not used, but here, so we have a common interface with wigner_capital_d
            return ((l + 1) / (4 * np.pi)) ** 0.5 * wigner_capital_d(
                phi, theta, psi, l, m, 0
            )

        return gamma_lm
    else:
        raise ValueError(f"Convention {convention} not recognized")


def f(h0, h1, h2, h3, resonance_lineshapes, convention="helicity"):
    helicity_list = [h0, h1, h2, h3]
    spin_list = [spin0, spin1, spin2, spin3]
    amplitude = 0

    final_state_rotations, helicity_angles = angles(convention)

    for topology in tg.topologies:
        final_state_rotation = final_state_rotations[topology.tuple]
        isobars = helicity_angles[topology.tuple]
        for (isobar, bachelor), (phi, theta) in isobars.items():
            if isobar not in resonance_lineshapes:
                # guard clause against key errors
                continue
            (i, j), k = isobar, bachelor
            hi, hj, hk = helicity_list[i], helicity_list[j], helicity_list[k]
            si, sj, sk = spin_list[i], spin_list[j], spin_list[k]

            theta_ij = isobars[isobar].theta_rf
            phi_ij = isobars[isobar].phi_rf

            psi = 0 if convention == "helicity" else -phi
            psi_ij = 0 if convention == "helicity" else -phi_ij

            parts = [
                (resonance.spin + 1) ** 0.5
                * resonance.helicity_coupling_times_lineshape(
                    topology.nodes[isobar].mass(momenta) ** 2, hi_, hj_
                )
                * np.conj(
                    wigner_capital_d(phi, theta, psi, spin0, h0, h_iso - hk_)
                )  # mother decay
                * np.conj(
                    wigner_capital_d(
                        phi_ij, theta_ij, psi_ij, resonance.spin, h_iso, hi_ - hj_
                    )
                )  # isobar decay
                * np.conj(wigner_capital_d(*final_state_rotation[i], si, hi_, hi))
                * np.conj(wigner_capital_d(*final_state_rotation[j], sj, hj_, hj))
                * np.conj(wigner_capital_d(*final_state_rotation[k], sk, hk_, hk))
                * resonance.h_mother(hk_, h_iso)
                for resonance in resonance_lineshapes.get(isobar, [])
                for h_iso in resonance.possible_helicities
                for hk_ in helicities[bachelor]
                for hi_ in helicities[i]
                for hj_ in helicities[j]
            ]
            amplitude += sum(parts)

    return amplitude


resonance_lineshapes_single_3 = {
    (1, 2): [
        resonance(
            1,
            spin0,
            spin1,
            spin2,
            spin3,
            [(2, 1)],
            [(2, 3)],
        )
    ],
}

resonance_lineshapes_single_1 = {
    (2, 3): [resonance(4, spin0, spin2, spin3, spin1, [(4, 3)], [(4, 2)])],
}


def amp_dict(func, resonances, **kwargs):
    return {
        (l1, l2, l3, l4): func(
            l1, l2, l3, l4, resonance_lineshapes=resonances, **kwargs
        )
        for l1 in helicities[0]
        for l2 in helicities[1]
        for l3 in helicities[2]
        for l4 in helicities[3]
    }


def unpolarized(dtc):
    return sum(abs(v) ** 2 for v in dtc.values())


def add_dicts(d1, d2):
    return {k: d1[k] + d2[k] for k in d1.keys()}


def basis_change(dtc, rotation):
    new_dtc = {}
    for key, value in dtc.items():
        l0, l1, l2, l3 = key
        new_dtc[key] = sum(
            dtc[(l0, l1_, l2_, l3_)]
            * np.conj(wigner_capital_d(*rotation[1], spin1, l1_, l1))
            * np.conj(wigner_capital_d(*rotation[2], spin2, l2_, l2))
            * np.conj(wigner_capital_d(*rotation[3], spin3, l3_, l3))
            for l1_ in helicities[1]
            for l2_ in helicities[2]
            for l3_ in helicities[3]
        )
    return new_dtc


def test_eqquivalence():
    terms_1 = amp_dict(f, resonance_lineshapes_single_1)
    terms_2 = amp_dict(f, resonance_lineshapes_single_3)

    terms_1_m = amp_dict(f, resonance_lineshapes_single_1, convention="minus_phi")
    terms_2_m = amp_dict(f, resonance_lineshapes_single_3, convention="minus_phi")

    assert np.allclose(
        unpolarized(add_dicts(terms_1_m, terms_2_m)),
        unpolarized(add_dicts(terms_1, terms_2)),
        rtol=1e-6,
    )
    assert np.allclose(unpolarized(terms_1_m), unpolarized(terms_1))
    assert np.allclose(unpolarized(terms_2_m), unpolarized(terms_2))

    assert np.allclose(
        terms_1[(-1, 1, 2, 0)][-1], -0.14315554700441074 + 0.12414558894503328j
    )

    assert np.allclose(
        terms_2[(-1, 1, 2, 0)][-1], -0.49899891547281655 + 0.030820810874496913j
    )

    assert np.allclose(
        terms_1_m[(-1, 1, 2, 0)][-1], -0.03883258888101088 + 0.1854660829732478j
    )

    assert np.allclose(
        terms_2_m[(-1, 1, 2, 0)][-1], -0.37859261634645197 + 0.32652330831650717j
    )

    rotdict = {
        1: (
            reference_topology.boost(1, momenta, convention="minus_phi")
            @ reference_topology.boost(1, momenta, convention="helicity").inverse()
        ).wigner_angles(),
        2: (
            reference_topology.boost(2, momenta, convention="minus_phi")
            @ reference_topology.boost(2, momenta, convention="helicity").inverse()
        ).wigner_angles(),
        3: (
            reference_topology.boost(3, momenta, convention="minus_phi")
            @ reference_topology.boost(3, momenta, convention="helicity").inverse()
        ).wigner_angles(),
    }

    terms_2_m_new_basis = basis_change(terms_2_m, rotdict)
    terms_1_m_new_basis = basis_change(terms_1_m, rotdict)

    for k, v in terms_2_m_new_basis.items():
        assert np.allclose(v, terms_2[k], atol=1e-6, rtol=1e-6)

    for k, v in terms_1_m_new_basis.items():
        if abs(np.mean(v)) < 1e-6:
            # zero is always a little less precise :/
            assert np.allclose(v, terms_1[k], atol=1e-6, rtol=1e-6)
        else:
            assert np.allclose(v, terms_1[k])


if __name__ == "__main__":
    test_eqquivalence()
