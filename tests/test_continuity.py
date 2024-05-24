import numpy as np
from decayangle.lorentz import LorentzTrafo
from decayangle.decay_topology import Topology
from decayangle.config import config as cfg
import matplotlib.pyplot as plt

def make_four_vectors(phi_rf, theta_rf, psi_rf):
    import numpy as np

    # Make sure, the sorting is turned off
    cfg.sorting = "off"

    # Given values
    # Lc -> p K pi
    m1, m2, m3, m0 = 0.93827, 0.493677, 0.139570, 2.28646
    m12 = 2.756020646168232**0.5
    m23 = 1.3743747462964881**0.5

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

    # Vectors
    p1 = np.array([0, 0, p1z, E1])
    p2 = np.array([p2x, 0, p2z, E2])
    p3 = np.array([p3x, 0, p3z, E3])

    # Lorentz transformation
    momenta = {i: p for i, p in zip([1, 2, 3], [p1, p2, p3])}
    tree1 = Topology(root=0, decay_topology=((2, 3), 1))
    tree2 = Topology(root=0, decay_topology=((3, 1), 2))
    tree3 = Topology(root=0, decay_topology=((1, 2), 3))
    #  
    rotation = LorentzTrafo(0, 0, 0, phi_rf, theta_rf, psi_rf)
    # 
    momenta_23_rotated = tree1.root.transform(rotation, momenta)
    return tree1.relative_wigner_angles(tree3, momenta_23_rotated)


x = np.linspace(1e-5, np.pi - 1e-5, 100)
y = np.linspace(-np.pi + 1e-5, np.pi - 1e-5, 100)

X, Y = np.meshgrid(x, y)
result_full = make_four_vectors(0, X, Y)

result_psi = result_full[1].psi_rf
result_phi = result_full[1].phi_rf
result = result_psi + result_phi

def fmod_4(val):
    return np.fmod(val/np.pi + 4,4)

img = plt.imshow(fmod_4(result), cmap='hot', origin='lower',
                 extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')

plt.colorbar(label="angle in multiples of pi")
plt.xlabel('Theta_rf')
plt.ylabel('Psi_rf')
plt.title('psi_3(1)_for_1 + phi_3(1)_for_1')
plt.savefig('test.png')
plt.close('all')


img = plt.imshow(fmod_4(result_phi), cmap='hot', origin='lower',
                 extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')

plt.colorbar(label="angle in multiples of pi")
plt.xlabel('Theta_rf')
plt.ylabel('Psi_rf')
plt.title('phi_3(1)_for_1')
plt.savefig('test_phi.png')
plt.close('all')


img = plt.imshow(fmod_4(result_psi), cmap='hot', origin='lower',
                 extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')

plt.colorbar(label="angle in multiples of pi")
plt.xlabel('Theta_rf')
plt.ylabel('Psi_rf')
plt.title('psi_3(1)_for_1')
plt.savefig('test_psi.png')
plt.close('all')