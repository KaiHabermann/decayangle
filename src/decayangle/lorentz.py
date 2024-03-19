from decayangle.kinematics import *
from decayangle.config import config


class LorentzTrafo:
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            theta, phi, xi, theta_rf, phi_rf, xi_rf = args
            self.M4 = build_4_4(theta, phi, xi, theta_rf, phi_rf, xi_rf)
            self.M2 = build_2_2(theta, phi, xi, theta_rf, phi_rf, xi_rf)
        M2 = kwargs.get('M2', None)
        M4 = kwargs.get('M4', None)
        if M2 is not None and M4 is not None:
            self.M2 = M2
            self.M4 = M4
        elif len(args) == 0:
            raise ValueError("LorentzTrafo must be initialized with either 6 values or 2 matrices")

    def __matmul__(self, other):
        if isinstance(other, LorentzTrafo):
            return LorentzTrafo(M2=self.M2 @ other.M2, M4=self.M4 @ other.M4)

    def decode(self, two_pi_aware=True):
        params = decode_4_4(self.M4)
        if two_pi_aware:
            params = adjust_for_2pi_rotation(self.M2, *params)
        return params
    
    def __repr__(self):
        return f"LorentzTrafo" + "\n SU(2): \n" + f"{self.M2}" + "\n O(3): \n" + f"{self.M4}"
    
    def inverse(self):
        return LorentzTrafo(M2=config.backend.linalg.inv(self.M2), M4=config.backend.linalg.inv(self.M4))
    
    def wigner_angles(self):
        theta, phi, xi, theta_rf, phi_rf, xi_rf = self.decode(two_pi_aware=True)
        return phi_rf, xi_rf