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
    
    def __floor(self):
        """
        Set very small values to zero. This helps with numerical stability.
        """
        self.M4 = config.backend.where(config.backend.abs(self.M4) < 1e-14, 0, self.M4)
        self.M2 = config.backend.where(config.backend.abs(self.M2) < 1e-14, 0 + 0j, self.M2)

    def decode(self, two_pi_aware=True):
        self.__floor()
        params = decode_4_4(self.M4)
        if two_pi_aware:
            params = adjust_for_2pi_rotation(self.M2, *params)
        return params
    
    def __repr__(self):
        return f"LorentzTrafo" + "\n SU(2): \n" + f"{self.M2}" + "\n O(3): \n" + f"{self.M4}"
    
    def inverse(self):
        return LorentzTrafo(M2=config.backend.linalg.inv(self.M2), M4=config.backend.linalg.inv(self.M4))

    # TODO: think about whether this is a good idea
    # def __getattr__(self, name):
    #     try:
    #         attr2 = getattr(self.M2, name)
    #         attr4 = getattr(self.M4, name)
    #         return lambda *args, **kwargs: LorentzTrafo(M2=attr2(*args, **kwargs), M4=attr4(*args, **kwargs))
    #     except AttributeError:
    #         raise AttributeError(f"Attribute {name} not found in LorentzTrafo")
