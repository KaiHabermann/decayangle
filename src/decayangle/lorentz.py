from typing import Tuple, Union
from collections import namedtuple
import numpy as np
import jax.numpy as jnp
from decayangle.kinematics import (
    build_4_4,
    build_2_2,
    decode_4_4,
    adjust_for_2pi_rotation,
)
from decayangle.config import config as cfg

cb = cfg.backend

WignerAngles = namedtuple("WignerAngles", ["phi_rf", "theta_rf", "psi_rf"])


class LorentzTrafo:
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            phi, theta, xi, phi_rf, theta_rf, psi_rf = args
            self.matrix_4x4 = build_4_4(phi, theta, xi, phi_rf, theta_rf, psi_rf)
            self.matrix_2x2 = build_2_2(phi, theta, xi, phi_rf, theta_rf, psi_rf)
        matrix_2x2 = kwargs.get("matrix_2x2", None)
        matrix_4x4 = kwargs.get("matrix_4x4", None)
        if matrix_2x2 is not None and matrix_4x4 is not None:
            self.matrix_2x2 = matrix_2x2
            self.matrix_4x4 = matrix_4x4
        elif len(args) == 0:
            raise ValueError(
                "LorentzTrafo must be initialized with either 6 values or 2 matrices"
            )

    def __matmul__(self, other):
        if isinstance(other, LorentzTrafo):
            return LorentzTrafo(
                matrix_2x2=self.matrix_2x2 @ other.matrix_2x2,
                matrix_4x4=self.matrix_4x4 @ other.matrix_4x4,
            )
        raise ValueError("Only LorentzTrafo can be multiplied with LorentzTrafo")

    def decode(self, two_pi_aware=True) -> Tuple[Union[np.array, jnp.array]]:
        """Decode the parameters of the Lorentz transformation

        Args:
            two_pi_aware (bool, optional): If true the check for a roation of 2 pi will be made. Defaults to True.

        Returns:
            Tuple[Union[np.array, jnp.array]]: The parameters of the Lorentz transformation
        """
        params = decode_4_4(self.matrix_4x4)
        if two_pi_aware:
            params = adjust_for_2pi_rotation(self.matrix_2x2, *params)
        return params

    def __repr__(self):
        return (
            "LorentzTrafo"
            + "\n SU(2): \n"
            + f"{self.matrix_2x2}"
            + "\n O(3): \n"
            + f"{self.matrix_4x4}"
        )

    def inverse(self):
        """Inverse of the Lorentz transformation

        Returns:
            LorentzTrafo: the inverse of the Lorentz transformation
        """
        return LorentzTrafo(
            matrix_2x2=cb.linalg.inv(self.matrix_2x2),
            matrix_4x4=cb.linalg.inv(self.matrix_4x4),
        )

    def wigner_angles(self) -> Tuple[Union[np.array, jnp.array]]:
        """The wigner angles of a transformation
        These are usually the angles of the rotation before the boost

        Returns:
            Tuple[Union[np.array, jnp.array]]: the angles of the rotation in the frame before the boost
        """
        _, _, _, phi_rf, theta_rf, psi_rf = self.decode(two_pi_aware=True)
        return WignerAngles(phi_rf, theta_rf, psi_rf)
