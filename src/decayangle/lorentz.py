from typing import Tuple, Union, Optional
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
    """
    A class to represent a Lorentz transformation. It is initialized with either 6 values or 2 matrices.
    The matrices are the 2x2 SU(2) matrix and the 4x4 O(3) matrix.
    Both representations are held in the class and can be accessed via the attributes matrix_2x2 and matrix_4x4.

    The 4x4 matrix is used to perfrom the Lorentz transformation on a 4-vector and to decode the parameters of the transformation up to a rotation of 2 pi.
    The 2x2 matrix can then be used to determine if a rotation of 2 pi has been performed, as this implies matrix_2x2(decoded params) = -matrix_2x2(original params).
    This is important information for fermions. 
    """

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

    def decode(self, two_pi_aware=True, tol:Optional[float]=None) -> Tuple[Union[np.array, jnp.array]]:
        """Decode the parameters of the Lorentz transformation

        Args:
            two_pi_aware (bool, optional): If true the check for a roation of 2 pi will be made. Defaults to True.
            tol (Optional[float], optional): The tolerance for the check of a 2 pi rotation. If None the default tolerance of the config will be used. Defaults to None.

        Returns:
            Tuple[Union[np.array, jnp.array]]: The parameters of the Lorentz transformation
        """
        params = decode_4_4(self.matrix_4x4, tol=tol)
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
