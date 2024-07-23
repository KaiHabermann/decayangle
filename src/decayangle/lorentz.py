from __future__ import annotations
from typing import Tuple, Union, Optional, Literal
from collections import namedtuple
import numpy as np
import jax.numpy as jnp
from decayangle.kinematics import (
    build_4_4,
    build_2_2,
    decode_4_4_boost,
    decode_4_4,
    decode_su2_rotation,
    rotation_matrix_2_2_y,
    boost_matrix_2_2_z,
    rotation_matrix_2_2_z,
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

    The 4x4 matrix is used to perform the Lorentz transformation on a 4-vector and to decode the parameters of the transformation up to a rotation of 2 pi.
    The 2x2 matrix can then be used to determine if a rotation of 2 pi has been performed, as this implies matrix_2x2(decoded params) = -matrix_2x2(original params).
    This is important information for fermions.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the Lorentz transformation with either 6 values or 2 matrices.
        The matrices are the 2x2 SU(2) matrix and the 4x4 O(3) matrix.
        The 6 values are the parameters of the Lorentz transformation in the order phi, theta, xi, phi_rf, theta_rf, psi_rf.
        These parameters correspont to the following chain of transformations:
        1. Rotation around the z-axis by phi
        2. Rotation around the y-axis by theta
        3. Boost along the z-axis by xi
        4. Rotation around the z-axis by phi_rf
        5. Rotation around the y-axis by theta_rf
        6. Rotation around the z-axis by psi_rf

        Args:
            *args: The parameters of the Lorentz transformation in the order phi, theta, xi, phi_rf, theta_rf, psi_rf
            **kwargs: The matrices of the Lorentz transformation

        Raises:
            ValueError: If the Lorentz transformation is not initialized with either 6 values or 2 matrices

        Examples:
        ```python
        # Initialize with 6 values
        LorentzTrafo(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        # Initialize with matrices

        matrix_2x2 = np.array([[1, 0], [0, 1]])
        matrix_4x4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        LorentzTrafo(matrix_2x2=matrix_2x2, matrix_4x4=matrix_4x4)
        ```
        """

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

    def __matmul__(self, other: LorentzTrafo) -> LorentzTrafo:
        """
        Multiply two Lorentz transformations together. This is equivalent to applying the transformations in sequence.
        Overloads the @ operator.

        Args:
            other (LorentzTrafo): The other Lorentz transformation

        Returns:
            LorentzTrafo: The result of the multiplication
        """
        if isinstance(other, LorentzTrafo):
            return LorentzTrafo(
                matrix_2x2=self.matrix_2x2 @ other.matrix_2x2,
                matrix_4x4=self.matrix_4x4 @ other.matrix_4x4,
            )
        raise ValueError("Only LorentzTrafo can be multiplied with LorentzTrafo")

    def decode(
        self,
        two_pi_aware=True,
        tol: Optional[float] = None,
        method: Literal["flip", "su2_decode"] = "flip",
    ) -> Tuple[Union[np.array, jnp.array]]:
        """Decode the parameters of the Lorentz transformation

        Args:
            two_pi_aware (bool, optional): If true the check for a rotation of 2 pi will be made. Defaults to True.
            tol (Optional[float], optional): The tolerance for the check of a 2 pi rotation. If None the default tolerance of the config will be used. Defaults to None.

        Returns:
            Tuple[Union[np.array, jnp.array]]: The parameters of the Lorentz transformation
        """
        if method == "flip":
            phi, theta, xi, phi_rf, theta_rf, psi_rf = decode_4_4(
                self.matrix_4x4, tol=tol
            )
            if two_pi_aware:
                phi, theta, xi, phi_rf, theta_rf, psi_rf = adjust_for_2pi_rotation(
                    self.matrix_2x2, phi, theta, xi, phi_rf, theta_rf, psi_rf
                )
            return phi, theta, xi, phi_rf, theta_rf, psi_rf
        if method == "su2_decode":
            phi, theta, xi = decode_4_4_boost(self.matrix_4x4, tol=tol)
            su2_rot = (
                boost_matrix_2_2_z(-xi)
                @ rotation_matrix_2_2_y(-theta)
                @ rotation_matrix_2_2_z(-phi)
                @ self.matrix_2x2
            )
            # check for the special case of no absolute boost
            phi_rf_no_boost, theta_rf_no_boost, psi_rf_no_boost = decode_su2_rotation(
                su2_rot
            )
            return phi, theta, xi, phi_rf_no_boost, theta_rf_no_boost, psi_rf_no_boost

        raise ValueError(f"Invalid method for decoding: {method}")

    def __repr__(self) -> str:
        """
        String representation of the Lorentz transformation. It shows the SU(2) and O(3,1) matrices.

        Returns:
            str: The string representation of the Lorentz transformation
        """
        return (
            "LorentzTrafo"
            + "\n SU(2): \n"
            + f"{self.matrix_2x2}"
            + "\n O(3): \n"
            + f"{self.matrix_4x4}"
        )

    def inverse(self) -> LorentzTrafo:
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
