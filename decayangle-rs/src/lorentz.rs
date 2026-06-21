use nalgebra::{Matrix2, Matrix4};
use num_complex::Complex64;

use crate::kinematics::{
    build_4_4, build_2_2, boost_2_2_z, rotation_2_2_y, rotation_2_2_z,
    decode_4_4_boost, decode_su2_rotation,
};

type C64 = Complex64;
type Mat2c = Matrix2<C64>;

/// Per-event Lorentz transformation holding both representations.
#[derive(Clone, Debug)]
pub struct LorentzTrafo {
    pub matrix_4x4: Matrix4<f64>,
    pub matrix_2x2: Mat2c,
}

impl LorentzTrafo {
    /// Build from 6 kinematic parameters.
    pub fn from_params(phi: f64, theta: f64, xi: f64, phi_rf: f64, theta_rf: f64, psi_rf: f64) -> Self {
        Self {
            matrix_4x4: build_4_4(phi, theta, xi, phi_rf, theta_rf, psi_rf),
            matrix_2x2: build_2_2(phi, theta, xi, phi_rf, theta_rf, psi_rf),
        }
    }

    /// Identity transformation.
    pub fn identity() -> Self {
        Self {
            matrix_4x4: Matrix4::identity(),
            matrix_2x2: Matrix2::identity(),
        }
    }

    /// Compose: self @ other  (apply other first, then self)
    pub fn compose(&self, other: &LorentzTrafo) -> LorentzTrafo {
        LorentzTrafo {
            matrix_4x4: self.matrix_4x4 * other.matrix_4x4,
            matrix_2x2: self.matrix_2x2 * other.matrix_2x2,
        }
    }

    /// Inverse using nalgebra's built-in matrix inversion.
    pub fn inverse(&self) -> LorentzTrafo {
        let inv4 = self.matrix_4x4.try_inverse()
            .expect("4x4 Lorentz matrix is not invertible");
        let inv2 = self.matrix_2x2.try_inverse()
            .expect("2x2 SU(2) matrix is not invertible");
        LorentzTrafo {
            matrix_4x4: inv4,
            matrix_2x2: inv2,
        }
    }

    /// Decode Wigner angles (phi_rf, theta_rf, psi_rf) using the su2_decode method.
    /// Mirrors `LorentzTrafo.wigner_angles(method="su2_decode")`.
    pub fn wigner_angles(
        &self,
        tol: f64,
        safety_checks: bool,
    ) -> Result<(f64, f64, f64), String> {
        let (phi, theta, xi) = decode_4_4_boost(&self.matrix_4x4, tol, safety_checks)?;

        // Remove the boost part from the 2×2 matrix to isolate the rest-frame rotation.
        // su2_rot = B_z(-xi) @ R_y(-theta) @ R_z(-phi) @ matrix_2x2
        let su2_rot = boost_2_2_z(-xi)
            * rotation_2_2_y(-theta)
            * rotation_2_2_z(-phi)
            * self.matrix_2x2;

        // decode_su2_rotation returns (gamma, beta, alpha) == (phi_rf, theta_rf, psi_rf)
        let (phi_rf, theta_rf, psi_rf) = decode_su2_rotation(&su2_rot);

        Ok((phi_rf, theta_rf, psi_rf))
    }
}
