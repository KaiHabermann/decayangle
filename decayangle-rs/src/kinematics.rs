use nalgebra::{Matrix2, Matrix4};
use num_complex::Complex64;

// ── 4-vector accessors ────────────────────────────────────────────────────────
// Convention: v = [px, py, pz, E]  (energy is index 3)

#[inline]
pub fn time_component(v: &[f64; 4]) -> f64 {
    v[3]
}

#[inline]
pub fn x_component(v: &[f64; 4]) -> f64 {
    v[0]
}

#[inline]
pub fn y_component(v: &[f64; 4]) -> f64 {
    v[1]
}

#[inline]
pub fn z_component(v: &[f64; 4]) -> f64 {
    v[2]
}

/// Spatial 3-momentum magnitude |p|
#[inline]
pub fn p_mag(v: &[f64; 4]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Invariant mass  m = sqrt(E^2 - |p|^2)
#[inline]
pub fn mass(v: &[f64; 4]) -> f64 {
    let m2 = v[3] * v[3] - v[0] * v[0] - v[1] * v[1] - v[2] * v[2];
    m2.max(0.0).sqrt()
}

/// Lorentz factor gamma = E / m
#[inline]
pub fn gamma(v: &[f64; 4]) -> f64 {
    time_component(v) / mass(v)
}

/// Velocity beta = |p| / E
#[inline]
pub fn beta(v: &[f64; 4]) -> f64 {
    p_mag(v) / time_component(v)
}

/// Rapidity  xi = 0.5 * ln((1+beta)/(1-beta))
#[inline]
pub fn rapidity(v: &[f64; 4]) -> f64 {
    let b = beta(v);
    0.5 * ((1.0 + b) / (1.0 - b)).ln()
}

// ── 4-vector operations ───────────────────────────────────────────────────────

/// Add two 4-vectors
#[inline]
pub fn add4(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// Apply a 4×4 matrix to a 4-vector
#[inline]
pub fn mat4_vec4(m: &Matrix4<f64>, v: &[f64; 4]) -> [f64; 4] {
    let col = nalgebra::Vector4::new(v[0], v[1], v[2], v[3]);
    let res = m * col;
    [res[0], res[1], res[2], res[3]]
}

/// Lorentz boost of `vector` towards `boostvector`.
/// Equivalent to Python `lorentz_boost(vector, boostvector)` where boostvector
/// is a spatial-only 3-vector encoded as a 4-vector with E=0.
pub fn lorentz_boost(vector: &[f64; 4], boost: &[f64; 3]) -> [f64; 4] {
    let bx = boost[0];
    let by = boost[1];
    let bz = boost[2];
    let b2 = bx * bx + by * by + bz * bz;
    let gma = 1.0 / (1.0 - b2).sqrt();
    let gma2 = if b2.abs() < 1e-14 {
        0.0
    } else {
        (gma - 1.0) / b2
    };

    let ve = vector[3];
    let vx = vector[0];
    let vy = vector[1];
    let vz = vector[2];

    let bp = vx * bx + vy * by + vz * bz;
    let factor = gma2 * bp + gma * ve;

    [
        vx + factor * bx,
        vy + factor * by,
        vz + factor * bz,
        gma * (ve + bp),
    ]
}

/// Boost `vector` to the rest frame of `boostvector`.
/// Equivalent to Python `boost_to_rest`.
pub fn boost_to_rest(vector: &[f64; 4], boostvector: &[f64; 4]) -> [f64; 4] {
    let e = time_component(boostvector);
    let boost = [-boostvector[0] / e, -boostvector[1] / e, -boostvector[2] / e];
    lorentz_boost(vector, &boost)
}

// ── angles ────────────────────────────────────────────────────────────────────

/// Given a 4-vector, return (minus_phi_rf, minus_theta_rf) — the negated angles
/// needed to rotate that vector onto the +z axis.  Mirrors `rotate_to_z_axis`.
pub fn rotate_to_z_axis(v: &[f64; 4]) -> (f64, f64) {
    let phi_rf = v[1].atan2(v[0]);
    let pm = p_mag(v);
    let theta_rf = if pm < 1e-300 {
        0.0
    } else {
        (v[2] / pm).clamp(-1.0, 1.0).acos()
    };
    (-phi_rf, -theta_rf)
}

// ── 4×4 real matrix builders ──────────────────────────────────────────────────
// nalgebra::Matrix4::new() is column-major, so we use explicit index assignment.

pub fn rotation_4_4_z(theta: f64) -> Matrix4<f64> {
    let (s, c) = theta.sin_cos();
    let mut m = Matrix4::identity();
    m[(0, 0)] =  c;  m[(0, 1)] = -s;
    m[(1, 0)] =  s;  m[(1, 1)] =  c;
    m
}

pub fn rotation_4_4_y(theta: f64) -> Matrix4<f64> {
    let (s, c) = theta.sin_cos();
    let mut m = Matrix4::identity();
    m[(0, 0)] =  c;  m[(0, 2)] = s;
    m[(2, 0)] = -s;  m[(2, 2)] = c;
    m
}

pub fn boost_4_4_z(xi: f64) -> Matrix4<f64> {
    let ch = xi.cosh();
    let sh = xi.sinh();
    let mut m = Matrix4::identity();
    m[(2, 2)] = ch;  m[(2, 3)] = sh;
    m[(3, 2)] = sh;  m[(3, 3)] = ch;
    m
}

/// Build the full 4×4 Lorentz matrix from 6 parameters.
/// Order: R_z(phi) @ R_y(theta) @ B_z(xi) @ R_z(phi_rf) @ R_y(theta_rf) @ R_z(psi_rf)
pub fn build_4_4(phi: f64, theta: f64, xi: f64, phi_rf: f64, theta_rf: f64, psi_rf: f64) -> Matrix4<f64> {
    rotation_4_4_z(phi)
        * rotation_4_4_y(theta)
        * boost_4_4_z(xi)
        * rotation_4_4_z(phi_rf)
        * rotation_4_4_y(theta_rf)
        * rotation_4_4_z(psi_rf)
}

// ── 2×2 complex (SU(2)) matrix builders ──────────────────────────────────────

type C64 = Complex64;
type Mat2c = Matrix2<C64>;

#[inline]
fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

// NOTE: nalgebra::Matrix2::new(m00, m10, m01, m11) fills column-major.
// We use explicit index assignment to avoid confusion.

pub fn rotation_2_2_z(theta: f64) -> Mat2c {
    let (s, co) = (theta / 2.0).sin_cos();
    // [[exp(-i*t/2), 0], [0, exp(+i*t/2)]]
    let mut m = Mat2c::zeros();
    m[(0, 0)] = c(co, -s);
    m[(1, 1)] = c(co,  s);
    m
}

pub fn rotation_2_2_y(theta: f64) -> Mat2c {
    let (s, co) = (theta / 2.0).sin_cos();
    // [[cos, -sin], [sin, cos]]
    let mut m = Mat2c::zeros();
    m[(0, 0)] = c(co, 0.0);
    m[(0, 1)] = c(-s, 0.0);
    m[(1, 0)] = c( s, 0.0);
    m[(1, 1)] = c(co, 0.0);
    m
}

pub fn boost_2_2_z(xi: f64) -> Mat2c {
    let ch = (xi / 2.0).cosh();
    let sh = (xi / 2.0).sinh();
    // [[exp(xi/2), 0], [0, exp(-xi/2)]]
    let mut m = Mat2c::zeros();
    m[(0, 0)] = c(ch + sh, 0.0);
    m[(1, 1)] = c(ch - sh, 0.0);
    m
}

/// Build the full 2×2 SU(2) matrix from 6 parameters.
pub fn build_2_2(phi: f64, theta: f64, xi: f64, phi_rf: f64, theta_rf: f64, psi_rf: f64) -> Mat2c {
    rotation_2_2_z(phi)
        * rotation_2_2_y(theta)
        * boost_2_2_z(xi)
        * rotation_2_2_z(phi_rf)
        * rotation_2_2_y(theta_rf)
        * rotation_2_2_z(psi_rf)
}

// ── Decoding ──────────────────────────────────────────────────────────────────

/// Decode boost parameters (phi, theta, xi) from a 4×4 Lorentz matrix.
/// Mirrors `decode_4_4_boost` in kinematics.py.
pub fn decode_4_4_boost(
    m: &Matrix4<f64>,
    tol: f64,
    safety_checks: bool,
) -> Result<(f64, f64, f64), String> {
    // Apply matrix to rest-frame 4-vector [0, 0, 0, 1]
    let vx = m[(0, 3)];
    let vy = m[(1, 3)];
    let vz = m[(2, 3)];
    let vt = m[(3, 3)];

    let mut gma = vt; // mass = 1

    // Clamp values below 1 that are within a reasonable numerical noise band.
    // Composed matrix products accumulate rounding errors (~1e-9), so we use
    // a fixed floor of 1e-6 (well below any physical deviation) before erroring.
    const CLAMP_TOL: f64 = 1e-6;
    if gma < 1.0 && (gma - 1.0).abs() < CLAMP_TOL {
        gma = 1.0;
    }
    if gma < 1.0 {
        let msg = format!("gamma = {gma} < 1: not a valid Lorentz transformation");
        if safety_checks {
            return Err(msg);
        }
    }

    let abs_mom = (vx * vx + vy * vy + vz * vz).sqrt();

    // identity matrix check
    let identity = Matrix4::identity();
    let is_unity = (m - identity).abs().max() < tol;

    let (phi, theta, xi) = if is_unity || (gma - 1.0).abs() < tol {
        (0.0, 0.0, 0.0)
    } else {
        let xi = gma.max(1.0).acosh();
        let phi = vy.atan2(vx);
        let cos_input = if abs_mom <= tol {
            0.0
        } else {
            (vz / abs_mom).clamp(-1.0, 1.0)
        };
        let theta = cos_input.acos();
        (phi, theta, xi)
    };

    Ok((phi, theta, xi))
}

/// Decode Euler angles (phi, theta, psi) from a 3×3 rotation matrix (upper-left of a 4×4).
/// Mirrors `decode_rotation_4x4`.
pub fn decode_rotation_3x3(m: &Matrix4<f64>) -> (f64, f64, f64) {
    let phi = m[(1, 2)].atan2(m[(0, 2)]);
    let theta = m[(2, 2)].clamp(-1.0, 1.0).acos();
    let psi = m[(2, 1)].atan2(-m[(2, 0)]);
    (phi, theta, psi)
}

/// Decode Euler angles (gamma, beta, alpha) from a 2×2 SU(2) rotation matrix.
/// Mirrors `decode_su2_rotation`.
pub fn decode_su2_rotation(m: &Mat2c) -> (f64, f64, f64) {
    let cosbeta = (m[(0, 0)] * m[(1, 1)] + m[(0, 1)] * m[(1, 0)]).re.clamp(-1.0, 1.0);
    let beta = cosbeta.acos();
    let alpha_p_gamma = m[(1, 1)].arg();
    let alpha_m_gamma = -m[(1, 0)].arg();
    let alpha = alpha_p_gamma + alpha_m_gamma;
    let gamma = alpha_p_gamma - alpha_m_gamma;
    (gamma, beta, alpha)
}
