use std::collections::HashMap;
use rayon::prelude::*;

use crate::kinematics::{
    add4, mat4_vec4, gamma, rapidity, rotate_to_z_axis, boost_to_rest,
    build_4_4, build_2_2, decode_4_4_boost, decode_su2_rotation,
    boost_2_2_z, rotation_2_2_y, rotation_2_2_z,
};
use nalgebra::{Matrix4, Matrix2};
use num_complex::Complex64;

// ── Convention ────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Convention {
    Helicity,
    MinusPhi,
    Canonical,
}

impl Convention {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "helicity"  => Ok(Convention::Helicity),
            "minus_phi" => Ok(Convention::MinusPhi),
            "canonical" => Ok(Convention::Canonical),
            other => Err(format!("Unknown convention '{other}'. Use 'helicity', 'minus_phi' or 'canonical'.")),
        }
    }
}

// ── Tree representation ───────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Leaf(i32),
    Internal {
        label: Vec<i32>,
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl Node {
    pub fn particles(&self) -> Vec<i32> {
        match self {
            Node::Leaf(i) => vec![*i],
            Node::Internal { label, .. } => label.clone(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf(_))
    }

    pub fn momentum(&self, momenta: &HashMap<i32, Vec<[f64; 4]>>) -> Vec<[f64; 4]> {
        match self {
            Node::Leaf(i) => momenta[i].clone(),
            Node::Internal { left, right, .. } => {
                let lm = left.momentum(momenta);
                let rm = right.momentum(momenta);
                lm.iter().zip(rm.iter()).map(|(a, b)| add4(a, b)).collect()
            }
        }
    }

    pub fn daughters(&self) -> Option<(&Node, &Node)> {
        match self {
            Node::Internal { left, right, .. } => Some((left.as_ref(), right.as_ref())),
            Node::Leaf(_) => None,
        }
    }

    pub fn preorder(&self) -> Vec<&Node> {
        let mut out = vec![self as &Node];
        if let Node::Internal { left, right, .. } = self {
            out.extend(left.preorder());
            out.extend(right.preorder());
        }
        out
    }
}

// ── Structure-of-Arrays Lorentz transformation batch ─────────────────────────
//
// Instead of Vec<LorentzTrafo> (array of structs), we store N 4×4 matrices
// as a flat Vec<f64> of length N*16 (row-major), and N 2×2 complex matrices
// as Vec<[Complex64; 4]>. This keeps all matrix data for event i contiguous
// and is SIMD/cache friendly for the composition loops.

struct TrafoSoA {
    n: usize,
    m4: Vec<f64>,           // N * 16 floats, row-major
    m2: Vec<Complex64>,     // N * 4 complex, row-major
}

impl TrafoSoA {
    fn identity(n: usize) -> Self {
        let eye4: [f64; 16] = [
            1.,0.,0.,0.,
            0.,1.,0.,0.,
            0.,0.,1.,0.,
            0.,0.,0.,1.,
        ];
        let eye2: [Complex64; 4] = [
            Complex64::new(1.,0.), Complex64::new(0.,0.),
            Complex64::new(0.,0.), Complex64::new(1.,0.),
        ];
        let mut m4 = Vec::with_capacity(n * 16);
        let mut m2 = Vec::with_capacity(n * 4);
        for _ in 0..n {
            m4.extend_from_slice(&eye4);
            m2.extend_from_slice(&eye2);
        }
        TrafoSoA { n, m4, m2 }
    }

    fn from_params_batch(params: &[(f64, f64, f64, f64, f64, f64)]) -> Self {
        let n = params.len();
        let mut m4 = vec![0.0f64; n * 16];
        let mut m2 = vec![Complex64::new(0.,0.); n * 4];
        params.par_iter().zip(m4.par_chunks_mut(16)).zip(m2.par_chunks_mut(4))
            .for_each(|(((phi, theta, xi, phi_rf, theta_rf, psi_rf), chunk4), chunk2)| {
                let mat4 = build_4_4(*phi, *theta, *xi, *phi_rf, *theta_rf, *psi_rf);
                let mat2 = build_2_2(*phi, *theta, *xi, *phi_rf, *theta_rf, *psi_rf);
                for row in 0..4 {
                    for col in 0..4 {
                        chunk4[row * 4 + col] = mat4[(row, col)];
                    }
                }
                chunk2[0] = mat2[(0,0)]; chunk2[1] = mat2[(0,1)];
                chunk2[2] = mat2[(1,0)]; chunk2[3] = mat2[(1,1)];
            });
        TrafoSoA { n, m4, m2 }
    }

    #[inline]
    fn mat4(&self, i: usize) -> Matrix4<f64> {
        Matrix4::from_row_slice(&self.m4[i * 16..i * 16 + 16])
    }

    #[inline]
    fn mat2(&self, i: usize) -> Matrix2<Complex64> {
        Matrix2::from_row_slice(&self.m2[i * 4..i * 4 + 4])
    }

    /// compose: result[i] = self[i] @ other[i]
    fn compose(&self, other: &TrafoSoA) -> TrafoSoA {
        let n = self.n;
        let mut m4 = vec![0.0f64; n * 16];
        let mut m2 = vec![Complex64::new(0.,0.); n * 4];

        m4.par_chunks_mut(16).zip(m2.par_chunks_mut(4))
            .enumerate()
            .for_each(|(i, (c4, c2))| {
                let a4 = self.mat4(i);
                let b4 = other.mat4(i);
                let r4 = a4 * b4;
                for row in 0..4 {
                    for col in 0..4 {
                        c4[row * 4 + col] = r4[(row, col)];
                    }
                }
                let a2 = self.mat2(i);
                let b2 = other.mat2(i);
                let r2 = a2 * b2;
                c2[0] = r2[(0,0)]; c2[1] = r2[(0,1)];
                c2[2] = r2[(1,0)]; c2[3] = r2[(1,1)];
            });

        TrafoSoA { n, m4, m2 }
    }

    /// compose_into: self[i] = step[i] @ self[i]  (left-multiply in place)
    fn left_compose_assign(&mut self, step: &TrafoSoA) {
        self.m4.par_chunks_mut(16).zip(self.m2.par_chunks_mut(4))
            .enumerate()
            .for_each(|(i, (c4, c2))| {
                let a4 = step.mat4(i);
                let b4 = Matrix4::from_row_slice(c4);
                let r4 = a4 * b4;
                for row in 0..4 {
                    for col in 0..4 {
                        c4[row * 4 + col] = r4[(row, col)];
                    }
                }
                let a2 = step.mat2(i);
                let b2 = Matrix2::from_row_slice(c2);
                let r2 = a2 * b2;
                c2[0] = r2[(0,0)]; c2[1] = r2[(0,1)];
                c2[2] = r2[(1,0)]; c2[3] = r2[(1,1)];
            });
    }

    /// right_compose_inv_assign: self[i] = self[i] @ step[i]^{-1}
    fn right_compose_inv_assign(&mut self, step: &TrafoSoA) {
        self.m4.par_chunks_mut(16).zip(self.m2.par_chunks_mut(4))
            .enumerate()
            .for_each(|(i, (c4, c2))| {
                let b4 = step.mat4(i).try_inverse().expect("non-invertible 4x4");
                let a4 = Matrix4::from_row_slice(c4);
                let r4 = a4 * b4;
                for row in 0..4 {
                    for col in 0..4 {
                        c4[row * 4 + col] = r4[(row, col)];
                    }
                }
                let b2 = step.mat2(i).try_inverse().expect("non-invertible 2x2");
                let a2 = Matrix2::from_row_slice(c2);
                let r2 = a2 * b2;
                c2[0] = r2[(0,0)]; c2[1] = r2[(0,1)];
                c2[2] = r2[(1,0)]; c2[3] = r2[(1,1)];
            });
    }

    /// Apply all 4×4 matrices to the corresponding event momenta.
    fn transform_momenta(&self, momenta: &HashMap<i32, Vec<[f64; 4]>>) -> HashMap<i32, Vec<[f64; 4]>> {
        momenta.iter().map(|(k, batch)| {
            let transformed: Vec<[f64; 4]> = batch.par_iter().enumerate()
                .map(|(i, v)| {
                    let m = self.mat4(i);
                    mat4_vec4(&m, v)
                })
                .collect();
            (*k, transformed)
        }).collect()
    }

    /// Decode Wigner angles for all N events. Returns (phis, thetas, psis).
    fn wigner_angles_batch(
        &self,
        tol: f64,
        safety_checks: bool,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
        let n = self.n;
        let mut phis   = vec![0.0f64; n];
        let mut thetas = vec![0.0f64; n];
        let mut psis   = vec![0.0f64; n];

        // Parallel decode — collect errors then propagate
        let results: Vec<Result<(f64, f64, f64), String>> = (0..n).into_par_iter().map(|i| {
            let m4 = self.mat4(i);
            let m2 = self.mat2(i);
            let (phi, theta, xi) = decode_4_4_boost(&m4, tol, safety_checks)?;
            let su2_rot = boost_2_2_z(-xi)
                * rotation_2_2_y(-theta)
                * rotation_2_2_z(-phi)
                * m2;
            let (phi_rf, theta_rf, psi_rf) = decode_su2_rotation(&su2_rot);
            Ok((phi_rf, theta_rf, psi_rf))
        }).collect();

        for (i, r) in results.into_iter().enumerate() {
            let (phi_rf, theta_rf, psi_rf) = r?;
            phis[i]   = phi_rf;
            thetas[i] = theta_rf;
            psis[i]   = psi_rf;
        }
        Ok((phis, thetas, psis))
    }
}

// ── Topology ──────────────────────────────────────────────────────────────────

pub struct Topology {
    pub root: Node,
}

impl Topology {
    pub fn new(root: Node) -> Self {
        Topology { root }
    }

    pub fn final_state_nodes(&self) -> Vec<&Node> {
        self.root.preorder().into_iter().filter(|n| n.is_leaf()).collect()
    }

    pub fn to_rest_frame(
        &self,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
    ) -> HashMap<i32, Vec<[f64; 4]>> {
        let root_mom = self.root.momentum(momenta);
        let already_at_rest = root_mom.iter().all(|v| (gamma(v) - 1.0).abs() < tol);
        if already_at_rest {
            return momenta.clone();
        }
        momenta.iter().map(|(k, batch)| {
            let boosted: Vec<[f64; 4]> = batch.par_iter().enumerate()
                .map(|(i, v)| boost_to_rest(v, &root_mom[i]))
                .collect();
            (*k, boosted)
        }).collect()
    }

    fn path_to<'a>(&'a self, target: &Node) -> Vec<&'a Node> {
        fn dfs<'a>(current: &'a Node, tp: &[i32], path: &mut Vec<&'a Node>) -> bool {
            path.push(current);
            if current.particles() == tp { return true; }
            if let Some((l, r)) = current.daughters() {
                if dfs(l, tp, path) { return true; }
                if dfs(r, tp, path) { return true; }
            }
            path.pop();
            false
        }
        let tp = target.particles();
        let mut path = Vec::new();
        dfs(&self.root, &tp, &mut path);
        path
    }

    /// Compute one step of the boost: node → target daughter, for a given convention.
    fn boost_step(
        node: &Node,
        target: &Node,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
        convention: Convention,
    ) -> Result<TrafoSoA, String> {
        let n = momenta.values().next().map(|v| v.len()).unwrap_or(0);

        if node.particles() == target.particles() {
            return Ok(TrafoSoA::identity(n));
        }

        let (left, _) = node.daughters().ok_or("node has no daughters")?;
        let target_is_first = left.particles() == target.particles();

        let left_mom   = left.momentum(momenta);
        let target_mom = target.momentum(momenta);

        if safety_checks {
            let node_mom = node.momentum(momenta);
            for v in &node_mom {
                let g = gamma(v);
                if (g - 1.0).abs() > tol {
                    return Err(format!("gamma = {g} — node not at rest"));
                }
            }
        }

        let n_events = n;
        let mut m4 = vec![0.0f64; n_events * 16];
        let mut m2 = vec![Complex64::new(0., 0.); n_events * 4];

        // For canonical, we rotate to target (not always left daughter)
        // For helicity/minus_phi, we always rotate to the left daughter
        let rotate_ref_mom = if convention == Convention::Canonical {
            target_mom.clone()
        } else {
            left_mom.clone()
        };

        m4.par_chunks_mut(16).zip(m2.par_chunks_mut(4))
            .enumerate()
            .for_each(|(i, (c4, c2))| {
                let (minus_phi_rf, minus_theta_rf) = rotate_to_z_axis(&rotate_ref_mom[i]);
                let xi = -rapidity(&target_mom[i]);

                let bst4 = build_4_4(0., 0., xi, 0., 0., 0.);
                let bst2 = build_2_2(0., 0., xi, 0., 0., 0.);

                let (r4, r2) = match convention {
                    Convention::Canonical => {
                        // rot aligns target with z; result = rot^{-1} @ boost @ rot
                        // no flip needed — canonical always acts on the specific target
                        let rot4 = build_4_4(0., 0., 0., 0., minus_theta_rf, minus_phi_rf);
                        let rot2 = build_2_2(0., 0., 0., 0., minus_theta_rf, minus_phi_rf);
                        let rot4_inv = rot4.try_inverse().expect("rotation invertible");
                        let rot2_inv = rot2.try_inverse().expect("rotation invertible");
                        (rot4_inv * bst4 * rot4, rot2_inv * bst2 * rot2)
                    }
                    Convention::Helicity => {
                        // rot aligns left daughter with z; result = boost @ [flip @] rot
                        let rot4 = build_4_4(0., 0., 0., 0., minus_theta_rf, minus_phi_rf);
                        let rot2 = build_2_2(0., 0., 0., 0., minus_theta_rf, minus_phi_rf);
                        if target_is_first {
                            (bst4 * rot4, bst2 * rot2)
                        } else {
                            let flip4 = build_4_4(0., 0., 0., 0., -std::f64::consts::PI, 0.);
                            let flip2 = build_2_2(0., 0., 0., 0., -std::f64::consts::PI, 0.);
                            (bst4 * flip4 * rot4, bst2 * flip2 * rot2)
                        }
                    }
                    Convention::MinusPhi => {
                        // rot uses (-psi, theta, psi); result = boost @ [flip @] rot
                        let rot4 = build_4_4(0., 0., 0., -minus_phi_rf, minus_theta_rf, minus_phi_rf);
                        let rot2 = build_2_2(0., 0., 0., -minus_phi_rf, minus_theta_rf, minus_phi_rf);
                        if target_is_first {
                            (bst4 * rot4, bst2 * rot2)
                        } else {
                            let flip4 = build_4_4(0., 0., 0., 0., -std::f64::consts::PI, 0.);
                            let flip2 = build_2_2(0., 0., 0., 0., -std::f64::consts::PI, 0.);
                            (bst4 * flip4 * rot4, bst2 * flip2 * rot2)
                        }
                    }
                };

                for row in 0..4 { for col in 0..4 {
                    c4[row * 4 + col] = r4[(row, col)];
                }}
                c2[0] = r2[(0,0)]; c2[1] = r2[(0,1)];
                c2[2] = r2[(1,0)]; c2[3] = r2[(1,1)];
            });

        Ok(TrafoSoA { n: n_events, m4, m2 })
    }

    /// Full boost from root to target. Returns the composed SoA trafo.
    fn boost_to_soa(
        &self,
        target: &Node,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
        convention: Convention,
    ) -> Result<TrafoSoA, String> {
        let path = self.path_to(target);
        let n = momenta.values().next().map(|v| v.len()).unwrap_or(0);

        if path.len() < 2 {
            return Ok(TrafoSoA::identity(n));
        }

        let mut current_momenta = momenta.clone();
        let first_step = Self::boost_step(path[0], path[1], &current_momenta, tol, safety_checks, convention)?;
        current_momenta = first_step.transform_momenta(&current_momenta);
        let mut composed = first_step;

        for i in 1..path.len() - 1 {
            let step = Self::boost_step(path[i], path[i+1], &current_momenta, tol, safety_checks, convention)?;
            current_momenta = step.transform_momenta(&current_momenta);
            composed.left_compose_assign(&step);
        }

        Ok(composed)
    }

    /// Full inverse boost from root to target.
    fn boost_to_inverse_soa(
        &self,
        target: &Node,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
        convention: Convention,
    ) -> Result<TrafoSoA, String> {
        let path = self.path_to(target);
        let n = momenta.values().next().map(|v| v.len()).unwrap_or(0);

        if path.len() < 2 {
            return Ok(TrafoSoA::identity(n));
        }

        let mut all_steps: Vec<TrafoSoA> = Vec::new();
        let mut current_momenta = momenta.clone();

        for i in 0..path.len() - 1 {
            let step = Self::boost_step(path[i], path[i+1], &current_momenta, tol, safety_checks, convention)?;
            current_momenta = step.transform_momenta(&current_momenta);
            all_steps.push(step);
        }

        let mut result = TrafoSoA::identity(n);
        result.right_compose_inv_assign(&all_steps[0]);
        for step in &all_steps[1..] {
            result.right_compose_inv_assign(step);
        }
        Ok(result)
    }

    pub fn helicity_angles(
        &self,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
        convention: Convention,
    ) -> Result<HashMap<(Vec<i32>, Vec<i32>), (Vec<f64>, Vec<f64>)>, String> {
        let n = momenta.values().next().map(|v| v.len()).unwrap_or(0);
        let mut result = HashMap::new();

        for node in self.root.preorder() {
            if node.is_leaf() { continue; }

            let frame_momenta_owned: HashMap<i32, Vec<[f64; 4]>>;
            let frame_momenta = if node.particles() == self.root.particles() {
                momenta
            } else {
                let trafos = self.boost_to_soa(node, momenta, tol, safety_checks, convention)?;
                frame_momenta_owned = trafos.transform_momenta(momenta);
                &frame_momenta_owned
            };

            let (left, right) = node.daughters().unwrap();
            let left_mom = left.momentum(frame_momenta);

            let (phis, thetas): (Vec<f64>, Vec<f64>) = (0..n).into_par_iter().map(|i| {
                let (minus_phi_rf, minus_theta_rf) = rotate_to_z_axis(&left_mom[i]);
                (-minus_phi_rf, -minus_theta_rf)
            }).unzip();

            result.insert((left.particles(), right.particles()), (phis, thetas));
        }

        Ok(result)
    }

    pub fn relative_wigner_angles(
        &self,
        other: &Topology,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
        convention: Convention,
    ) -> Result<HashMap<i32, (Vec<f64>, Vec<f64>, Vec<f64>)>, String> {
        let n = momenta.values().next().map(|v| v.len()).unwrap_or(0);
        let mut result = HashMap::new();

        for fs_node in self.final_state_nodes() {
            let particle = match fs_node {
                Node::Leaf(i) => *i,
                _ => unreachable!(),
            };

            if self.root == other.root {
                result.insert(particle, (vec![0.0; n], vec![0.0; n], vec![0.0; n]));
                continue;
            }

            let boost1_inv = self.boost_to_inverse_soa(fs_node, momenta, tol, safety_checks, convention)?;
            let boost2     = other.boost_to_soa(fs_node, momenta, tol, safety_checks, convention)?;
            let combined   = boost2.compose(&boost1_inv);

            let (phis, thetas, psis) = combined.wigner_angles_batch(tol, safety_checks)?;
            result.insert(particle, (phis, thetas, psis));
        }

        Ok(result)
    }
}
