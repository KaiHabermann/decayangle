use std::collections::HashMap;

use crate::kinematics::{
    add4, mat4_vec4, gamma, rapidity, rotate_to_z_axis, boost_to_rest,
};
use crate::lorentz::LorentzTrafo;

// ── Tree representation ───────────────────────────────────────────────────────

/// A decay node.  Leaf nodes hold a single final-state integer.
/// Internal nodes hold two children and the sorted composite label.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Leaf(i32),
    Internal {
        label: Vec<i32>,  // sorted particle indices of all descendants
        left: Box<Node>,
        right: Box<Node>,
    },
}

impl Node {
    /// The "value" of a node — a single i32 for leaves, or the sorted vec of descendants.
    pub fn particles(&self) -> Vec<i32> {
        match self {
            Node::Leaf(i) => vec![*i],
            Node::Internal { label, .. } => label.clone(),
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Node::Leaf(_))
    }

    /// Sum 4-momenta of all leaf descendants.
    pub fn momentum<'a>(&self, momenta: &'a HashMap<i32, Vec<[f64; 4]>>) -> Vec<[f64; 4]> {
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

    /// Preorder traversal: root first, then children.
    pub fn preorder(&self) -> Vec<&Node> {
        let mut out = vec![self as &Node];
        if let Node::Internal { left, right, .. } = self {
            out.extend(left.preorder());
            out.extend(right.preorder());
        }
        out
    }
}

// ── Topology (wraps root Node) ────────────────────────────────────────────────

pub struct Topology {
    pub root: Node,
}

impl Topology {
    pub fn new(root: Node) -> Self {
        Topology { root }
    }

    /// All leaf nodes.
    pub fn final_state_nodes(&self) -> Vec<&Node> {
        self.root.preorder().into_iter().filter(|n| n.is_leaf()).collect()
    }

    // ── per-node operations ────────────────────────────────────────────────

    /// Transform all momenta by a 4×4 matrix (in place on a cloned map).
    fn transform(
        trafo: &LorentzTrafo,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
    ) -> HashMap<i32, Vec<[f64; 4]>> {
        momenta
            .iter()
            .map(|(k, batch)| {
                let transformed = batch.iter().map(|v| mat4_vec4(&trafo.matrix_4x4, v)).collect();
                (*k, transformed)
            })
            .collect()
    }

    /// Boost all momenta to the rest frame of the root node.
    pub fn to_rest_frame(
        &self,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
    ) -> HashMap<i32, Vec<[f64; 4]>> {
        let root_mom = self.root.momentum(momenta);
        let n = root_mom.len();

        // Check if already at rest (all gammas ≈ 1)
        let already_at_rest = root_mom.iter().all(|v| (gamma(v) - 1.0).abs() < tol);
        if already_at_rest {
            return momenta.clone();
        }

        momenta
            .iter()
            .map(|(k, batch)| {
                let boosted = (0..n)
                    .map(|i| boost_to_rest(&batch[i], &root_mom[i]))
                    .collect();
                (*k, boosted)
            })
            .collect()
    }

    /// `node.rotate_to(target, momenta)` — returns `(LorentzTrafo, minus_theta_rf, minus_phi_rf)`.
    /// Precondition: `node`'s momentum must be at rest.
    fn node_rotate_to(
        node: &Node,
        target: &Node,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
    ) -> Result<(Vec<LorentzTrafo>, Vec<f64>, Vec<f64>), String> {
        let node_mom = node.momentum(momenta);
        let n = node_mom.len();

        // Safety check: node must be at rest
        if safety_checks {
            for v in &node_mom {
                let g = gamma(v);
                if (g - 1.0).abs() > tol {
                    return Err(format!(
                        "gamma = {g} — node is not at rest. Call to_rest_frame first."
                    ));
                }
            }
        }

        // Identity case
        if node.particles() == target.particles() {
            let trafos = (0..n).map(|_| LorentzTrafo::identity()).collect();
            let zeros = vec![0.0; n];
            return Ok((trafos, zeros.clone(), zeros));
        }

        let target_mom = target.momentum(momenta);
        let mut trafos = Vec::with_capacity(n);
        let mut minus_theta_rfs = Vec::with_capacity(n);
        let mut minus_phi_rfs = Vec::with_capacity(n);

        for i in 0..n {
            let (minus_phi_rf, minus_theta_rf) = rotate_to_z_axis(&target_mom[i]);
            trafos.push(LorentzTrafo::from_params(0.0, 0.0, 0.0, 0.0, minus_theta_rf, minus_phi_rf));
            minus_theta_rfs.push(minus_theta_rf);
            minus_phi_rfs.push(minus_phi_rf);
        }

        Ok((trafos, minus_theta_rfs, minus_phi_rfs))
    }

    /// `node.boost(target, momenta)` — helicity convention.
    fn node_boost_helicity(
        node: &Node,
        target: &Node,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
    ) -> Result<Vec<LorentzTrafo>, String> {
        let node_mom = node.momentum(momenta);
        let n = node_mom.len();

        if safety_checks {
            for v in &node_mom {
                let g = gamma(v);
                if (g - 1.0).abs() > tol {
                    return Err(format!(
                        "gamma = {g} — node is not at rest. Call to_rest_frame first."
                    ));
                }
            }
        }

        // Identity
        if node.particles() == target.particles() {
            return Ok((0..n).map(|_| LorentzTrafo::identity()).collect());
        }

        let (left, _right) = node.daughters().ok_or("node has no daughters")?;
        let target_is_first_daughter = left.particles() == target.particles();

        let target_mom = target.momentum(momenta);
        let (rotation_trafos, _, _) = Self::node_rotate_to(node, left, momenta, tol, safety_checks)?;

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let xi = -rapidity(&target_mom[i]);
            let boost = LorentzTrafo::from_params(0.0, 0.0, xi, 0.0, 0.0, 0.0);

            let rotation = if target_is_first_daughter {
                rotation_trafos[i].clone()
            } else {
                // Apply R_y(-pi) before the rotation to flip to particle 2
                let flip = LorentzTrafo::from_params(0.0, 0.0, 0.0, 0.0, -std::f64::consts::PI, 0.0);
                flip.compose(&rotation_trafos[i])
            };

            out.push(boost.compose(&rotation));
        }
        Ok(out)
    }

    // ── path-following boost ──────────────────────────────────────────────

    /// Find the path from root to target (sequence of particle-sets).
    fn path_to<'a>(&'a self, target: &Node) -> Vec<&'a Node> {
        fn dfs<'a>(current: &'a Node, target_particles: &[i32], path: &mut Vec<&'a Node>) -> bool {
            path.push(current);
            if current.particles() == target_particles {
                return true;
            }
            if let Some((left, right)) = current.daughters() {
                if dfs(left, target_particles, path) {
                    return true;
                }
                if dfs(right, target_particles, path) {
                    return true;
                }
            }
            path.pop();
            false
        }
        let tp = target.particles();
        let mut path = Vec::new();
        dfs(&self.root, &tp, &mut path);
        path
    }

    /// Boost from root to `target`, composing trafos along the path.
    /// Mirrors `Topology.boost(target, momenta, convention="helicity")`.
    pub fn boost_to(
        &self,
        target: &Node,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
    ) -> Result<Vec<LorentzTrafo>, String> {
        let path = self.path_to(target);
        let n = momenta.values().next().map(|v| v.len()).unwrap_or(0);

        if path.len() < 2 {
            // target is root — return identity
            return Ok((0..n).map(|_| LorentzTrafo::identity()).collect());
        }

        // First step: root → path[1]
        let mut step_trafos =
            Self::node_boost_helicity(path[0], path[1], momenta, tol, safety_checks)?;
        let mut composed: Vec<LorentzTrafo> = step_trafos.clone();

        // Transform momenta for next step
        let mut current_momenta = momenta.clone();
        for i in 0..n {
            let single: HashMap<i32, Vec<[f64; 4]>> = current_momenta
                .iter()
                .map(|(k, batch)| {
                    let v = mat4_vec4(&step_trafos[i].matrix_4x4, &batch[i]);
                    (*k, vec![v])
                })
                .collect();
            // We'll do this properly with a full-batch transform below
            let _ = single; // placeholder
        }
        current_momenta = Self::transform_batch(&step_trafos, &current_momenta);

        // Remaining steps
        for step_idx in 1..path.len() - 1 {
            step_trafos = Self::node_boost_helicity(
                path[step_idx],
                path[step_idx + 1],
                &current_momenta,
                tol,
                safety_checks,
            )?;
            current_momenta = Self::transform_batch(&step_trafos, &current_momenta);
            for i in 0..n {
                composed[i] = step_trafos[i].compose(&composed[i]);
            }
        }

        Ok(composed)
    }

    /// Inverse boost (more precise: compose individual inverses in reverse order).
    pub fn boost_to_inverse(
        &self,
        target: &Node,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
    ) -> Result<Vec<LorentzTrafo>, String> {
        let path = self.path_to(target);
        let n = momenta.values().next().map(|v| v.len()).unwrap_or(0);

        if path.len() < 2 {
            return Ok((0..n).map(|_| LorentzTrafo::identity()).collect());
        }

        // Collect all step trafos
        let mut all_steps: Vec<Vec<LorentzTrafo>> = Vec::new();
        let mut current_momenta = momenta.clone();

        for step_idx in 0..path.len() - 1 {
            let step_trafos = Self::node_boost_helicity(
                path[step_idx],
                path[step_idx + 1],
                &current_momenta,
                tol,
                safety_checks,
            )?;
            current_momenta = Self::transform_batch(&step_trafos, &current_momenta);
            all_steps.push(step_trafos);
        }

        // Mirror Python's inverse construction:
        //   inverse_trafo = step_0^{-1}
        //   inverse_trafo = inverse_trafo @ step_1^{-1}
        //   ...
        // giving step_0^{-1} @ step_1^{-1} @ ... = (step_n @ ... @ step_0)^{-1}
        let mut result: Vec<LorentzTrafo> = all_steps[0].iter().map(|t| t.inverse()).collect();
        for step in all_steps[1..].iter() {
            for i in 0..n {
                result[i] = result[i].compose(&step[i].inverse());
            }
        }
        Ok(result)
    }

    /// Apply a batch of per-event trafos to all particle momenta.
    fn transform_batch(
        trafos: &[LorentzTrafo],
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
    ) -> HashMap<i32, Vec<[f64; 4]>> {
        momenta
            .iter()
            .map(|(k, batch)| {
                let transformed = batch
                    .iter()
                    .zip(trafos.iter())
                    .map(|(v, t)| mat4_vec4(&t.matrix_4x4, v))
                    .collect();
                (*k, transformed)
            })
            .collect()
    }

    // ── public angle computation ──────────────────────────────────────────

    /// Helicity angles for all internal nodes.
    /// Returns a map from `(isobar_label, spectator_label)` encoded as Vec<i32>
    /// to `(phi_batch, theta_batch)` each of length N.
    pub fn helicity_angles(
        &self,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
    ) -> Result<HashMap<(Vec<i32>, Vec<i32>), (Vec<f64>, Vec<f64>)>, String> {
        let n = momenta.values().next().map(|v| v.len()).unwrap_or(0);
        let mut result = HashMap::new();

        for node in self.root.preorder() {
            if node.is_leaf() {
                continue;
            }

            let momenta_in_frame: HashMap<i32, Vec<[f64; 4]>>;
            let frame_momenta = if node.particles() == self.root.particles() {
                momenta
            } else {
                let node_leaf = node; // use as target
                let trafos = self.boost_to(node_leaf, momenta, tol, safety_checks)?;
                momenta_in_frame = Self::transform_batch(&trafos, momenta);
                &momenta_in_frame
            };

            let (left, _right) = node.daughters().unwrap();

            let left_mom = left.momentum(frame_momenta);
            let mut phis = Vec::with_capacity(n);
            let mut thetas = Vec::with_capacity(n);

            for i in 0..n {
                let (minus_phi_rf, minus_theta_rf) = rotate_to_z_axis(&left_mom[i]);
                phis.push(-minus_phi_rf);
                thetas.push(-minus_theta_rf);
            }

            let (right_node, _) = node.daughters().unwrap();
            let spectator = node.daughters().unwrap().1;
            result.insert((right_node.particles(), spectator.particles()), (phis, thetas));
        }

        Ok(result)
    }

    /// Relative Wigner angles between `self` and `other` for all final-state particles.
    /// Returns a map from particle index (i32) to `(phi_rf, theta_rf, psi_rf)` each of length N.
    pub fn relative_wigner_angles(
        &self,
        other: &Topology,
        momenta: &HashMap<i32, Vec<[f64; 4]>>,
        tol: f64,
        safety_checks: bool,
    ) -> Result<HashMap<i32, (Vec<f64>, Vec<f64>, Vec<f64>)>, String> {
        let n = momenta.values().next().map(|v| v.len()).unwrap_or(0);
        let mut result = HashMap::new();

        for fs_node in self.final_state_nodes() {
            let particle = match fs_node {
                crate::topology::Node::Leaf(i) => *i,
                _ => unreachable!(),
            };

            if self.root == other.root {
                // Same topology — identity rotation for all particles
                result.insert(particle, (vec![0.0; n], vec![0.0; n], vec![0.0; n]));
                continue;
            }

            let boost1_inv = self.boost_to_inverse(fs_node, momenta, tol, safety_checks)?;
            let boost2 = other.boost_to(fs_node, momenta, tol, safety_checks)?;

            let mut phis = Vec::with_capacity(n);
            let mut thetas = Vec::with_capacity(n);
            let mut psis = Vec::with_capacity(n);

            for i in 0..n {
                let combined = boost2[i].compose(&boost1_inv[i]);
                let (phi_rf, theta_rf, psi_rf) =
                    combined.wigner_angles(tol, safety_checks)?;
                phis.push(phi_rf);
                thetas.push(theta_rf);
                psis.push(psi_rf);
            }

            result.insert(particle, (phis, thetas, psis));
        }

        Ok(result)
    }
}
