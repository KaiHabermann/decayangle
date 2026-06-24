mod kinematics;
mod lorentz;
mod topology;


use std::collections::HashMap;
use numpy::{IntoPyArray, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyList};

use topology::{Node, Topology, Convention};

// ── Python-tuple → Rust Node parser ──────────────────────────────────────────

fn parse_node(ob: &Bound<'_, PyAny>) -> PyResult<Node> {
    if let Ok(i) = ob.extract::<i32>() {
        return Ok(Node::Leaf(i));
    }
    let tup = ob.downcast::<PyTuple>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(
            "topology must be a nested tuple of ints",
        ))?;
    if tup.len() != 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "each internal node must be a 2-element tuple",
        ));
    }
    let left  = parse_node(&tup.get_item(0)?)?;
    let right = parse_node(&tup.get_item(1)?)?;

    // Preserve left-to-right order from the tuple — sorting is already encoded
    // by the Python topology's ordering function before the tuple is passed in.
    let label: Vec<i32> = left.particles().into_iter().chain(right.particles()).collect();

    Ok(Node::Internal {
        label,
        left:  Box::new(left),
        right: Box::new(right),
    })
}

/// Convert a Python momenta dict `{int: ndarray of shape (N, 4)}` to Rust.
fn parse_momenta(py_momenta: &Bound<'_, PyDict>) -> PyResult<HashMap<i32, Vec<[f64; 4]>>> {
    let mut out = HashMap::new();
    for (k, v) in py_momenta.iter() {
        let key: i32 = k.extract()?;
        let arr: PyReadonlyArray2<f64> = v.extract()?;
        let shape = arr.shape();
        if shape.len() != 2 || shape[1] != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("momenta[{key}] must have shape (N, 4), got {shape:?}"),
            ));
        }
        let n = shape[0];
        let mut batch = Vec::with_capacity(n);
        for i in 0..n {
            batch.push([
                *arr.get([i, 0]).unwrap(),
                *arr.get([i, 1]).unwrap(),
                *arr.get([i, 2]).unwrap(),
                *arr.get([i, 3]).unwrap(),
            ]);
        }
        out.insert(key, batch);
    }
    Ok(out)
}

/// Convert a Python list of ints to a Rust Vec<i32>.
fn parse_massless(py_massless: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
    if py_massless.is_none() {
        return Ok(vec![]);
    }
    let list = py_massless.downcast::<PyList>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("massless must be a list of ints"))?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        out.push(item.extract::<i32>()?);
    }
    Ok(out)
}

// ── Public Python functions ───────────────────────────────────────────────────

#[pyfunction]
#[pyo3(signature = (topology_tuple, momenta, tol=1e-10, safety_checks=true, convention="helicity", massless=None))]
fn helicity_angles_rust<'py>(
    py: Python<'py>,
    topology_tuple: &Bound<'py, PyAny>,
    momenta: &Bound<'py, PyDict>,
    tol: f64,
    safety_checks: bool,
    convention: &str,
    massless: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let root_node = parse_node(topology_tuple)?;
    let topo = Topology::new(root_node);
    let rust_momenta = parse_momenta(momenta)?;
    let conv = Convention::from_str(convention)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let massless_ids = match massless {
        Some(m) => parse_massless(m)?,
        None => vec![],
    };

    let angles = topo
        .helicity_angles(&rust_momenta, tol, safety_checks, conv, &massless_ids)
        .map_err(|e| {
            if e.contains("not at rest") {
                pyo3::exceptions::PyValueError::new_err(e)
            } else {
                pyo3::exceptions::PyRuntimeError::new_err(e)
            }
        })?;

    let result = PyDict::new_bound(py);
    for ((isobar, spectator), (phis, thetas)) in angles {
        let isobar_tuple   = PyTuple::new_bound(py, isobar.iter());
        let spectator_tuple = PyTuple::new_bound(py, spectator.iter());
        let key = PyTuple::new_bound(py, [isobar_tuple.as_any(), spectator_tuple.as_any()]);
        let inner = PyDict::new_bound(py);
        inner.set_item("phi",   phis.into_pyarray_bound(py))?;
        inner.set_item("theta", thetas.into_pyarray_bound(py))?;
        result.set_item(key, inner)?;
    }
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (topology1_tuple, topology2_tuple, momenta, tol=1e-10, safety_checks=true, convention="helicity", massless=None))]
fn wigner_angles_rust<'py>(
    py: Python<'py>,
    topology1_tuple: &Bound<'py, PyAny>,
    topology2_tuple: &Bound<'py, PyAny>,
    momenta: &Bound<'py, PyDict>,
    tol: f64,
    safety_checks: bool,
    convention: &str,
    massless: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let root1 = parse_node(topology1_tuple)?;
    let root2 = parse_node(topology2_tuple)?;
    let topo1 = Topology::new(root1);
    let topo2 = Topology::new(root2);
    let rust_momenta = parse_momenta(momenta)?;
    let conv = Convention::from_str(convention)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    let massless_ids = match massless {
        Some(m) => parse_massless(m)?,
        None => vec![],
    };

    let angles = topo1
        .relative_wigner_angles(&topo2, &rust_momenta, tol, safety_checks, conv, &massless_ids)
        .map_err(|e| {
            if e.contains("not at rest") {
                pyo3::exceptions::PyValueError::new_err(e)
            } else {
                pyo3::exceptions::PyRuntimeError::new_err(e)
            }
        })?;

    let result = PyDict::new_bound(py);
    for (particle, (phis, thetas, psis)) in angles {
        let inner = PyDict::new_bound(py);
        inner.set_item("phi_rf",   phis.into_pyarray_bound(py))?;
        inner.set_item("theta_rf", thetas.into_pyarray_bound(py))?;
        inner.set_item("psi_rf",   psis.into_pyarray_bound(py))?;
        result.set_item(particle, inner)?;
    }
    Ok(result)
}

// ── Module registration ───────────────────────────────────────────────────────

#[pymodule]
fn decayangle_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(helicity_angles_rust, m)?)?;
    m.add_function(wrap_pyfunction!(wigner_angles_rust, m)?)?;
    Ok(())
}
