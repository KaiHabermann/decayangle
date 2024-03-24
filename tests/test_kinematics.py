from decayangle.kinematics import rotation_matrix_2_2_x, rotation_matrix_2_2_y, rotation_matrix_2_2_z, boost_matrix_2_2_x, rapidity, build_4_4, decode_4_4, adjust_for_2pi_rotation, boost_matrix_2_2_y, boost_matrix_2_2_z
from jax import numpy as jnp
import jax
import numpy as np
jax.config.update("jax_enable_x64", True)
from decayangle.config import config as cfg
cb = cfg.backend
import pytest


def test_kinematics():
    m = 139.57018 # MeV
    p = cb.array([-400., 0, 0])
    P = cb.array([*p, (m**2 + cb.sum(p**2))**0.5])


    for matrix in [rotation_matrix_2_2_x, rotation_matrix_2_2_y, rotation_matrix_2_2_z]:
        print(matrix(1.5) @ cb.conj(matrix(1.5).T))
        assert (
                
                cb.allclose(matrix(1.5) @ cb.conj(matrix(1.5).T), cb.eye(2))
        )

    assert ( cb.allclose(
            boost_matrix_2_2_x(-rapidity(P)) @ boost_matrix_2_2_x(rapidity(P)),
            cb.eye(2)
        ) 
    )
    assert ( cb.allclose(
        boost_matrix_2_2_y(-rapidity(P)) @ boost_matrix_2_2_y(rapidity(P)),
            cb.eye(2)
        ) 
    )
    assert ( cb.allclose(
        boost_matrix_2_2_z(-rapidity(P)) @ boost_matrix_2_2_z(rapidity(P)),
            cb.eye(2)
        ) 
    )
    assert ( cb.allclose(
        rotation_matrix_2_2_x(-1.2) @ rotation_matrix_2_2_x(1.2),
        cb.eye(2)
        ) 
    )
    assert ( cb.allclose(
        rotation_matrix_2_2_y(-1.2) @ rotation_matrix_2_2_y(1.2),
        cb.eye(2)
        ) 
    )
    assert ( cb.allclose(
        rotation_matrix_2_2_z(-1.2) @ rotation_matrix_2_2_z(1.2),
        cb.eye(2)
        ) 
    )

    assert ( cb.allclose(
        rotation_matrix_2_2_y(1.2) , -rotation_matrix_2_2_y(1.2 + 2*cb.pi)
    )
    )

    assert ( cb.allclose(
        rotation_matrix_2_2_z(1.2) , -rotation_matrix_2_2_z(1.2 + 2*cb.pi)
    )
    )

    assert ( np.sum(
        abs(
        rotation_matrix_2_2_z(-np.pi/2) @ boost_matrix_2_2_y(rapidity(P)) @ rotation_matrix_2_2_z(np.pi/2) @ boost_matrix_2_2_x(-rapidity(P))
                    )
        ) < 2 + 1e-3
    )

    psi, theta, xi, theta_rf, phi_rf, psi_rf = -1.2, 2.3, 1.4, 2.5, 2.6, -2.7
    M = build_4_4(psi, theta, xi, theta_rf, phi_rf, psi_rf)
    psi_, theta_, xi_, theta_rf_, phi_rf_,  psi_rf_ = decode_4_4(M)
    assert np.allclose(M, build_4_4(psi_, theta_, xi_, theta_rf_, phi_rf_, psi_rf_))
    assert np.allclose(psi, psi_)
    assert np.allclose(theta, theta_)
    assert np.allclose(xi, xi_)
    assert np.allclose(phi_rf, phi_rf_)
    assert np.allclose(theta_rf, theta_rf_)
    assert np.allclose(psi_rf, psi_rf_)



if __name__ == "__main__":
    test_kinematics()