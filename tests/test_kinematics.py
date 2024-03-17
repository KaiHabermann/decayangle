from DecayAngle.kinematics import *
from jax import numpy as jnp
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
def test_kinematics():
    m = 139.57018 # MeV
    p = jnp.array([-400., 0, 0])
    P = jnp.array([*p, (m**2 + jnp.sum(p**2))**0.5])

    assert ( jnp.sum(
        abs(
            boost_matrix_2_2_x(-rapidity(P)) @ boost_matrix_2_2_x(rapidity(P))
            )
        ) < 2 + 1e-3
    )
    assert ( jnp.sum(
        abs(
        boost_matrix_2_2_y(-rapidity(P)) @ boost_matrix_2_2_y(rapidity(P))
        )
        ) < 2 + 1e-10
    )
    assert ( jnp.sum(
        abs(
        boost_matrix_2_2_z(-rapidity(P)) @ boost_matrix_2_2_z(rapidity(P))
        )
        ) < 2 + 1e-10
    )
    assert ( jnp.allclose(
        rotation_matrix_2_2_x(-1.2) @ rotation_matrix_2_2_x(1.2),
        jnp.eye(2)
        ) 
    )
    assert ( jnp.allclose(
        rotation_matrix_2_2_y(-1.2) @ rotation_matrix_2_2_y(1.2),
        jnp.eye(2)
        ) 
    )
    assert ( jnp.allclose(
        rotation_matrix_2_2_z(-1.2) @ rotation_matrix_2_2_z(1.2),
        jnp.eye(2)
        ) 
    )


    print(
        rotation_matrix_2_2_y(1.2) ,"\n" ,-rotation_matrix_2_2_y(1.2 + 2*jnp.pi)
    )
    assert ( jnp.allclose(
        rotation_matrix_2_2_y(1.2) , -rotation_matrix_2_2_y(1.2 + 2*jnp.pi)
    )
    )

    print(
        rotation_matrix_2_2_z(1.2) ,"\n" ,-rotation_matrix_2_2_z(1.2 + 2*jnp.pi)
    )
    assert ( jnp.allclose(
        rotation_matrix_2_2_z(1.2) , -rotation_matrix_2_2_z(1.2 + 2*jnp.pi)
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