from typing import Tuple, Union, Optional
from functools import partial
from jax import numpy as jnp
import numpy as np
from decayangle.config import config as cfg
from decayangle.numerics_helpers import save_arccos

cb = cfg.backend


def boost_matrix_2_2_x(
    xi: Union[float, jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    r"""
    Build a 2x2 boost matrix in the x-direction
    Args:
        xi (float): rapidity of the boost

    Returns:
        Union[jnp.array, np.array]: the 2x2 boost matrix with shape (...,2,2)
    """
    zero = cb.zeros_like(xi)
    one = cb.ones_like(xi)
    sigma_x = cb.array([[zero, one], [one, zero]])
    eye = cb.array([[one, zero], [zero, one]])
    return cb.moveaxis(
        (cb.cosh(xi / 2) * eye + cb.sinh(xi / 2) * sigma_x), [0, 1], [-2, -1]
    )


def boost_matrix_2_2_y(
    xi: Union[float, jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    r"""
    Build a 2x2 boost matrix in the y-direction
    Args:
        xi (float): rapidity of the boost

    Returns:
        Union[jnp.array, np.array]: the 2x2 boost matrix with shape (...,2,2)
    """
    zero = cb.zeros_like(xi)
    one = cb.ones_like(xi)
    sigma_y = cb.array([[zero, -1j * one], [1j * one, zero]])
    eye = cb.array([[one, zero], [zero, one]])
    return cb.moveaxis(
        (cb.cosh(xi / 2) * eye + cb.sinh(xi / 2) * sigma_y), [0, 1], [-2, -1]
    )


def boost_matrix_2_2_z(
    xi: Union[float, jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    r"""
    Build a 2x2 boost matrix in the z-direction
    Args:
        xi (float): rapidity of the boost

    Returns:
        Union[jnp.array, np.array]: the 2x2 boost matrix with shape (...,2,2)
    """
    zero = cb.zeros_like(xi)
    one = cb.ones_like(xi)
    eye = cb.array([[one, zero], [zero, one]])
    sigma_z = cb.array([[one, zero], [zero, -one]])
    return cb.moveaxis(
        (cb.cosh(xi / 2) * eye + cb.sinh(xi / 2) * sigma_z), [0, 1], [-2, -1]
    )


def rotate_to_z_axis(v: Union[jnp.array, np.array]) -> Union[jnp.array, np.array]:
    """Given a vector, calculate the angles to rotate it to the z-axis
    This is done by rotating into the x-z plane (rotation around z axis by -psi_rf) and then into the z-axis (rotation around y axis by -theta_rf)

    Args:
        v (Union[jnp.array, np.array]): the 4 vector to be rotated

    Returns:
        Union[jnp.array, np.array]: the rotation angles around first z and then y axis
    """
    v = cb.array(v)
    phi_rf = cb.arctan2(y_component(v), x_component(v))
    theta_rf = cb.arccos(z_component(v) / p(v))
    return -phi_rf, -theta_rf


def rotation_matrix_2_2_x(
    theta: Union[float, jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    """Build a 2x2 rotation matrix around the x-axis

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jnp.array, np.array]: the rotation matrix with shape (...,2,2)
    """
    zero = cb.zeros_like(theta)
    one = cb.ones_like(theta)
    eye = cb.array([[one, zero], [zero, one]])
    sgma_x = cb.array([[zero, one], [one, zero]])
    return cb.moveaxis(
        (cb.cos(theta / 2) * eye - 1j * cb.sin(theta / 2) * sgma_x), [0, 1], [-2, -1]
    )


def rotation_matrix_2_2_y(
    theta: Union[float, jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    """Build a 2x2 rotation matrix around the y-axis

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jnp.array, np.array]: the rotation matrix with shape (...,2,2)
    """
    zero = cb.zeros_like(theta)
    one = cb.ones_like(theta)
    eye = cb.array([[one, zero], [zero, one]])
    sgma_y = cb.array([[zero, -1j * one], [1j * one, zero]])
    return cb.moveaxis(
        (cb.cos(theta / 2) * eye - 1j * cb.sin(theta / 2) * sgma_y), [0, 1], [-2, -1]
    )


def rotation_matrix_2_2_z(
    theta: Union[float, jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    """Build a 2x2 rotation matrix around the z-axis

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jnp.array, np.array]: the rotation matrix with shape (...,2,2)
    """
    zero = cb.zeros_like(theta)
    one = cb.ones_like(theta)
    eye = cb.array([[one, zero], [zero, one]])
    sgma_z = cb.array([[one, zero], [zero, -one]])
    return cb.moveaxis(
        (cb.cos(theta / 2) * eye - 1j * cb.sin(theta / 2) * sgma_z), [0, 1], [-2, -1]
    )


def boost_matrix_4_4_z(
    xi: Union[float, jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    r"""Build a 4x4 boost matrix in the z-direction

    Args:
        xi (float): rapidity of the boost

    Returns:
        Union[jnp.array, np.array]: the 4x4 boost matrix with shape (...,4,4)
    """
    zero = cb.zeros_like(xi)
    one = cb.ones_like(xi)
    gma = cb.cosh(xi)
    beta_gamma = cb.sinh(xi)
    return cb.moveaxis(
        cb.array(
            [
                [
                    one,
                    zero,
                    zero,
                    zero,
                ],
                [
                    zero,
                    one,
                    zero,
                    zero,
                ],
                [
                    zero,
                    zero,
                    gma,
                    beta_gamma,
                ],
                [
                    zero,
                    zero,
                    beta_gamma,
                    gma,
                ],
            ]
        ),
        [0, 1],
        [-2, -1],
    )


def rotation_matrix_4_4_y(
    theta: Union[float, jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    """Build a 4x4 rotation matrix around the y-axis

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jnp.array, np.array]: the rotation matrix with shape (...,4,4)
    """
    zero = cb.zeros_like(theta)
    one = cb.ones_like(theta)
    return cb.moveaxis(
        cb.array(
            [
                [
                    cb.cos(theta),
                    zero,
                    cb.sin(theta),
                    zero,
                ],
                [
                    zero,
                    one,
                    zero,
                    zero,
                ],
                [
                    -cb.sin(theta),
                    zero,
                    cb.cos(theta),
                    zero,
                ],
                [zero, zero, zero, one],
            ]
        ),
        [0, 1],
        [-2, -1],
    )


def rotation_matrix_4_4_z(theta: float) -> Union[jnp.array, np.array]:
    """Build a 4x4 rotation matrix around the z-axis

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jax.numpy.ndarray, np.array]: the rotation matrix with shape (...,4,4)
    """
    zero = cb.zeros_like(theta)
    one = cb.ones_like(theta)
    return cb.moveaxis(
        cb.array(
            [
                [
                    cb.cos(theta),
                    -cb.sin(theta),
                    zero,
                    zero,
                ],
                [
                    cb.sin(theta),
                    cb.cos(theta),
                    zero,
                    zero,
                ],
                [
                    zero,
                    zero,
                    one,
                    zero,
                ],
                [zero, zero, zero, one],
            ]
        ),
        [0, 1],
        [-2, -1],
    )


def build_2_2(phi, theta, xi, phi_rf, theta_rf, psi_rf):
    r"""Build a 2x2 matrix from the 6 kinematic parameters

    Args:
        psi (float): the rotation angle around the z-axis
        theta (float): the rotation angle around the y-axis
        xi (float): the rapidity of the boost
        phi_rf (float): the rotation angle around the z-axis in the rest frame
        theta_rf (float): the rotation angle around the y-axis in the rest frame
        psi_rf (float): the rotation angle around the z-axis in the rest frame

    Returns:
        jax.numpy.ndarray: the 2x2 matrix
    """
    return (
        rotation_matrix_2_2_z(phi)
        @ rotation_matrix_2_2_y(theta)
        @ boost_matrix_2_2_z(xi)
        @ rotation_matrix_2_2_z(phi_rf)
        @ rotation_matrix_2_2_y(theta_rf)
        @ rotation_matrix_2_2_z(psi_rf)
    )


def build_4_4(phi, theta, xi, phi_rf, theta_rf, psi_rf):
    r"""Build a 4x4 matrix from the 6 kinematic parameters

    Args:
        psi (float): the rotation angle around the z-axis
        theta (float): the rotation angle around the y-axis
        xi (float): the rapidity of the boost
        phi_rf (float): the rotation angle around the z-axis in the rest frame
        theta_rf (float): the rotation angle around the y-axis in the rest frame
        psi_rf (float): the rotation angle around the z-axis in the rest frame

    Returns:
        jax.numpy.ndarray: the 4x4 matrix
    """
    return (
        rotation_matrix_4_4_z(phi)
        @ rotation_matrix_4_4_y(theta)
        @ boost_matrix_4_4_z(xi)
        @ rotation_matrix_4_4_z(phi_rf)
        @ rotation_matrix_4_4_y(theta_rf)
        @ rotation_matrix_4_4_z(psi_rf)
    )


def decode_rotation_4x4(
    rotation_matrix: Union[jnp.array, np.array]
) -> Tuple[float, float, float]:
    r"""decode a 4x4 rotation matrix into the 3 rotation angles

    Args:
        matrix (jax.numpy.ndarray): the 4x4 rotation matrix

    Returns:
        Tuple[float, float, float]: the 3 rotation angles (phi, theta, psi)
    """
    phi = cb.arctan2(rotation_matrix[..., 1, 2], rotation_matrix[..., 0, 2])
    theta = save_arccos(rotation_matrix[..., 2, 2])
    psi = cb.arctan2(rotation_matrix[..., 2, 1], -rotation_matrix[..., 2, 0])
    return phi, theta, psi


def decode_4_4(
    matrix, tol: Optional[float] = None
) -> Tuple[Union[float, np.array, jnp.array]]:
    r"""decode a 4x4 matrix into the 6 kinematic parameters

    Args:
        matrix (jax.numpy.ndarray): the 4x4 matrix
        tol (float, optional): the tolerance for the gamma factor. When not given the default value from the config ('gamma_tolerance') will be used.
    """
    if tol is None:
        tol = cfg.gamma_tolerance

    m = 1.0
    v_0 = cb.array([0, 0, 0, m])

    v = matrix @ v_0
    w = time_component(v)
    abs_mom = p(v)
    gma = w / m

    # gamma can be smaller than 1 due to numerical errors
    # for large deviations we will raise an exception
    gma = cb.where((abs(gma) < 1) & (abs(gma - 1) < tol), 1, gma)
    if cb.any(gma < 1):
        cfg.raise_if_safety_on(
            ValueError(
                f"gamma is {gma}, which is less than 1. This is not a valid Lorentz transformation"
            )
        )

    xi = cb.arccosh(gma)
    phi = cb.arctan2(y_component(v), x_component(v))

    cosine_input = cb.where(abs(abs_mom) <= tol, 0, z_component(v) / abs_mom)
    theta = cb.arccos(cosine_input)

    m_rf = (
        boost_matrix_4_4_z(-xi)
        @ rotation_matrix_4_4_y(-theta)
        @ rotation_matrix_4_4_z(-phi)
        @ matrix
    )
    # check for the special case of no absolute boost
    phi_rf, theta_rf, psi_rf = decode_rotation_4x4(m_rf[..., :3, :3])
    phi_rf_no_boost, theta_rf_no_boost, psi_rf_no_boost = decode_rotation_4x4(
        matrix[..., :3, :3]
    )
    phi_rf = cb.where(abs(gma - 1) < tol, phi_rf_no_boost, phi_rf)
    theta_rf = cb.where(abs(gma - 1) < tol, theta_rf_no_boost, theta_rf)
    psi_rf = cb.where(abs(gma - 1) < tol, psi_rf_no_boost, psi_rf)
    phi = cb.where(abs(gma - 1) < tol, 0, phi)
    theta = cb.where(abs(gma - 1) < tol, 0, theta)
    xi = cb.where(abs(gma - 1) < tol, 0, xi)

    is_unity = cb.all(cb.all(cb.isclose(matrix, cb.eye(4)), axis=-1), axis=-1)

    def check_unity(val):
        return cb.where(is_unity, 0, val)

    # replace the values with 0 if the matrix is unity
    phi, theta, xi, phi_rf, theta_rf, psi_rf = [
        check_unity(val) for val in [phi, theta, xi, phi_rf, theta_rf, psi_rf]
    ]

    return phi, theta, xi, phi_rf, theta_rf, psi_rf


def adjust_for_2pi_rotation(
    m_original_2x2, phi, theta, xi, phi_rf, theta_rf, psi_rf
) -> Tuple[Union[jnp.array, np.array]]:
    """Adjust the rotation angles for the 2pi rotation ambiguity

    Args:
        M_original_2x2 (jax.numpy.ndarray): the original 2x2 matrix
        psi (float): the recovered psi angle from the 4x4 matrix
        theta (float): the recovered theta angle from the 4x4 matrix
        xi (float): the recovered rapidity angle from the 4x4 matrix
        theta_rf (float): the recovered theta_rf angle from the 4x4 matrix
        phi_rf (float): the recovered phi_rf angle from the 4x4 matrix
        psi_rf (float): the recovered psi_rf angle from the 4x4 matrix

    Returns:
        tuple: the adjusted rotation angles
    """
    new_2x2 = build_2_2(phi, theta, xi, phi_rf, theta_rf, psi_rf)

    not_two_pi_shifted = cb.all(
        cb.all(cb.isclose(m_original_2x2, new_2x2, rtol=cfg.shift_precision), axis=-1),
        axis=-1,
    )

    two_pi_shifted = cb.all(
        cb.all(cb.isclose(m_original_2x2, -new_2x2, rtol=cfg.shift_precision), axis=-1),
        axis=-1,
    )
    if cb.any(not_two_pi_shifted & two_pi_shifted):
        cfg.raise_if_safety_on(
            ValueError(
                f"The 2x2 matrix does not match the reconstruced parameters!"
                f"This can happen due to numerical errors."
                f"The original matrix is {m_original_2x2} and the reconstructed matrix is {new_2x2}"
                f"Difference is {m_original_2x2 - new_2x2}"
                f"Parameters are {phi}, {theta}, {xi}, {theta_rf}, {phi_rf}, {psi_rf}"
            )
        )

    phi_rf = cb.where(two_pi_shifted, phi_rf + 2 * cb.pi, phi_rf)
    return phi, theta, xi, phi_rf, theta_rf, psi_rf


def spatial_components(vector):
    """Return spatial components of the input Lorentz vector

    :param vector: input Lorentz vector
    :returns: tensor of spatial components

    """
    return vector[..., 0:3]


def time_component(vector):
    """Return time component of the input Lorentz vector

    :param vector: input Lorentz vector (where indexes 0-2 are space, index 3 is time)
    :returns: vector of time components

    """
    return vector[..., 3]


def x_component(vector):
    """Return spatial X component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of X-components

    """
    return vector[..., 0]


def y_component(vector):
    """Return spatial Y component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of Y-components

    """
    return vector[..., 1]


def z_component(vector):
    """Return spatial Z component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of Z-components

    """
    return vector[..., 2]


def pt(vector):
    """Return transverse (X-Y) component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of transverse components

    """
    return cb.sqrt(x_component(vector) ** 2 + y_component(vector) ** 2)


def eta(vector):
    """Return pseudorapidity component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of pseudorapidity components

    """
    return -cb.log(pt(vector) / 2.0 / z_component(vector))


def build_vector(x, y, z):
    """
    Make a 3-vector from components. Components are stacked along the last index.

    :param x: x-component of the vector
    :param y: y-component of the vector
    :param z: z-component of the vector
    :returns: 3-vector
    """
    return cb.stack([x, y, z], axis=-1)


def mass_squared(vector):
    """
    Calculate squared invariant mass scalar for Lorentz 4-momentum vector

    :param vector: input Lorentz momentum vector
    :returns: scalar invariant mass squared

    """
    return cb.sum(vector * vector * metric_tensor(), -1)


def metric_tensor():
    """
    Constant metric tensor for Lorentz space

    :returns: Metric tensor
    """
    return cb.array([-1.0, -1.0, -1.0, 1.0], dtype=cb.float64)


def lorentz_vector(space, time):
    """
    Make a Lorentz vector from spatial and time components

    :param space: 3-vector of spatial components
    :param time: time component
    :returns: Lorentz vector

    """
    return cb.concatenate([space, cb.stack([time], axis=-1)], axis=-1)


def from_mass_and_momentum(m, mom):
    """
    Create a Lorentz vector from mass and momentum vector

    :param mass: Mass scalar
    :param momentum: 3-vector of momentum components
    :returns: Lorentz vector

    """
    return lorentz_vector(mom, cb.sqrt(norm(mom) ** 2 + m**2))


def mass(vector):
    """
    Calculate mass scalar for Lorentz 4-momentum vector

    :param vector: input Lorentz momentum vector
    :returns: scalar invariant mass

    """
    return cb.sqrt(mass_squared(vector))


def gamma(momentum):
    r"""calculate gamma factor

    Args:
        p (jax.numpy.ndarray): momentum 4-vector
    """
    return time_component(momentum) / mass(momentum)


def beta(momentum):
    r"""calculate beta factor

    Args:
        p (jax.numpy.ndarray): momentum 4-vector
    """
    return p(momentum) / time_component(momentum)


def rapidity(momentum):
    r"""calculate rapidity

    Args:
        p (jax.numpy.ndarray): momentum 4-vector
    """
    b = beta(momentum)
    return 0.5 * cb.log((b + 1) / (1 - b))


def norm(vec: Union[jnp.array, np.array]):
    """
    Calculate norm of 3-vector

    Args:
        vec (Union[jnp.array, np.array]): 3-vector
    Returns:
        Union[jnp.array, np.array]: norm of the 3-vector

    """
    return cb.sqrt(cb.sum(vec * vec, -1))


def p(vector: Union[jnp.array, np.array]) -> Union[jnp.array, np.array]:
    """
    Calculate absolute value of the 4-momentum

    Args:
        vector (Union[jnp.array, np.array]): 4-momentum vector
    Returns:
        Union[jnp.array, np.array]: absolute value of the 4-momentum

    """
    return norm(spatial_components(vector))


def psi(vector: Union[jnp.array, np.array]) -> Union[jnp.array, np.array]:
    """Return azimuthal angle component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of azimuthal angle components

    """
    return cb.arctan2(y_component(vector), x_component(vector))


def scalar_product(
    vec1: Union[jnp.array, np.array], vec2: Union[jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    """
    Calculate scalar product of two 3-vectors

    Args:
        vec1 (Union[jnp.array, np.array]): first 3-vector
        vec2 (Union[jnp.array, np.array]): second 3-vector

    Returns:
        Union[jnp.array, np.array]: scalar product of the two vectors

    """
    return cb.sum(vec1 * vec2, -1)


def scalar(x: Union[jnp.array, np.array]) -> Union[jnp.array, np.array]:
    """
    Create a scalar (array with only one component in last index) which can be used
    to e.g. scale a vector.

    Args:
        x (Union[jnp.array, np.array]): input array
    Returns:
        Union[jnp.array, np.array]: scalar array

    """
    return cb.stack([x], axis=-1)


def lorentz_boost(
    vector: Union[jnp.array, np.array], boostvector: Union[jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    """
    Perform Lorentz boost of the 4-vector vector using boost vector boostvector.
    We do not use the matrices here, to make things a little easier

    Args:
        vector (Union[jnp.array, np.array]): 4-vector to be boosted
        boostvector (Union[jnp.array, np.array]): Boost momentum 4-vector in the same frame as vector
                        (should have nonzero mass!)
    Returns:
        Union[jnp.array, np.array]: 4-vector boosted to the boostvector frame

    """
    boost = spatial_components(boostvector)
    b2 = scalar_product(boost, boost)
    gma = 1.0 / cb.sqrt(1.0 - b2)
    gma2 = (gma - 1.0) / b2
    gma2 = cb.where(cb.isclose(b2, 0), 0, gma2)
    ve = time_component(vector)
    vp = spatial_components(vector)
    bp = scalar_product(vp, boost)
    vp2 = vp + scalar(gma2 * bp + gma * ve) * boost
    ve2 = gma * (ve + bp)

    return lorentz_vector(vp2, ve2)


def boost_to_rest(
    vector: Union[jnp.array, np.array], boostvector: Union[jnp.array, np.array]
) -> Union[jnp.array, np.array]:
    """
    Perform Lorentz boost to the rest frame of the
    4-vector boostvector.

    Args:
        vector (Union[jnp.array, np.array]): 4-vector to be boosted
        boostvector (Union[jnp.array, np.array]): Boost momentum 4-vector in the same frame as vector
                        (should have nonzero mass!)
    Returns:
        Union[jnp.array, np.array]: 4-vector boosted to the rest frame of boostvector

    """
    boost = -spatial_components(boostvector) / scalar(time_component(boostvector))
    return lorentz_boost(vector, boost)
