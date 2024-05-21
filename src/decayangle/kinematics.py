from typing import Tuple, Union
from functools import partial
from jax import numpy as jnp
import numpy as np
from decayangle.config import config as cfg
from decayangle.numerics_helpers import save_arccos

cb = cfg.backend


@partial(cb.vectorize, signature="()->(2,2)")
def boost_matrix_2_2_x(xi: float) -> Union[jnp.array, np.array]:
    r"""
    Build a 2x2 boost matrix in the x-direction
    Args:
        xi (float): rapidity of the boost

    Returns:
        Union[jnp.array, np.array]: the 2x2 boost matrix with shape (...,2,2)
    """
    sigma_x = cb.array([[0, 1], [1, 0]])
    eye = cb.eye(2)
    return cb.cosh(xi / 2) * eye + cb.sinh(xi / 2) * sigma_x


@partial(cb.vectorize, signature="()->(2,2)")
def boost_matrix_2_2_y(xi: float) -> Union[jnp.array, np.array]:
    r"""
    Build a 2x2 boost matrix in the y-direction
    Args:
        xi (float): rapidity of the boost

    Returns:
        Union[jnp.array, np.array]: the 2x2 boost matrix with shape (...,2,2)
    """
    sigma_y = cb.array([[0, -1j], [1j, 0]])
    eye = cb.eye(2)
    return cb.cosh(xi / 2) * eye + cb.sinh(xi / 2) * sigma_y


@partial(cb.vectorize, signature="()->(2,2)")
def boost_matrix_2_2_z(xi: float) -> Union[jnp.array, np.array]:
    r"""
    Build a 2x2 boost matrix in the z-direction
    Args:
        xi (float): rapidity of the boost

    Returns:
        Union[jnp.array, np.array]: the 2x2 boost matrix with shape (...,2,2)
    """
    sigma_z = cb.array([[1, 0], [0, -1]])
    eye = cb.eye(2)
    return cb.cosh(xi / 2) * eye + cb.sinh(xi / 2) * sigma_z


def rotate_to_z_axis(v: Union[jnp.array, np.array]) -> Union[jnp.array, np.array]:
    """Given a vector, calculate the angles to rotate it to the z-axis
    This is done by rotating into the x-z plane (rotation around z axis by -psi_rf) and then into the z-axis (rotation around y axis by -theta_rf)

    Args:
        v (Union[jnp.array, np.array]): the 4 vector to be rotated

    Returns:
        Union[jnp.array, np.array]: the rotation angles around first z and then y axis
    """
    v = cb.array(v)
    psi_rf = cb.arctan2(y_component(v), x_component(v))
    theta_rf = cb.arccos(z_component(v) / p(v))
    return -psi_rf, -theta_rf


@partial(cb.vectorize, signature="()->(2,2)")
def rotation_matrix_2_2_x(theta: float) -> Union[jnp.array, np.array]:
    """Build a 2x2 rotation matrix around the x-axis

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jnp.array, np.array]: the rotation matrix with shape (...,2,2)
    """
    eye = cb.eye(2)
    sgma_x = cb.array([[0, 1], [1, 0]])
    return cb.cos(theta / 2) * eye - 1j * cb.sin(theta / 2) * sgma_x


@partial(cb.vectorize, signature="()->(2,2)")
def rotation_matrix_2_2_y(theta: float) -> Union[jnp.array, np.array]:
    """Build a 2x2 rotation matrix around the y-axis

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jnp.array, np.array]: the rotation matrix with shape (...,2,2)
    """
    eye = cb.eye(2)
    sgma_y = cb.array([[0, -1j], [1j, 0]])
    return cb.cos(theta / 2) * eye - 1j * cb.sin(theta / 2) * sgma_y


@partial(cb.vectorize, signature="()->(2,2)")
def rotation_matrix_2_2_z(theta: float) -> Union[jnp.array, np.array]:
    """Build a 2x2 rotation matrix around the z-axis

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jnp.array, np.array]: the rotation matrix with shape (...,2,2)
    """
    eye = cb.eye(2)
    sgma_z = cb.array([[1, 0], [0, -1]])
    return cb.cos(theta / 2) * eye - 1j * cb.sin(theta / 2) * sgma_z


@partial(cb.vectorize, signature="()->(4,4)")
def boost_matrix_4_4_z(xi: float) -> Union[jnp.array, np.array]:
    r"""Build a 4x4 boost matrix in the z-direction

    Args:
        xi (float): rapidity of the boost

    Returns:
        Union[jnp.array, np.array]: the 4x4 boost matrix with shape (...,4,4)
    """
    gma = cb.cosh(xi)
    beta_gamma = cb.sinh(xi)
    return cb.array(
        [
            [
                1,
                0,
                0,
                0,
            ],
            [
                0,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                gma,
                beta_gamma,
            ],
            [
                0,
                0,
                beta_gamma,
                gma,
            ],
        ]
    )


@partial(cb.vectorize, signature="()->(4,4)")
def rotation_matrix_4_4_y(theta: float) -> Union[jnp.array, np.array]:
    """Build a 4x4 rotation matrix around the y-axis

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jnp.array, np.array]: the rotation matrix with shape (...,4,4)
    """
    return cb.array(
        [
            [
                cb.cos(theta),
                0,
                cb.sin(theta),
                0,
            ],
            [
                0,
                1,
                0,
                0,
            ],
            [
                -cb.sin(theta),
                0,
                cb.cos(theta),
                0,
            ],
            [0, 0, 0, 1],
        ]
    )


@partial(cb.vectorize, signature="()->(4,4)")
def rotation_matrix_4_4_z(theta: float) -> Union[jnp.array, np.array]:
    """Build a 4x4 rotation matrix around the z-axis^

    Args:
        theta (float): the rotation angle

    Returns:
        Union[jax.numpy.ndarray, np.array]: the rotation matrix with shape (...,4,4)
    """
    return cb.array(
        [
            [
                cb.cos(theta),
                -cb.sin(theta),
                0,
                0,
            ],
            [
                cb.sin(theta),
                cb.cos(theta),
                0,
                0,
            ],
            [
                0,
                0,
                1,
                0,
            ],
            [0, 0, 0, 1],
        ]
    )


@partial(cb.vectorize, signature="(), (), (), (), (), ()->(2,2)")
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


@partial(cb.vectorize, signature="(), (), (), (), (), ()->(4,4)")
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


def decode_rotation_4x4(rotation_matrix: jnp.array) -> Tuple[float, float, float]:
    r"""decode a 4x4 rotation matrix into the 3 rotation angles

    Args:
        matrix (_type_): _description_
    """
    phi = cb.arctan2(rotation_matrix[..., 1, 2], rotation_matrix[..., 0, 2])
    theta = save_arccos(rotation_matrix[..., 2, 2])
    psi = cb.arctan2(rotation_matrix[..., 2, 1], -rotation_matrix[..., 2, 0])
    return phi, theta, psi


def decode_4_4(matrix, tol=1e-14):
    r"""decode a 4x4 matrix into the 6 kinematic parameters

    Args:
        matrix (jax.numpy.ndarray): the 4x4 matrix
    """
    m = 1.0
    v_0 = cb.array([0, 0, 0, m])

    v = matrix @ v_0
    w = time_component(v)
    abs_mom = p(v)
    gma = w / m

    # gamma can be smaller than 1 due to numerical errors
    # for large deviations we will raise an exception
    gma = cb.where((abs(gma) < 1) & (abs(gma - 1) < cfg.gamma_tolerance), 1, gma)
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
        cb.all(cb.isclose(m_original_2x2, new_2x2), axis=-1), axis=-1
    )

    two_pi_shifted = cb.all(
        cb.all(cb.isclose(m_original_2x2, -new_2x2), axis=-1), axis=-1
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

    psi_rf = cb.where(two_pi_shifted, psi_rf + 2 * cb.pi, psi_rf)
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


def norm(vec):
    """
    Calculate norm of 3-vector

    :param vec: Input 3-vector
    :returns: Scalar norm

    """
    return cb.sqrt(cb.sum(vec * vec, -1))


def p(vector):
    """
    Calculate absolute value of the 4-momentum

    :param vector: Input 4-momentum vector
    :returns: Absolute momentum (scalar)

    """
    return norm(spatial_components(vector))


def scalar_product(vec1, vec2):
    """
    Calculate scalar product of two 3-vectors

    :param vec1: First 3-vector
    :param vec2: Secont 3-vector
    :returns: Scalar product

    """
    return cb.sum(vec1 * vec2, -1)


def scalar(x):
    """
    Create a scalar (array with only one component in last index) which can be used
    to e.g. scale a vector.

    :param x: Initial value
    :returns: Scalar value

    """
    return cb.stack([x], axis=-1)


def lorentz_boost(vector, boostvector):
    """
    Perform Lorentz boost of the 4-vector vector using boost vector boostvector.
    We do not use the matrices here, to make things a little easier

    :param vector: 4-vector to be boosted
    :param boostvector: boost vector.
                        Can be either 3-vector or 4-vector
                        (only spatial components are used)
    :returns: Boosted 4-vector

    """
    boost = spatial_components(boostvector)
    b2 = scalar_product(boost, boost)
    gma = 1.0 / cb.sqrt(1.0 - b2)
    gma2 = (gma - 1.0) / b2
    gma2 = cb.where(cb.isclose(b2, 0), 0 , gma2)
    ve = time_component(vector)
    vp = spatial_components(vector)
    bp = scalar_product(vp, boost)
    vp2 = vp + scalar(gma2 * bp + gma * ve) * boost
    ve2 = gma * (ve + bp)
    
    return lorentz_vector(vp2, ve2)


def boost_to_rest(vector, boostvector):
    """
    Perform Lorentz boost to the rest frame of the
    4-vector boostvector.

    :param vector: 4-vector to be boosted
    :param boostvector: Boost momentum 4-vector in the same frame as vector
                        (should have nonzero mass!)
    :returns: 4-vector boosed to boostvector rest frame

    """
    boost = -spatial_components(boostvector) / scalar(time_component(boostvector))
    return lorentz_boost(vector, boost)
