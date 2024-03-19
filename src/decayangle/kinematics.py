from jax import numpy as jnp
import numpy as np
from jax import jit
from decayangle.config import config
from typing import Tuple, Union


def boost_matrix_2_2_x(xi):
    r""" 
    Build a 2x2 boost matrix in the x-direction
    Args:
        xi (float): rapidity of the boost
    """
    sigma_x = config.backend.array([[0, 1],
                        [1, 0]])
    I = config.backend.array([[1, 0],
                [0, 1]])
    return config.backend.cosh(xi/2)*I + config.backend.sinh(xi/2)*sigma_x

def boost_matrix_2_2_y(xi):
    r"""
    Build a 2x2 boost matrix in the y-direction
    Args:
        xi (float): rapidity of the boost
    """
    sigma_y = config.backend.array([[0, -1j],
                        [1j, 0]])
    I = config.backend.array([[1, 0],
                [0, 1]])
    return config.backend.cosh(xi/2)*I  + config.backend.sinh(xi/2)*sigma_y

def boost_matrix_2_2_z(xi):
    r"""
    Build a 2x2 boost matrix in the z-direction
    Args:
        xi (float): rapidity of the boost
    """
    sigma_z = config.backend.array([[1, 0],
                        [0, -1]])
    I = config.backend.array([[1, 0],
                [0, 1]])
    return (config.backend.cosh(xi/2)*I + config.backend.sinh(xi/2) * sigma_z)

def rotate_to_z_axis(v):
    """Given a vector, rotate it to the z-axis

    Args:
        v (jax.numpy.ndarray): the 4 vector to be rotated
    
    Returns:
        jax.numpy.ndarray: the rotation angles around first z and then y axis
    """ 
    v = config.backend.array(v)
    psi_rf = -config.backend.arctan2(y_component(v), x_component(v))
    theta_rf = config.backend.arccos(z_component(v) / p(v))
    return psi_rf, -theta_rf

def rotation_matrix_2_2_x(theta):
    """Build a 2x2 rotation matrix around the x-axis

    Args:
        theta (jax.numpy.ndarray): the rotation angle

    Returns:
        jax.numpy.ndarray: the rotation matrix with shape (...,2,2)
    """
    I = config.backend.array([[1, 0],
                   [0, 1]])
    sgma_x = config.backend.array([[0, 1],
                        [1, 0]])
    return (config.backend.cos(theta/2) * I - 1j*config.backend.sin(theta/2)*sgma_x)

def rotation_matrix_2_2_y(theta):
    """Build a 2x2 rotation matrix around the y-axis

    Args:
        theta (jax.numpy.ndarray): the rotation angle

    Returns:
        jax.numpy.ndarray: the rotation matrix with shape (...,2,2)
    """
    I = config.backend.array([[1, 0],
                     [0, 1]])
    sgma_y = config.backend.array([[0, -1j],
                        [1j, 0]])
    return (config.backend.cos(theta/2)*I - 1j*config.backend.sin(theta/2)*sgma_y)

def rotation_matrix_2_2_z(theta):
    """Build a 2x2 rotation matrix around the z-axis

    Args:
        theta (jax.numpy.ndarray): the rotation angle

    Returns:
        jax.numpy.ndarray: the rotation matrix with shape (...,2,2)
    """
    I = config.backend.array([[1, 0], 
                   [0, 1]])
    sgma_z = config.backend.array([[1,  0], 
                        [0, -1]])
    return (config.backend.cos(theta/2)*I - 1j*config.backend.sin(theta/2)*sgma_z)

def boost_matrix_4_4_z(xi):
    r"""Build a 4x4 boost matrix in the z-direction

    Args:
        xi (float): rapidity of the boost
    
    Returns:
        jax.numpy.ndarray: the 4x4 boost matrix with shape (...,4,4)
    """
    gamma = config.backend.cosh(xi)
    beta_gamma = config.backend.sinh(xi)
    return config.backend.array([
        [1, 0, 0, 0,],
        [0, 1, 0, 0,],
        [0, 0, gamma, beta_gamma,],
        [0, 0, beta_gamma, gamma,]
    ])

def rotation_matrix_4_4_y(theta):
    """Build a 4x4 rotation matrix around the y-axis

    Args:
        theta (jax.numpy.ndarray): the rotation angle

    Returns:
        jax.numpy.ndarray: the rotation matrix with shape (...,4,4)
    """
    return config.backend.array([
        [config.backend.cos(theta), 0, config.backend.sin(theta), 0,],
        [0, 1, 0, 0,],
        [-config.backend.sin(theta), 0, config.backend.cos(theta), 0,],
        [0, 0, 0, 1]
    ])

def rotation_matrix_4_4_z(theta):
    return config.backend.array([
        [config.backend.cos(theta), -config.backend.sin(theta), 0, 0,],
        [config.backend.sin(theta), config.backend.cos(theta), 0, 0,],
        [0, 0, 1, 0,],
        [0, 0, 0, 1]
    ])

def build_2_2(psi, theta, xi, theta_rf, phi_rf, psi_rf):
    r"""Build a 2x2 matrix from the 6 kinematic parameters

    Args:
        psi (float): the rotation angle around the z-axis
        theta (float): the rotation angle around the y-axis
        xi (float): the rapidity of the boost
        theta_rf (float): the rotation angle around the y-axis in the rest frame
        phi_rf (float): the rotation angle around the z-axis in the rest frame
        psi_rf (float): the rotation angle around the z-axis in the rest frame

    Returns:
        jax.numpy.ndarray: the 2x2 matrix
    """
    return (rotation_matrix_2_2_z(psi) @ rotation_matrix_2_2_y(theta) @ boost_matrix_2_2_z(xi) @ rotation_matrix_2_2_z(phi_rf) @ rotation_matrix_2_2_y(theta_rf) @ rotation_matrix_2_2_z(psi_rf))

def build_4_4(psi, theta, xi, theta_rf, phi_rf, psi_rf):
    r"""Build a 4x4 matrix from the 6 kinematic parameters

    Args:
        psi (float): the rotation angle around the z-axis
        theta (float): the rotation angle around the y-axis
        xi (float): the rapidity of the boost
        theta_rf (float): the rotation angle around the y-axis in the rest frame
        phi_rf (float): the rotation angle around the z-axis in the rest frame
        psi_rf (float): the rotation angle around the z-axis in the rest frame

    Returns:
        jax.numpy.ndarray: the 4x4 matrix
    """
    return (rotation_matrix_4_4_z(psi) @ rotation_matrix_4_4_y(theta) @ boost_matrix_4_4_z(xi) @ rotation_matrix_4_4_z(phi_rf) @ rotation_matrix_4_4_y(theta_rf) @ rotation_matrix_4_4_z(psi_rf))


def decode_rotation_4x4(R):
    r"""decode a 4x4 rotation matrix into the 3 rotation angles

    Args:
        matrix (_type_): _description_
    """
    phi = config.backend.arctan2(R[1,2], R[0,2])
    theta = config.backend.arccos(R[2,2])
    psi = config.backend.arctan2(R[2,1], -R[2,0])
    return phi, theta, psi

def decode_4_4(matrix):
    r"""decode a 4x4 matrix into the 6 kinematic parameters

    Args:
        matrix (jax.numpy.ndarray): the 4x4 matrix
    """

    m = 1.0
    V0 = config.backend.array([0, 0, 0, m])

    V = matrix @ V0
    w = time_component(V)
    abs_mom = p(V)
    gamma = w / m
    if gamma < 1:
        # gamma can be smaller than 1 due to numerical errors
        # for large deviations we will raise an exception
        if abs(gamma - 1) < 1e-10:
            gamma = 1.0
        else:
            raise ValueError(f"gamma is {gamma}, which is less than 1. This is not a valid Lorentz transformation")
    xi = config.backend.arccosh(gamma)

    psi = config.backend.arctan2(y_component(V), x_component(V))

    cosine_input = config.backend.where(abs(abs_mom) <= 1e-19, 0, z_component(V) / abs_mom)
    theta = config.backend.arccos(cosine_input)

    M_rf = boost_matrix_4_4_z(-xi) @ rotation_matrix_4_4_y(-theta) @ rotation_matrix_4_4_z(-psi) @ matrix
    phi_rf, theta_rf, psi_rf = decode_rotation_4x4(M_rf[:3, :3])
    return psi, theta, xi, theta_rf, phi_rf,  psi_rf

def adjust_for_2pi_rotation(M_original_2x2, psi, theta, xi, theta_rf, phi_rf,  psi_rf) -> Tuple[Union[jnp.array, np.array]]:
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
    new_2x2 = build_2_2(psi, theta, xi, theta_rf, phi_rf,  psi_rf)
    if np.allclose(M_original_2x2, new_2x2):
        return psi, theta, xi, theta_rf, phi_rf,  psi_rf
    elif np.allclose(M_original_2x2, -new_2x2):
        return psi, theta, xi, theta_rf, phi_rf,  psi_rf + 2*np.pi
    else:
        raise ValueError(f"The 2x2 matrix does not match the reconstruced parameters!"
                         f"This can happen due to numerical errors." 
                         f"The original matrix is {M_original_2x2} and the reconstructed matrix is {new_2x2}")


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
    return config.backend.sqrt(x_component(vector) ** 2 + y_component(vector) ** 2)

def eta(vector):
    """Return pseudorapidity component of the input Lorentz or 3-vector

    :param vector: input vector (Lorentz or 3-vector)
    :returns: vector of pseudorapidity components

    """
    return -config.backend.log(pt(vector) / 2.0 / z_component(vector))

def vector(x, y, z):
    """
    Make a 3-vector from components. Components are stacked along the last index.

    :param x: x-component of the vector
    :param y: y-component of the vector
    :param z: z-component of the vector
    :returns: 3-vector
    """
    return config.backend.stack([x, y, z], axis=-1)

def mass_squared(vector):
    """
    Calculate squared invariant mass scalar for Lorentz 4-momentum vector

    :param vector: input Lorentz momentum vector
    :returns: scalar invariant mass squared

    """
    return config.backend.sum(vector * vector * metric_tensor(), -1)

def metric_tensor():
    """
    Constant metric tensor for Lorentz space

    :returns: Metric tensor
    """
    return config.backend.array([-1.0, -1.0, -1.0, 1.0], dtype=config.backend.float64)

def lorentz_vector(space, time):
    """
    Make a Lorentz vector from spatial and time components

    :param space: 3-vector of spatial components
    :param time: time component
    :returns: Lorentz vector

    """
    return config.backend.concatenate([space, config.backend.stack([time], axis=-1)], axis=-1)

def mass(vector):
    """
    Calculate mass scalar for Lorentz 4-momentum vector

    :param vector: input Lorentz momentum vector
    :returns: scalar invariant mass

    """
    return config.backend.sqrt(mass_squared(vector))

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
    return 0.5 * config.backend.log((b + 1) / (1 - b))

def norm(vec):
    """
    Calculate norm of 3-vector

    :param vec: Input 3-vector
    :returns: Scalar norm

    """
    return config.backend.sqrt(config.backend.sum(vec * vec, -1))

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
    return config.backend.sum(vec1 * vec2, -1)

def scalar(x):
    """
    Create a scalar (array with only one component in last index) which can be used
    to e.g. scale a vector. 

    :param x: Initial value
    :returns: Scalar value

    """
    return config.backend.stack([x], axis=-1)

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
    gamma = 1.0 / config.backend.sqrt(1.0 - b2)
    gamma2 = (gamma - 1.0) / b2
    ve = time_component(vector)
    vp = spatial_components(vector)
    bp = scalar_product(vp, boost)
    vp2 = vp + scalar(gamma2 * bp + gamma * ve) * boost
    ve2 = gamma * (ve + bp)
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