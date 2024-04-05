from decayangle.config import config as cfg

cb = cfg.backend


def save_arccos(x):
    """Save version of arccos by clipping the argument to the range [-1, 1]

    Args:
        x (Union[float, jnp.array]): The argument of the arccos

    Returns:
        Union[float, jnp.array]: The result of the arccos
    """
    x = cb.clip(x, -1, 1)
    return cb.arccos(x)


def matrix_vector_product(matrix, vector):
    """Vectorized version of the matrix vector product using einsum

    Args:
        matrix (Union[np.array, jnp.array]): The matrix
        vector (Union[np.array, jnp.array]): The vector

    Returns:
        Union[np.array, jnp.array]: The result of the matrix vector product
    """
    return cb.einsum("...ij, ...j", matrix, vector)
