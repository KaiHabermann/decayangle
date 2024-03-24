from decayangle.config import config as cfg
cb = cfg.backend


def save_arccos(x):
    # clip the argument to the range [-1, 1]
    # this may be needed in cases where the argument is slightly outside the range
    x = cb.clip(x, -1, 1)
    return cb.arccos(x)