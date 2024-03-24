from decayangle.config import config as cfg
cb = cfg.backend


def save_arccos(x):
    x = cb.clip(x, -1, 1)
    return cb.arccos(x)