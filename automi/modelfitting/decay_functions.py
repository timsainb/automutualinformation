import numpy as np

# decay types
def powerlaw_decay(p, x):
    return p["p_init"] * x ** (p["p_decay_const"]) + p["intercept"]


def exponential_decay(p, x):
    return p["e_init"] * np.exp(-x * p["e_decay_const"]) + p["intercept"]


def pow_exp_decay(p, x):
    powerlaw = p["p_init"] * x ** (p["p_decay_const"])
    exponential = p["e_init"] * np.exp(-x * p["e_decay_const"])
    return powerlaw + exponential + p["intercept"]
