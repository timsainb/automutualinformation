from matplotlib import mlab
import matplotlib.pyplot as plt
import numpy as np
import colorednoise as cn
from automutualinformation import sequential_mutual_information as smi
from automutualinformation import fit_model

beta = 0.5  # the exponent
samples = 10000  # number of samples to generate
y = cn.powerlaw_psd_gaussian(beta, samples)
nbins = 10  # how many bins to compute over
bins = np.linspace(np.min(y), np.max(y), nbins)
y_dig = np.digitize(y, bins, right=True)
range_ = np.arange(1, 10)


def test_compute_mi():
    (MI, _), (shuff_MI, _) = smi([y_dig], distances=range_, n_jobs=1)


def test_compute_mi_fit_model():
    (MI, _), (shuff_MI, _) = smi([y_dig], distances=range_, n_jobs=1)

    decay_model, model_y = fit_model(
        distances=range_,
        sig=MI - shuff_MI,
    )
