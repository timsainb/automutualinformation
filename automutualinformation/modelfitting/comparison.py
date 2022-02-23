from scipy.stats.distributions import chi2
from automutualinformation.modelfitting.fit import residuals
import numpy as np


def likelihood_ratio(llmin, llmax):
    return 2 * (llmax - llmin)


def log_likelihood(N_data, y_true, y_model, x, logscaled=False):
    return -(N_data / 2) * np.log(RSS(y_true, y_model, x, logscaled) / N_data)


def LRT(
    true_y,
    model1_y,
    model2_y,
    model1_nparams,
    model2_nparams,
    distances,
    logscaled=False,
):
    """performs a likelihood ratio test for two lmfit models"""
    # n samples
    n = len(distances)

    # get likelihood
    LL1 = log_likelihood(n, true_y, model1_y, distances, logscaled=logscaled)
    LL2 = log_likelihood(n, true_y, model2_y, distances, logscaled=logscaled)

    # perform likelihood ratio test
    LR = likelihood_ratio(LL1, LL2)

    # get probability
    p = chi2.sf(LR, model2_nparams - model1_nparams)
    return LL1, LL2, p


def RSS(y_true, y_model, x, logscaled=False):
    return np.sum(residuals(y_true, y_model, x, logscaled=logscaled) ** 2)


def r2(y_true, y_model, x, logscaled=False):
    ss_res = RSS(y_true, y_model, x, logscaled=logscaled)
    ss_tot = RSS(y_true, np.mean(y_true), x, logscaled=logscaled)
    return 1 - ss_res / ss_tot


def AICc(N_data, N_params, y_true, y_model, x, logscaled=False):
    return AIC(N_data, N_params, y_true, y_model, x, logscaled=logscaled) + (
        2 * N_params * (N_params + 1)
    ) / (N_data - N_params - 1)


def AIC(N_data, N_params, y_true, y_model, x, logscaled=False):
    return (
        N_data * np.log(RSS(y_true, y_model, x, logscaled=logscaled) / N_data)
        + 2 * N_params
    )


def compute_relative_likelihoods(delta_AIC):
    return np.exp(-0.5 * delta_AIC)


def compute_relative_probabilities(model_relative_likelihoods):
    """probability of the model given data and the other models"""
    return model_relative_likelihoods / np.sum(model_relative_likelihoods)
