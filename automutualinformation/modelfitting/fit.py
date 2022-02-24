import lmfit
from automutualinformation.modelfitting.decay_functions import (
    powerlaw_decay,
    exponential_decay,
    pow_exp_decay,
)
import numpy as np


def fit_model(
    distances,
    sig,
    decay_function="powerlaw",
    decay_params=None,
    n_iter=1,
    methods=["nelder", "leastsq", "least-squares"],
    fit_scale="log",
    model_selection_criteria="aic",
):
    """_summary_

    Args:
        distances (list / np.array): Distances / time-lag to compute MI
        sig (list / np.array): Signal
        decay_function (str or function, optional): Decay function to fit. Available functions are "powerlaw", "exponential" or "pow_exp". Alternatively, you can pass your own decay function in. Defaults to "powerlaw".
        decay_params (list, optional): # A sequence of tuples, or a sequence
             of Parameter instances. If it is a sequence of tuples, then each tuple
             must contain at least a name. The order in each tuple must be (name,
             value, vary, min, max, expr, brute_step).. Defaults to ().        n_iter (int, optional): _description_. Defaults to 1.
        method (list, optional): Optimization method to fit model. Best fit model will be chosen on the basis of AIC. Each fitting method will be tried n_iter times. Defaults to ["nelder", "leastsq", "least-squares"].
        fit_scale (str, optional): Fitting on either log scaled residuals, or linearly. Options: 'log', 'linear'. Defaults to "log".
        model_selection_criteria (str, optional): _description_. Defaults to 'aic'.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # initialize decay function
    if decay_function == "powerlaw":
        decay_params = [
            ("p_init", 0.5, True, 1e-10),
            ("p_decay_const", -0.5, True, -np.inf, -1e-10),
            ("intercept", 1e-5, True, 1e-10),
        ]
        decay_function = powerlaw_decay

    elif decay_function == "exponential":
        decay_params = [
            ("e_init", 0.5, True, 1e-10),
            ("e_decay_const", 0.1, True, 1e-10),
            ("intercept", 1e-5, True, 1e-10),
        ]
        decay_function = exponential_decay

    elif decay_function == "pow_exp":
        decay_params = [
            ("e_init", 0.5, True, 1e-19),
            ("e_decay_const", 0.1, True, 1e-10),
            ("p_init", 0.5, True, 1e-10),
            ("p_decay_const", -0.5, True, -np.inf, -1e-10),
            ("intercept", 1e-5, True, 1e-10),
        ]
        decay_function = pow_exp_decay

    else:
        if not callable(decay_function):
            # if this is a function
            print(f"decay_function {decay_function} not supported")

    # initialize parameters
    model_paremeters = lmfit.Parameters()
    model_paremeters.add_many(*decay_params)

    # initialize minimizer
    minimizer = lmfit.Minimizer(
        model_res,
        model_paremeters,
        fcn_args=(distances, sig, fit_scale, decay_function),
        nan_policy="omit",
    )

    # fit models
    results = [
        fit_model_iter(minimizer, n_iter=n_iter, **{"method": method})
        for method in methods
    ]

    # select best fit model
    if model_selection_criteria == "aic":
        best_fit_model_results = results[np.argmin([i.aic for i in results])]
    elif model_selection_criteria == "bic":
        best_fit_model_results = results[np.argmin([i.bic for i in results])]
    else:
        raise ValueError("model_selection_criteria must be aic or bic")

    # get model fit
    model_y = get_model_y(decay_function, best_fit_model_results, distances)

    return best_fit_model_results, model_y


def fit_model_iter(model, n_iter=10, **kwargs):
    """re-fit model n_iter times and choose the best fit
    chooses method based upon best-fit
    """
    models = []
    AICs = []
    for iter in np.arange(n_iter):
        results_model = model.minimize(**kwargs)
        models.append(results_model)
        AICs.append(results_model.aic)
    return models[np.argmin(AICs)]


def residuals(y_true, y_model, x, logscaled=False):
    if logscaled:
        return np.abs(np.log(y_true) - np.log(y_model)) * (1 / (np.log(1 + x)))
    else:
        return np.abs(y_true - y_model)


def model_res(p, x, y, fit, model):
    if fit == "linear":
        return residuals(y, model(p, x), x)
    elif fit == "log":
        return residuals(y, model(p, x), x, logscaled=True)
    else:
        raise ValueError(
            "Only 'log' or 'linear' fits supported for parameter `fit_scale`"
        )


def get_model_y(model, results, x):
    return model({i: results.params[i].value for i in results.params}, x)
