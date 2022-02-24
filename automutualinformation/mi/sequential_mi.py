import numpy as np
from tqdm.auto import tqdm
import copy
from joblib import Parallel, delayed

# mutual information estimation functions
from automutualinformation.mi.grassberger import est_mutual_info_p
from sklearn.metrics import mutual_info_score

######## Mutual Information From distributions ############
def MI_from_distributions(
    sequences,
    dist,
    unclustered_element=None,
    shuffle=False,
    mi_estimation="grassberger",
    **mi_kwargs
):
    """_summary_
    TODO
    Args:
        sequences (_type_): _description_
        dist (_type_): _description_
        unclustered_element (_type_, optional): _description_. Defaults to None.
        shuffle (bool, optional): _description_. Defaults to False.
        mi_estimation (str, optional): _description_. Defaults to "grassberger".

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    np.random.seed()  # set seed
    # create distributions
    if np.sum([len(seq) > dist for seq in sequences]) == 0:
        return (np.nan, np.nan)

    if shuffle:
        distribution_a = np.concatenate(
            [np.random.permutation(seq[dist:]) for seq in sequences if len(seq) > dist]
        )

        distribution_b = np.concatenate(
            [np.random.permutation(seq[:-dist]) for seq in sequences if len(seq) > dist]
        )

    else:
        distribution_a = np.concatenate(
            [seq[dist:] for seq in sequences if len(seq) > dist]
        )

        distribution_b = np.concatenate(
            [seq[:-dist] for seq in sequences if len(seq) > dist]
        )

    # mask unclustered so they are not considered in MI
    if unclustered_element is not None:
        mask = (distribution_a == unclustered_element) | (
            distribution_b == unclustered_element
        )
        distribution_a = distribution_a[mask == False]
        distribution_b = distribution_b[mask == False]

    # calculate MI

    if mi_estimation == "grassberger":
        # See Grassberger, P. Entropy estimates from insufficient samplings. arXiv 2003, arXiv:0307138
        return est_mutual_info_p(distribution_a, distribution_b)
    elif mi_estimation == "naive":
        # sklearns mi implementation
        return (mutual_info_score(distribution_a, distribution_b, **mi_kwargs), np.nan)
    else:
        raise ValueError("MI estimator '{}' is not implemented".format(mi_estimation))


def sequential_mutual_information(
    sequences,
    distances=np.arange(1, 100),
    n_jobs=-1,
    n_shuff_repeats=1,
    use_tqdm=True,
    mi_estimation="grassberger",
    parallel_kwargs={
        "verbose": 5,
        "prefer": None,
    },
    unclustered_element=None,
    **mi_kwargs
):
    """Compute the auto mutual information as a function of distance.

    Args:
        sequences (list or list of lists): List or list of lists
        distances (np.array): Distances over which to compute MI.
        n_jobs (int, optional): Number of jobs used to compute MI. Each
            jobs will be run in parallel over distances. Defaults to 1.
        n_shuff_repeats (int, optional): how many times to shuffle distribution in
            order to estimate lower bound. Defaults to 1.
        parallel_kwargs (dict, optional): _description_. Defaults to None.
        unclustered_element (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # convert to numeric for faster computation
    unique_elements = np.unique(np.concatenate(sequences))
    n_unique = len(unique_elements)
    seq_dict = {j: i for i, j in enumerate(unique_elements)}
    if n_unique < 256:
        sequences = [
            np.array([seq_dict[i] for i in seq]).astype("uint8") for seq in sequences
        ]
    elif n_unique < 65535:
        sequences = [
            np.array([seq_dict[i] for i in seq]).astype("uint16") for seq in sequences
        ]
    else:
        sequences = [
            np.array([seq_dict[i] for i in seq]).astype("uint32") for seq in sequences
        ]

    # adjust dataset so that unclustered elements are not factored into MI elements
    if unclustered_element is not None:
        unclustered_element = seq_dict[unclustered_element]
        print(unclustered_element)
    else:
        unclustered_element = None

    # because parallelization occurs within the function
    if mi_estimation == "adjusted_mi":
        _n_jobs = copy.deepcopy(n_jobs)
        n_jobs = 1
    else:
        _n_jobs = 1
    # compute MI
    if n_jobs == 1:
        MI = [
            MI_from_distributions(
                sequences,
                dist,
                unclustered_element=unclustered_element,
                mi_estimation=mi_estimation,
                **mi_kwargs
            )
            for dist_i, dist in enumerate(
                tqdm(distances, leave=False, disable=use_tqdm == False)
            )
        ]
        distances_rep = np.repeat(distances, n_shuff_repeats)
        shuff_MI = [
            MI_from_distributions(
                sequences,
                dist,
                unclustered_element=unclustered_element,
                mi_estimation=mi_estimation,
                shuffle=True,
                **mi_kwargs
            )
            for dist_i, dist in enumerate(
                tqdm(distances_rep, leave=False, disable=use_tqdm == False)
            )
        ]

        shuff_MI = np.reshape(shuff_MI, (len(distances), n_shuff_repeats, 2))
        shuff_MI = np.mean(shuff_MI, axis=1)

    else:
        with Parallel(n_jobs=n_jobs, **parallel_kwargs) as parallel:
            MI = parallel(
                delayed(MI_from_distributions)(
                    sequences,
                    dist,
                    unclustered_element=unclustered_element,
                    mi_estimation=mi_estimation,
                    **mi_kwargs
                )
                for dist_i, dist in enumerate(
                    tqdm(distances, leave=False, disable=use_tqdm == False)
                )
            )

        with Parallel(n_jobs=n_jobs, **parallel_kwargs) as parallel:
            distances_rep = np.repeat(distances, n_shuff_repeats)
            shuff_MI = parallel(
                delayed(MI_from_distributions)(
                    sequences,
                    dist,
                    unclustered_element=unclustered_element,
                    mi_estimation=mi_estimation,
                    shuffle=True,
                    **mi_kwargs
                )
                for dist_i, dist in enumerate(
                    tqdm(distances_rep, leave=False, disable=use_tqdm == False)
                )
            )
            shuff_MI = np.reshape(shuff_MI, (len(distances), n_shuff_repeats, 2))
            shuff_MI = np.mean(shuff_MI, axis=1)

    return np.array(MI).T, np.array(shuff_MI).T
