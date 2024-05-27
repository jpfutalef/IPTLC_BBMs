import numpy as np


def ks_statistic(ecdf_1, ecdf_2):
    """
    Compute the Kolmogorov-Smirnov statistic between two empirical cumulative distribution functions.

    Parameters
    ----------
    ecdf_1 : np.ndarray
        The first empirical cumulative distribution function.

    ecdf_2 : np.ndarray
        The second empirical cumulative distribution function.

    Returns
    -------
    float
        The Kolmogorov-Smirnov statistic.
    """
    abs_dff = np.abs(ecdf_1 - ecdf_2)

    # get max and the value where it happens
    max_diff = np.max(abs_dff)
    max_diff_idx = np.argmax(abs_dff)
    val = ecdf_1[max_diff_idx]

    return max_diff, val


def empirical_cdf(data, bins):
    """
    Compute the empirical cumulative distribution function of a dataset.

    Parameters
    ----------
    data : np.ndarray
        The data to compute the ECDF.

    bins : np.ndarray
        The bins to compute the ECDF.

    Returns
    -------
    np.ndarray
        The empirical cumulative distribution function.
    """
    # Compute the empirical pdf
    epdf, _ = np.histogram(data, bins=bins, density=True)

    # Empirical CDF
    ecdf = np.cumsum(epdf) * np.diff(bins)

    # Add zero and one to teh array to ensure we obtain an ECDF
    ecdf = np.concatenate(([0], ecdf, [1]))

    return ecdf


def comparison_bins(data1, data2, n_bins=50):
    """
    Compute the bins for the comparison of two datasets.

    Parameters
    ----------
    data1 : np.ndarray
        The first dataset.

    data2 : np.ndarray
        The second dataset.

    n_bins : int
        The number of bins.

    Returns
    -------
    np.ndarray
        The bins.
    """
    # Compute the bins
    bins = np.linspace(min(np.min(data1), np.min(data2)), max(np.max(data1), np.max(data2)), n_bins + 1)
    return bins


def lack_of_fit(data1, data2, n_bins=50, return_info=False):
    """
    Compute the lack of fit between two datasets.

    Parameters
    ----------
    data1 : np.ndarray
        The first dataset.

    data2 : np.ndarray
        The second dataset.

    n_bins : int
        The number of bins.

    Returns
    -------
    float
        The Kolmogorov-Smirnov statistic.
    """
    # Compute the bins
    bins = comparison_bins(data1, data2, n_bins)

    # Compute the empirical CDF
    ecdf_1 = empirical_cdf(data1, bins)
    ecdf_2 = empirical_cdf(data2, bins)

    # Compute the KS statistic
    lof, val = ks_statistic(ecdf_1, ecdf_2)

    if return_info:
        # Info dict
        info = {"bins": bins,
                "ecdf_1": ecdf_1,
                "ecdf_2": ecdf_2,
                "ks_statistic": lof,
                "ks_statistic_value": val
                }

        return lof, info

    return lof
