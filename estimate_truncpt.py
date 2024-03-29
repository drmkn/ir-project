import numpy as np
from scipy.stats import bootstrap


def truncpt_EXP_T(sample, delta):
    # Check if truncpt_mle_estimate is a good estimate using parameter bootstrap.
    # If not, then return inf i.e. do not truncate.
    def truncpt_statistic(data):
        lam = len(data) / np.sum(data)
        return lam * np.log(delta)

    confidence_interval = bootstrap((sample,), truncpt_statistic).confidence_interval
    truncpt_mle_estimate = truncpt_statistic(sample)

    if confidence_interval.low <= truncpt_mle_estimate <= confidence_interval.high:
        return truncpt_mle_estimate

    return np.inf


def truncpt_GLL_T(sample, delta):
    # Check if truncpt_mle_estimate is a good estimate using parameter bootstrap.
    # If not, then return inf i.e. do not truncate.
    def truncpt_statistic(data):
        k = len(data) / np.sum(np.log(1 + data))
        return delta**(-1 / k) - 1

    confidence_interval = bootstrap((sample,), truncpt_statistic).confidence_interval
    truncpt_mle_estimate = truncpt_statistic(sample)

    if confidence_interval.low <= truncpt_mle_estimate <= confidence_interval.high:
        return truncpt_mle_estimate

    return np.inf
