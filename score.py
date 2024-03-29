import numpy as np

import distributions
import tf_aspect
import estimate_truncpt.py
import estimate_parameter.py


def score_per_term(t, d, alpha):
    return idf(t) * alpha


def score(Q, d, alpha):
    return sum(score_per_term(t, d, alpha) for t in Q)
