import numpy as np

# NOTE: tau = inf means do not truncate


def EXP(x, lam):
    if x >= 0:
        return 1 - np.exp(-lam * x)
    return 0


def GLL(x, k):
    if x >= 0:
        return 1 - (1 + x)**(-k)
    return 0


def EXP_T(x, lam, tau):
    if tau == np.inf:
        return EXP(x, lam)
    if 0 <= x <= tau:
        if lam == 0:
            return x / tau
        return (1 - np.exp(-lam * x)) / (1 - np.exp(-lam * tau))
    return 0


def GLL_T(x, k, tau):
    if tau == np.inf:
        return GLL(x, k)
    if 0 <= x <= tau:
        return (1 - (1 + x)**(-k)) / (1 - (1 + tau)**(-k))
    return 0
