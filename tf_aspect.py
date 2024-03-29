import numpy as np


# Relative Intra-document TF (RITF)
# @param tf: TF(t, D) -> term frequency of t in D
# @param mtc: Avg.TF(D) -> average term frequency of D
# @param c: free parameter (default is 1)
def ritf(tf, mtc, c=1):
    return np.log2(1 + tf) / np.log2(c + mtc)


# Length regularized TF (LRTF)
# @param tf: TF(t, D) -> term frequency of t in D
# @param adl: ADL(C) -> average document length in collection C
# @param dl: len(D) -> length of document D
def lrtf(tf, adl, dl):
    return tf * np.log2(1 + adl / dl)
