import numpy as np

def dist_(a, b):#distance entre deux flottants...
    return abs(a-b)

def dist_moy(A,B):
    n_a = len(A)
    n_b = len(B)
    assert n_a == n_b
    dist_v = np.vectorize(dist_)
    return (np.sum(dist_v(A,B))/n_a)