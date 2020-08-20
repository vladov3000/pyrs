import scipy.special
from scipy.stats import norm
import numpy as np

EPSILON = 1e-10
factorial = lambda x: scipy.special.factorial(x.astype(int) if isinstance(x, np.ndarray) else int(x))

def phi(a):
    """ CDF of Gaussian """
    return norm.cdf(a)

def phi_inv(a):
    """ Inverse CDF of Gaussian """
    return norm.ppf(a)
