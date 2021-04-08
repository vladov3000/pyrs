""" Implementation of frequentist method """
from scipy.stats import poisson
import numpy as np

from utils import *

""" Generate expectations (average value) for n and m,
    the main and control measurements respectively. 
    See bottom of page 22 for formulas. """
def get_E_n(mu, s, b):
    return mu * s + b

def get_E_m(b, tau):
    return tau * b

def gen_data(exp, num):
    """ Generate num values along poisson distribution using given expectation. """
    return poisson.ppf(np.random.uniform(size=num), exp)

def gen_ns(mu, s, b, num):
    """ Generate num values of n given parameters. """
    return gen_data(get_E_n(mu, s, b), num)

def gen_ms(b, tau, num):
    """ Generate num values of m given parameters. """
    return gen_data(get_E_m(b, tau), num)

def get_likelihood(n, m, mu, b, s, tau):
    """ Calculate L(mu, b). Since this is only single bin, 
    we just multiply two terms together. See Eqn. 90 on page 23. """
    t1 = (mu * s + b) ** n / factorial(n) * np.exp(-(mu * s + b))
    t2 = (tau * b) ** m / factorial(m) * np.exp(-tau * b)
    return t1 * t2

""" Analystic solutions for maximum-likelihood(ML) estimators for a single bin. """
def get_mu_hat(n, m, s, tau):
    """ Signal strength that maximizes likelihood function unconditionally. See Eqn. 91 """
    return (n - m / tau) / s

def get_b_hat(m, tau):
    """ Nuissance parameter that maximizes likelihood function unconditionally. See Eqn. 92 """
    return m / tau

def get_b_2hat(n, m, mu, s, tau):
    """ Nuissance parameter that maximizes likelihood function for given mu. See Eqn. 93 """
    t1 = (n + m - (1 + tau) * mu * s) / 2 / (1 + tau)
    t2 = (n + m - (1 + tau * mu * s)) ** 2 + 4 * (1 + tau) * m * mu * s
    t2 /= 4 * (1 + tau) ** 2
    t2 **= 1 / 2
    return t1 + t2

def get_lambda(n, m, mu, b, s, tau):
    """ Get profile likelihood ratio. See Eqn. 7 on page 4. """
    t1 = get_likelihood(n, m, mu, get_b_2hat(n, m, mu, s, tau), s, tau)
    t2 = get_likelihood(n, m, get_mu_hat(n, m, s, tau), get_b_hat(m, tau), s, tau)
    return t1 / t2

""" Calculate test statistic q """
def get_q_0(n, m, b, s, tau):
    """ Calculate q at mu = 0. See Eqn. 12 on page 7."""
    mh_filter = np.where(get_mu_hat(n, m, s, tau) >= 0, 1, 0)
    return -2 * np.log(get_lambda(n, m, 0, b, s, tau)) * mh_filter

def get_q_mu(n, m, mu, b, s, tau):
    """ Calculate q given mu. See Eqn. 14 on page 7."""
    return np.where(get_mu_hat(n, m, s, tau) <= mu, -2 * np.log(get_lambda(n, m, mu, b, s, tau)), 0)

def gen_Z_0(ns, ms, q_0, b, s, tau, threshold = 1.0):
    """ Calculate significance Z_0. See Eqn. 1 on page 3 """
    q_0s = get_q_0(ns, ms, b, s, tau)
    p = 0
    for i in q_0s: 
#         if abs(i - q_0) < threshold: p += 1
        if i > q_0: p += 1
    p /= len(q_0s)
#    print(1 - p)
    return phi_inv(1 - p)

""" Calculate test statistic ~q_mu """
def get_t_lambda(n, m, mu, b, s, tau):
    """ See Eqn. 14 """

    func1 = lambda n, m, mu, s, tau: get_likelihood(n, m, mu, get_b_2hat(n, m,     mu, s, tau), s, tau) / get_likelihood(n, m, mu, get_b_hat(m, tau), s, tau)

    func2 = lambda n, m, mu, s, tau: get_likelihood(n, m, mu, get_b_2hat(n, m, mu, s, tau), s, tau) / get_likelihood(n, m, 0, get_b_2hat(n, m, 0, s, tau), s, tau)

    mu_hat = get_mu_hat(n, m, s, tau)
    res = np.where(mu_hat >= 0,  func1(n, m, mu, s, tau), 
            func2(n, m, mu, s, tau))
    return res

def get_t_q_mu(n, m, mu, b, s, tau):
    """ See Eqn. 16 """
    return np.where(get_mu_hat(n, m, s, tau) <= mu, -2 * np.log(get_t_lambda(n, m, mu, b, s, tau)), 0)
