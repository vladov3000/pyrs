""" Formulas for asymptotic approximation method. """
from scipy.stats import norm
from utils import *
from montecarlo import *

def pdf_mu0(q_0):
    """ Formula for pdf at mu=0. See Eqn. 49 on page 14 """
    q_0 = np.array(q_0)[q_0 != 0]
    return (8 * np.pi * q_0) ** -0.5 * np.exp(-0.5 * q_0)

def get_q_0A(mu, b, s, tau):
    """ Solved for in single-bin example """
    return get_q_0(mu * s + b, b * tau, b, s, tau)

def get_q_muA(mu, mu_p, b, s, tau):
    return get_q_mu(mu_p * s + b, b * tau, mu, b, s, tau)

def get_std_dev_q_0(mu_p, b, s, tau):
    """ Calculate std. dev. of q_0. see eqn. 32"""
    q_0A = get_q_0A(mu_p, b, s, tau)
    return (mu_p ** 2 / q_0A) ** 0.5

def get_std_dev(q_muA, mu, mu_p):
    """ Calculate std. dev. of q_0. see eqn. 30 """
    return ((mu - mu_p) ** 2 / q_muA) ** 0.5

def pdf(q_0, mu_p, b, s, tau):
    """ pdf function for the given mu prime. See Eqn. 48 """
    if mu_p == 0: return pdf_mu0(q_0)
    q_0 = np.array(q_0)[q_0 != 0]
    std_dev = get_std_dev_q_0(mu_p, b, s, tau)
    return 0.5 / np.sqrt(2 * np.pi * q_0) * np.exp(-0.5 * (q_0 ** 0.5 - mu_p / std_dev) ** 2)

def get_Z_0(q_0):
    return np.sqrt(q_0)

def pdf_q_mu(q_mu, mu, mu_p, b, s, tau):
    """ See Eqn. 56 """
    if mu == mu_p: return (8 * np.pi * q_mu) ** -0.5 * np.exp(-0.5 * q_mu)

    return (8 * np.pi * q_mu) ** -0.5 * np.exp(-0.5 * (q_mu ** 0.5  - (mu - mu_p) / get_std_dev(get_q_muA(mu, mu_p, b, s, tau), mu, mu_p)) ** 2)

def pdf_t_q_mu(t_q_mu, mu, mu_p, b, s, tau):
    """ See Eqn. 63 """
    sigma = get_std_dev(get_q_muA(mu, mu_p, b, s, tau), mu, mu_p)
    sigma = 1.5 ** -0.5
    bound = mu ** 2 / sigma ** 2

    return np.where(t_q_mu > bound,
            1 / ((8 * np.pi) ** 0.5 * mu / sigma) * np.exp(-0.5 * (t_q_mu - (mu ** 2 - 2 * mu * mu_p) / sigma ** 2) ** 2 / (2 * mu / sigma) ** 2),
            (8 * np.pi * t_q_mu) ** -0.5 * np.exp(-0.5 * (t_q_mu ** 0.5 - (mu - mu_p) / sigma) ** 2)
            )
