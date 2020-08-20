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

def get_std_dev(mu_p, b, s, tau):
    """ Calculate std. dev. of q_0. See Eqn. 32"""
    q_0A = get_q_0A(mu_p, b, s, tau)
    return (mu_p ** 2 / q_0A) ** 0.5

def pdf(q_0, mu_p, b, s, tau):
    """ pdf function for the given mu prime. See Eqn. 48 """
    if mu_p == 0: return pdf_mu0(q_0)
    q_0 = np.array(q_0)[q_0 != 0]    
    std_dev = get_std_dev(mu_p, b, s, tau)
    return 0.5 / np.sqrt(2 * np.pi * q_0) * np.exp(-0.5 * (q_0 ** 0.5 - mu_p / std_dev) ** 2)

def get_Z_0(q_0):
    return np.sqrt(q_0)
