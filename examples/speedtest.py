from typing import Callable
import time
import numpy as np

from montecarlo import *

def speed_test(fn: Callable, *args, **kwargs) -> float:
    """ Measures the speed of given function fn with 
    given arguments. Returns output and delta time. """
    start = time.time()
    out = fn(*args, **kwargs)
    dt = time.time() - start

    args_str = ', '.join([str(i) for i in args])
    kwargs_str = ', '.join(['{k}={v}' for k, v in kwargs.items()])
    if (args and kwargs): kwargs_str = f', {kwargs_str}'
    print(f'{fn.__name__}({args_str}{kwargs_str}) took {dt} seconds to run')
    return out, dt

def iter_q_0(mu, b, s, tau, gen_num=int(10e4), seed=0):
    np.random.seed(seed)
    ns = gen_ns(mu, s, b, gen_num)
    ms = gen_ms(b, tau, gen_num)
    q_0s = []

    for n, m in zip(ns, ms):
        if get_mu_hat(n, m, s, tau) >= 0:
            q_0s.append(-2 * np.log(get_lambda(n, m, 0, b, s, tau)))
        else:
            q_0s.append(0)
    return q_0s

def vec_q_0(mu, b, s, tau, gen_num=int(10e4), seed=0):
    np.random.seed(seed)
    ns = gen_ns(mu, s, b, gen_num)
    ms = gen_ms(b, tau, gen_num)
    q_0s = np.where(get_mu_hat(ns, ms, s, tau) >= 0, 
            -2 * np.log(get_lambda(ns, ms, 0, b, s, tau)), 0)
    return q_0s

if __name__ == '__main__':
    iter_res, _ = speed_test(iter_q_0, 0, 20, 10, 1)
    vec_res, _ = speed_test(vec_q_0, 0, 20, 10, 1)
    assert(iter_res == list(vec_res))
