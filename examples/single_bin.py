""" See page 22 for section 5.1 description """
import matplotlib.pyplot as plt
import numpy as np
import sys
import gc
import os

from montecarlo import *
from asymptotic import *

def fig_3a(gen_num, path='plots', show=False):
    # Input parameters
    bs = [2, 5, 20]
    styles = {2: ('red','solid'), 
              5: ('brown', 'dashed'), 
              20: ('blue', 'dotted')} # linestyles & colors for each b
    tau = 1
    mu = 0
    s = 10

    # Setup axes
    axes = plt.gca()
    axes.set_yscale('log')
    axes.set_ylim([10 ** -8, 10])
    axes.set_xlim([0, 40])
    axes.set_xlabel('q_0')
    axes.set_ylabel('f (q_0 | 0)')
    plt.gcf().set_size_inches(5,5)

    # Plot f(q_0 | 0)
    x = np.linspace(0, 40, 40_000) # one dimensional vector that represents x values
    pdfs = np.vectorize(pdf_mu0)(x) # get vector of pdf_mu0 applied to every element of x
    plt.plot(x, pdfs, color="black", label=f'f(q_0|0)')

    # Plot montecarlo for given bs
    for b in bs:
        gc.collect()
        ns = gen_ns(mu, s, b, gen_num)
        ms = gen_ms(b, tau, gen_num)
        q_0s = get_q_0(ns, ms, b, s, tau)
        weights = np.ones_like(q_0s) / float(len(q_0s)) # normalize q_0s
        plt.hist(q_0s, weights=weights, histtype='step', label=f'b={b}', 
                 color=styles[b][0], linestyle=styles[b][1])

    plt.legend(loc="upper right")
    if path: plt.savefig(f'{path}/fig_3a_10e{int(np.log10(gen_num))}.png')
    if show: plt.show()

def fig_3b(gen_num, path='plots', show=False):
    """ Recreat fig 3 (b) on page 24. """

    # Input parameters
    b = 20
    tau = 1
    s = 10

    # Format axes
    axes = plt.gca()
    axes.set_yscale('log')
    axes.set_ylim([10 ** -8, 10])
    axes.set_xlim([0, 40])
    axes.set_xlabel('q_0')
    axes.set_ylabel('f (q_0 | mu)')
    plt.gcf().set_size_inches(5,5)

    # Plot f(q_0 | 0)
    x = np.linspace(0, 40, 40_000)[x != 0] # one dimensional vector that represents x values
    pdfs = pdf(x, 0, b, s, tau) # get vector of pdf_mu0 applied to every element of x
    plt.plot(x, pdfs, color="black")

    # Plot f(q_0 | 1)
    pdfs = pdf(x, 1, b, s, tau) # get vector of pdf applied to every element of x
    plt.plot(x, pdfs, color="black")

    # Get n and m
    ns = gen_ns(0, s, b, gen_num)
    ms = gen_ms(b, tau, gen_num)

    # Plot mu = 0 montecarlo
    q_0s = [get_q_0(ns[i], ms[i], b, s, tau) for i in range(0, len(ns))]
    weights = np.ones_like(q_0s) / float(len(q_0s))
    plt.hist(q_0s, weights=weights, histtype='step', color='blue', linestyle='dashed')

    #Get n and m for for mu=1
    #ns = gen_ns(1, s, b, gen_num)
    #ms = gen_ms(b, tau, gen_num)

    # Plot mu = 1 montecarlo
    q_0s_mu1 = [get_q_mu(ns[i], ms[i], 1, b, s, tau) for i in range(0, len(ns))]
    weights = np.ones_like(q_0s_mu1) / float(len(q_0s_mu1))
    plt.hist(q_0s_mu1, weights=weights, histtype='step', color='red', linestyle='dashed')

    if path: plt.savefig(f'{path}/fig_3b_10e{int(np.log10(gen_num))}.png')
    if show: plt.show()

def fig_4a(gen_num, path='plots', show=True):
    # Input parameter
    bs = [2, 3, 5, 8, 9, 10, 30, 60, 100]
    tau = 1
    s = 10
    q_0 = 16

    # Format axes
    axes = plt.gca()
    axes.set_xscale('log')
    axes.set_xlim([0.9, 101])
    axes.set_ylim([2, 6])
    axes.set_ylabel('Z_0')
    axes.set_xlabel('b')
    plt.gcf().set_size_inches(5,5)

    # Plot approximation
    plt.hlines(get_Z_0(q_0), -1, 101, color='blue', linestyles='dashed', label='Nominal Signficance')

    # Montecarlo method
    Z_0s = []
    for b in bs:
        ns = gen_ns(0, s, b, gen_num)
        ms = gen_ms(b, tau, gen_num)
        Z_0s.append(gen_Z_0(ns, ms, q_0, b, s, tau))
    print(Z_0s)
    plt.scatter(bs, Z_0s, color='black', label='Monte Carlo')

    plt.legend(loc="lower left")
    if path: plt.savefig(f'{path}/fig_4a_10e{int(np.log10(gen_num))}.png')
    if show: plt.show()

def fig_4b(gen_num, path='plots', show=True):
    # Input parameter
    bs = [2, 3, 5, 8, 9, 10, 30, 60, 100]
    tau = 1
    ss = [1, 2, 5, 10, 20]

    # Generate data
    q_0_meds = []
    for b in bs:
        for s in ss:
            save_file = f'data/fig_4b_10e{int(np.log10(gen_num))}_b{b}_s{s}.npy'
            print(f'{save_file} exists: {os.path.exists(save_file)}') 
            if not os.path.exists(save_file):
                ns = gen_ns(0, s, b, gen_num)
                ms = gen_ms(b, tau, gen_num)
                q_0s = get_q_mu(ns, ms, 1, b, s, tau)
                q_0s = q_0s[~np.isnan(q_0s)]
                np.save(f'data/fig_4b_10e{int(np.log10(gen_num))}_b{b}_s{s}.npy', q_0s)
            else:
                q_0s = np.load(save_file)
            q_0_meds.append(float(np.median(q_0s)))
            print(f'Median for s={s} b={b}: {q_0_meds[-1]}')
    print(q_0_meds)

    # Format axes
    axes = plt.gca()
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlim([0.9, 101])
    axes.set_ylim([0.001, 100])
    axes.set_ylabel('q_0')
    axes.set_xlabel('b')
    plt.gcf().set_size_inches(5,5)

    # Plot approximations
    approx = lambda x, y: get_q_mu(get_E_n(0, x, y), get_E_m(y, tau), 1, y, x, tau) # x = s and y = b
    b = np.linspace(1, 50, 1_000) # one dimensional vector that represents x values
    for s in ss:
        plt.plot(x, approx(s, b), color='blue')

    # Plot medians
    plt.scatter([bs[i // len(ss)] for i in range(len(bs) * len(ss))], q_0_meds, color='red', label='median[q_0 | 1]')

    plt.legend(loc="lower left")
    if path: plt.savefig(f'{path}/fig_4a_10e{int(np.log10(gen_num))}.png')
    if show: plt.show()

def fig_5a(gen_num, path='plots', show=True):
    """ Recreate fig 5a on page 25 """
    # Input parameters
    s = 6
    b = 9
    tau = 1

    # Setup axes
    axes = plt.gca()
    axes.set_yscale('log')
    axes.set_ylim([10 ** -3, 5.0])
    axes.set_xlim([0, 9])
    axes.set_xlabel('q_1')
    plt.gcf().set_size_inches(5,5)

    # Plot f(q_1 | 1)
    x = np.linspace(0, 40, 40_000)
    pdfs = pdf_q_mu(x, 1, 1, b, s, tau)
    plt.plot(x, pdfs, color="black", label='f(q_1|1)')

    # Plot f(q_1 | 0)
    x = np.linspace(0, 40, 40_000)
    pdfs = pdf_q_mu(x, 1, 0, b, s, tau)
    plt.plot(x, pdfs, color="black", label='f(q_1|0)')

    # Get n and m where mu_p = 1
    ns = gen_ns(1, s, b, gen_num)
    ms = gen_ms(b, tau, gen_num)

    # Plot mu_p = 1 montecarlo
    q_1s = get_q_mu(ns, ms, 1, b, s, tau)
    plt.hist(q_1s, density=True, histtype='step', color='blue', linestyle='dashed', bins=np.linspace(0, 9, 20))

    # Get n and m where mu_p = 0
    ns = gen_ns(0, s, b, gen_num)
    ms = gen_ms(b, tau, gen_num)

    # Plot mu = 1 montecarlo
    q_1s = get_q_mu(ns, ms, 1, b, s, tau)
    plt.hist(q_1s, density=True, histtype='step', color='red', linestyle='dashed', bins=np.linspace(0, 9, 20))

    plt.legend(loc="lower left")
    if path: plt.savefig(f'{path}/fig_5a_10e{int(np.log10(gen_num))}.png')
    if show: plt.show()

def fig_5b(gen_num, path='plots', show=True):
    """ Recreate fig 5a on page 25 """
    # Input parameters
    s = 6
    b = 9
    tau = 1

    # Setup axes
    axes = plt.gca()
    axes.set_yscale('log')
    axes.set_ylim([10 ** -3, 2.0])
    axes.set_xlim([0, 9])
    axes.set_xlabel('q_1')
    plt.gcf().set_size_inches(5,5)

    # Plot f(~q_1 | 1)
    x = np.linspace(0, 40, 40_000)
    pdfs = pdf_t_q_mu(x, 1, 1, b, s, tau)
    plt.plot(x, pdfs, color="black", label='f(~q_1|1)')

    # Plot f(~q_1 | 0)
    x = np.linspace(0, 40, 40_000)
    pdfs = pdf_t_q_mu(x, 1, 0, b, s, tau)
    plt.plot(x, pdfs, color="black", label='f(~q_1|0)')

    # Get n and m where mu_hat = 1
    ns = gen_ns(1, s, b, gen_num)
    ms = gen_ms(b, tau, gen_num)

    # Plot mu_p = 1 montecarlo
    q_1s = get_t_q_mu(ns, ms, 1, b, s, tau)
    plt.hist(q_1s, density=True, histtype='step', color='blue', linestyle='dashed', bins=np.linspace(0, 9, 20))

    # Get n and m where mu_hat = 0
    ns = gen_ns(0, s, b, gen_num)
    ms = gen_ms(b, tau, gen_num)

    # Plot mu_p = 0 montecarlo
    q_1s = get_t_q_mu(ns, ms, 1, b, s, tau)
    plt.hist(q_1s, density=True, histtype='step', color='red', linestyle='dashed', bins=np.linspace(0, 9, 20))

    plt.legend(loc="lower left")
    if path: plt.savefig(f'{path}/fig_5b_10e{int(np.log10(gen_num))}.png')
    if show: plt.show()

def main():
   com = {'fig_3a': fig_3a,
           'fig_3b': fig_3b,
           'fig_4a': fig_4a,
           'fig_4b': fig_4b,
           'fig_5a': fig_5a,
           'fig_5b': fig_5b,
         }
   com[sys.argv[1]](int(float(sys.argv[2])))

if __name__ == '__main__':
    main()
