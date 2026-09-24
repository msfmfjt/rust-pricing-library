"""Independent two-step integration; no production path, RNG or covariance helpers.

This integrates the documented finite-step scheme, not its continuous-time limit.
"""
import itertools
import math
import numpy as np


def reference(order, two=False):
    h, sigma0, nu, kd, alpha, eta, sd = .5, .2, .3, .7, .6, .35, -.25
    k = np.array([.8, 2.1] if two else [.8])
    sv, dv = ([-.4, -.2], [.15, -.1]) if two else ([-.4], [.15])
    n = len(k)
    rho = np.eye(n+2)
    rho[0, 1] = rho[1, 0] = sd
    rho[0, 2:] = rho[2:, 0] = sv
    rho[1, 2:] = rho[2:, 1] = dv
    w = np.ones(1)
    if two:
        theta, vv = .35, .3
        rho[2, 3] = rho[3, 2] = vv
        w = np.array([1-theta, theta])/math.sqrt((1-theta)**2+theta**2+2*theta*(1-theta)*vv)
    rates = np.r_[0., 0., k]
    cov = np.empty_like(rho)
    for i in range(n+2):
        for j in range(n+2):
            rate = rates[i]+rates[j]
            integral = h if rate == 0 else -math.expm1(-rate*h)/rate
            cov[i,j] = rho[i,j]*integral
    lower = np.linalg.cholesky(cov)
    factor_var = w@cov[2:,2:]@w
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes, weights = nodes*math.sqrt(2), weights/math.sqrt(math.pi)
    growth = .98/.95
    reserve0 = 25/growth**1.4
    c0 = growth*reserve0
    decay_terminal = math.exp(-kd*.4)
    A = (100-reserve0)*growth+c0*(1-decay_terminal)*alpha
    B = c0*decay_terminal
    C = c0*(1-decay_terminal)*(1-alpha)
    a = math.exp(-kd*h/2)
    b = -math.expm1(-kd*h/2)
    cdf = np.vectorize(lambda z: .5*math.erfc(-z/math.sqrt(2)))
    total = 0.
    # First step: integrate joint equity, dividend and OU innovations. Second
    # step: integrate the dividend normal, analytically integrate equity's
    # conditional independent normal. Endpoint OU innovations do not affect payoff.
    for index in itertools.product(range(order), repeat=n+2):
        innovations = lower@nodes[list(index)]
        f = math.exp(-.5*sigma0**2*h+sigma0*innovations[0])
        y = a*math.exp(-.5*eta**2*h+eta*innovations[1])+b*(alpha*f+1-alpha)
        sigma = sigma0*math.exp(nu*(w@innovations[2:])-nu**2*factor_var)
        half = a*y+b*(alpha*f+1-alpha)
        known = B*a*half*np.exp(-.5*eta**2*h+eta*math.sqrt(h)*nodes)+B*b*(1-alpha)+C
        K = 100-known
        F = (A+B*b*alpha)*f*np.exp(-.5*(sigma*sd)**2*h+sigma*sd*math.sqrt(h)*nodes)
        root = sigma*math.sqrt(h*(1-sd*sd))
        d1 = (np.log(F)-np.log(np.where(K>0,K,1.0)))/root+.5*root
        call = np.where(K>0, F*cdf(d1)-K*cdf(d1-root), F-K)
        total += np.prod(weights[list(index)])*float(weights@call)
    return .95*total

if __name__ == '__main__':
    for two in (False,True):
        for order in (8,12,16,20):
            print(two,order,reference(order,two),flush=True)
