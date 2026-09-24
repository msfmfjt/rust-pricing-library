"""Independent two-step rough/Buehler price quadrature, not a continuous-time oracle.

Only NumPy/math are used: no production covariance, path, reserve or payoff code.
At t=1/2 the first Volterra cell is exact; integrate (W_f,W_D,X) jointly and
condition the final equity increment on the final dividend increment.
"""
import itertools
import math
import numpy as np


def reference(order, hurst=0.1):
    dt, sigma0, eta, kd, alpha, nu_d, sd = .5, .2, .6, .7, .6, .35, -.25
    near = math.sqrt(2*hurst)*dt**(hurst+.5)/(hurst+.5)
    cov = np.array([[dt, sd*dt, -.4*near],
                    [sd*dt, dt, .15*near],
                    [-.4*near, .15*near, dt**(2*hurst)]])
    lower = np.linalg.cholesky(cov)
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes, weights = nodes*math.sqrt(2), weights/math.sqrt(math.pi)
    growth = .98/.95
    reserve0 = 25/growth**1.4
    c0 = growth*reserve0
    w = math.exp(-kd*.4)
    A = (100-reserve0)*growth+c0*(1-w)*alpha
    B = c0*w
    C = c0*(1-w)*(1-alpha)
    a = math.exp(-kd*dt/2)
    b = -math.expm1(-kd*dt/2)
    cdf = np.vectorize(lambda z: .5*math.erfc(-z/math.sqrt(2)))
    total = 0.
    for index in itertools.product(range(order), repeat=3):
        df, dd, x = lower@nodes[list(index)]
        f = math.exp(-.5*sigma0**2*dt+sigma0*df)
        y = a*math.exp(-.5*nu_d**2*dt+nu_d*dd)+b*(alpha*f+1-alpha)
        sigma = sigma0*math.exp(.5*eta*x-.25*eta**2*dt**(2*hurst))
        half = a*y+b*(alpha*f+1-alpha)
        known = B*a*half*np.exp(-.5*nu_d**2*dt+nu_d*math.sqrt(dt)*nodes)+B*b*(1-alpha)+C
        strike = 100-known
        forward = (A+B*b*alpha)*f*np.exp(-.5*(sigma*sd)**2*dt+sigma*sd*math.sqrt(dt)*nodes)
        root = sigma*math.sqrt(dt*(1-sd*sd))
        d1 = (np.log(forward)-np.log(np.where(strike>0,strike,1.0)))/root+.5*root
        call = np.where(strike>0, forward*cdf(d1)-strike*cdf(d1-root), forward-strike)
        total += np.prod(weights[list(index)])*float(weights@call)
    return .95*total


if __name__ == '__main__':
    for h in (.1, .5):
        for n in (12, 16, 20, 24):
            print(h, n, reference(n, h), flush=True)
