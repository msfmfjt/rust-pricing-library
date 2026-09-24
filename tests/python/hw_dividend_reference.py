"""Independent Gaussian/variation-of-constants oracle; never imports rust_pricing.

The option target is ONE finite Buehler split with a continuous-time cash reserve,
not the continuous-time stochastic-dividend option price.
"""
import math
import numpy as np


def b(a, t):
    return t if a == 0 else -math.expm1(-a*t)/a


def j(a, t):
    return t*t/2 if a == 0 else (t-b(a, t))/a


def vi(a, sr, t):
    return sr*sr*(t**3/3 if a == 0 else (t-2*b(a,t)+b(2*a,t))/(a*a))


def coefficients(t, maturity, k=.7, alpha=.6, sd=.35, sf=.2,
                 a=.4, sr=.04, rho_f=.25, rho_d=-.2):
    tau=maturity-t
    total=sr*j(a,tau)
    by=math.exp(-k*tau-sd*rho_d*total)
    z,w=np.polynomial.legendre.leggauss(96)
    af=ac=0.
    for u,weight in zip((z+1)*tau/2,w*tau/2):
        tail=sr*j(a,u)
        common=-k*u-sd*rho_d*tail
        af+=weight*k*math.exp(common-sf*rho_f*(total-tail))
        ac+=weight*k*math.exp(common)
    return np.array([alpha*af,by,(1-alpha)*ac])


def conditional_price(order, k=.7):
    sf,sd,a,sr=.2,.35,.4,.04
    rho_sd,rho_f,rho_d=-.25,.25,-.2
    r,s=-math.log(.95),-math.log(.98)
    cash=[(1.,3.),(1.4,8.)]
    def bond(t,u,x):
        # Gaussian affine bond from conditional integral moments.
        return np.exp(-r*(u-t)-b(a,u-t)*x-.5*(vi(a,sr,u)-vi(a,sr,t))+.5*vi(a,sr,u-t))
    initial_reserve=sum(mean*math.exp(s*u-r*u)*sum(coefficients(0,u,k=k)) for u,mean in cash)
    risky=100-initial_reserve
    T=1.
    c=np.zeros((4,4)) # Wf, WD, x, integral x
    c[0,0]=c[1,1]=T
    c[0,1]=c[1,0]=rho_sd*T
    c[0,2]=c[2,0]=rho_f*sr*b(a,T)
    c[1,2]=c[2,1]=rho_d*sr*b(a,T)
    c[0,3]=c[3,0]=rho_f*sr*j(a,T)
    c[1,3]=c[3,1]=rho_d*sr*j(a,T)
    c[2,2]=sr*sr*b(2*a,T)
    c[2,3]=c[3,2]=.5*sr*sr*b(a,T)**2
    c[3,3]=vi(a,sr,T)
    conditional=np.linalg.solve(c[1:,1:],c[1:,0])
    variance=sf*sf*(T-c[0,1:]@conditional)
    root=math.sqrt(variance)
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    z=np.stack(np.meshgrid(*([nodes*math.sqrt(2)]*3),indexing='ij'),axis=-1).reshape(-1,3)
    w=np.prod(np.stack(np.meshgrid(*([weights/math.sqrt(math.pi)]*3),indexing='ij'),axis=-1),axis=-1).ravel()
    state=z@np.linalg.cholesky(c[1:,1:]).T
    wd,x,ix=state.T
    fmean=np.exp(-.5*sf*sf*T+sf*(state@conditional)+.5*variance)
    q=math.exp(-.5*k*T)
    A=risky*math.exp((r-s)*T+.5*vi(a,sr,T))*np.exp(ix)
    R=np.zeros_like(A)
    for u,mean in cash:
        if u<=T:
            continue
        af,by,ac=coefficients(T,u,k=k)
        scale=mean*math.exp(s*(u-T))*bond(T,u,x)
        A+=scale*(af+by*(1-q)*.6)
        R+=scale*(by*q*np.exp(-.5*sd*sd*T+sd*wd)+by*(1-q)*.4+ac)
    strike=100-R
    forward=A*fmean
    positive=strike>0
    call=forward-strike
    d1=np.log(forward[positive]/strike[positive])/root+.5*root
    cdf=np.vectorize(lambda x: .5*math.erfc(-x/math.sqrt(2)),otypes=[float])
    call[positive]=forward[positive]*cdf(d1)-strike[positive]*cdf(d1-root)
    df=np.exp(-r*T-ix-.5*vi(a,sr,T))
    return float(w@(df*call))

if __name__=='__main__':
    for k in [0.,.7]:
        for n in [24,32,40]:
            print(k,n,conditional_price(n,k))
