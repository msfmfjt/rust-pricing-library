"""Independent normalized-kernel and nondegenerate two-step Hurst expectations.
No pricing extension or production kernel/adjoint helper is imported.
"""
import argparse
import json
import math
from pathlib import Path
import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / 'fixtures/rough-volatility/mixed-hurst.json'


def kernel_reference(h, ts):
    mp.mp.dps = 70
    h = mp.mpf(h)
    times = list(map(mp.mpf, ts))
    def kernel(hh):
        older, near, residual, variance = [[]], [], [], [mp.mpf(0)]
        a = hh + mp.mpf('.5')
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            near.append(mp.sqrt(2*hh)*dt**(hh-mp.mpf('.5'))/a)
            residual.append(dt**hh*(mp.mpf('.5')-hh)/a)
            row = []
            for j in range(i-1):
                lo, hi = times[i]-times[j+1], times[i]-times[j]
                row.append(mp.sqrt(2*hh)*(hi**a-lo**a)/(a*(hi-lo)))
            older.append(row)
            variance.append(dt**(2*hh) + sum(w*w*(times[j+1]-times[j]) for j,w in enumerate(row)))
        return [*near, *residual, *variance, *(w for row in older for w in row)]
    d = [float(mp.diff(lambda hh: kernel(hh)[j], h)) for j in range(len(kernel(h)))]
    return dict(hurst=float(h), times=ts, derivatives=d)


def reference(h, order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes = np.sqrt(2.)*nodes
    weights = weights/math.sqrt(math.pi)
    z0,z1,z2 = np.meshgrid(nodes,nodes,nodes,indexing='ij')
    w0,w1,w2 = np.meshgrid(weights,weights,weights,indexing='ij')
    dt=.5; rho=-.6; q=math.sqrt(1-rho*rho); a=h+.5
    c=math.sqrt(2*h)*dt**h/a
    r=dt**h*(.5-h)/a
    x=c*(rho*z0+q*z1)+r*z2
    dx=c*(.5/h+math.log(dt)-1/a)*(rho*z0+q*z1) + dt**h*((.5-h)/a*math.log(dt)-1/a**2)*z2
    variance=dt**(2*h); dv=2*math.log(dt)*variance
    v=np.zeros_like(x); vh=np.zeros_like(x)
    for weight,eta in zip([.35,.65],[.3,.8]):
        comp=.05*weight*np.exp(eta*x-.5*eta*eta*variance)
        v+=comp
        vh+=comp*(eta*dx-.5*eta*eta*dv)
    f1=100*np.exp(-.5*.04*dt+math.sqrt(.04*dt)*z0)
    std=np.sqrt(v*dt); d1=np.log(f1/100)/std+.5*std; d2=d1-std
    def cdf(a):
        return np.fromiter((.5*math.erfc(-float(v)/math.sqrt(2)) for v in a.flat),float,count=a.size).reshape(a.shape)
    price=f1*cdf(d1)-100*cdf(d2)
    var_derivative=f1*np.exp(-.5*d1*d1)/math.sqrt(2*math.pi)*dt/(2*std)
    quadrature=w0*w1*w2
    return [float(np.sum(quadrature*price)),float(np.sum(quadrature*var_derivative*vh))]


def build():
    kernels=[kernel_reference(h, ts) for h in [.001,.03,.1,.3,.5]
             for ts in [[0.,.07,.21,.63,1.], [0.,.00000001,10.00000001]]]
    rows=[]
    for h in [.03,.1,.3,.5]:
        a,b=reference(h,48),reference(h,64)
        gap=max(abs(x-y) for x,y in zip(a,b))
        assert gap<=1e-7, (h,'quadrature',gap)
        gaps=[]
        for e in [2e-5,1e-5]:
            if h==.5:
                fd=(3*b[0]-4*reference(h-e,64)[0]+reference(h-2*e,64)[0])/(2*e)
            else:
                fd=(reference(h+e,64)[0]-reference(h-e,64)[0])/(2*e)
            gaps.append(abs(fd-b[1]))
            assert gaps[-1]<=2e-7*(1+abs(b[1])), (h,'reference FD',gaps[-1])
        rows.append(dict(hurst=h,price=b[0],hurst_sensitivity=b[1],order48=a,order64=b,
                         quadrature_gap=gap,finite_difference_gaps=gaps))
    return dict(method='independent-normalized-kernel-and-two-step-conditional-black-hurst',kernel=kernels,rows=rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    data=build()
    if args.write:
        FILE.write_text(json.dumps(data,indent=2)+'\n')
    else:
        old=json.loads(FILE.read_text())
        assert len(old['kernel'])==len(data['kernel']) and len(old['rows'])==len(data['rows'])
        for x,y in zip(old['kernel'],data['kernel'],strict=True):
            assert x['hurst']==y['hurst'] and x['times']==y['times']
            for a,b in zip(x['derivatives'],y['derivatives'],strict=True):
                assert abs(a-b)<=2e-12*(1+abs(b))
        for x,y in zip(old['rows'],data['rows'],strict=True):
            assert x['hurst']==y['hurst']
            for a,b in zip(x['order64'],y['order64'],strict=True):
                assert abs(a-b)<=1e-10*(1+abs(b))
    print(json.dumps({'status':'passed','kernel_rows':len(data['kernel']),'expectations':data['rows']}))

if __name__=='__main__':main()
