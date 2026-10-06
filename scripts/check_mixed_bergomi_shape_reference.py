"""Independent two-step expectations: simplex transfer and original xi nodes.
No pricing extension or production kernel, interpolation, or reverse is imported.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np

FILE=Path(__file__).resolve().parents[1]/'fixtures/rough-volatility/mixed-shape.json'


def reference(h, order, parameters=(.35,.04,.05,.045)):
    weight, initial, xi, _ = parameters
    z,w=np.polynomial.hermite.hermgauss(order)
    z=z*math.sqrt(2);w=w/math.sqrt(math.pi)
    z0,z1,z2=np.meshgrid(z,z,z,indexing='ij')
    w0,w1,w2=np.meshgrid(w,w,w,indexing='ij')
    quadrature=w0*w1*w2
    dt=.5;rho=-.6
    x=dt**h/(h+.5)*(math.sqrt(2*h)*(rho*z0+math.sqrt(1-rho*rho)*z1)+(.5-h)*z2)
    e0=np.exp(.3*x-.5*.3**2*dt**(2*h))
    e1=np.exp(.8*x-.5*.8**2*dt**(2*h))
    v=xi*(weight*e0+(1-weight)*e1)
    f1=100*np.exp(-.5*initial*dt+math.sqrt(initial*dt)*z0)
    std=np.sqrt(v*dt);d1=np.log(f1/100)/std+.5*std
    cdf=lambda a:np.fromiter((.5*math.erfc(-float(x)/math.sqrt(2)) for x in a.flat),float,count=a.size).reshape(a.shape)
    n1=cdf(d1);price=f1*n1-100*cdf(d1-std)
    dv=f1*np.exp(-.5*d1*d1)/math.sqrt(2*math.pi)*dt/(2*std)
    fields=[price,dv*xi*(e0-e1),n1*f1*(-.5*dt+.5*math.sqrt(dt/initial)*z0),dv*v/xi,np.zeros_like(v)]
    return [float(np.sum(quadrature*f)) for f in fields]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    rows=[]
    for h in [.1,.3,.5]:
        a=reference(h,48);b=reference(h,64)
        gap=max(abs(x-y) for x,y in zip(a,b))
        assert gap<1e-7,(h,gap)
        # A separate price-only central stencil checks the analytic oracle.
        fds=[]
        for j,g in enumerate(b[1:]):
            for e in [5e-6,2.5e-6]:
                up=[.35,.04,.05,.045];dn=up.copy();up[j]+=e;dn[j]-=e
                fd=(reference(h,48,up)[0]-reference(h,48,dn)[0])/(2*e)
                assert abs(fd-g)<1e-6*(1+abs(g)),(h,j,fd,g)
                fds.append(abs(fd-g))
        rows.append(dict(hurst=h,price=b[0],parameter_adjoints=b[1:],order48=a,order64=b,max_resolution_difference=gap,max_price_stencil_difference=max(fds)))
    result=dict(method='independent-two-step-conditional-black-shape',weights=[.35,.65],etas=[.3,.8],rho=-.6,xi_times=[0.,.5,1.],xi_values=[.04,.05,.045],rows=rows)
    if args.write:FILE.write_text(json.dumps(result,indent=2)+'\n')
    else:
        old=json.loads(FILE.read_text());assert old.keys()==result.keys()
        for a,b in zip(old['rows'],rows,strict=True):
            assert a['hurst']==b['hurst']
            for x,y in zip(a['order64'],b['order64'],strict=True):assert abs(x-y)<1e-10*(1+abs(y))
    print(json.dumps(result))
if __name__=='__main__':main()
