"""Independent two-step conditional-Black/Gauss-Hermite references (no extension).
The first asset normal, independent variance normal and newest-cell residual
are integrated directly. The last asset normal is integrated analytically.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT/'fixtures/rough-volatility/mixed-parameter.json'


def reference(h, order):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    nodes = np.sqrt(2.)*nodes; weights = weights/math.sqrt(math.pi)
    z0,z1,z2 = np.meshgrid(nodes,nodes,nodes,indexing='ij')
    w0,w1,w2 = np.meshgrid(weights,weights,weights,indexing='ij')
    rho=-.6; q=math.sqrt(1-rho*rho);dt=.5
    c=math.sqrt(2*h)*dt**h/(h+.5)
    r=dt**h*(.5-h)/(h+.5)
    x=c*(rho*z0+q*z1)+r*z2
    dx=c*(z0-rho/q*z1)
    v=np.zeros_like(x); dv=[]
    for weight,eta in zip([.35,.65],[.3,.8]):
        comp=.05*weight*np.exp(eta*x-.5*eta*eta*dt**(2*h))
        v+=comp;dv.append(comp*(x-eta*dt**(2*h)))
    dv.append(sum(.05*weight*np.exp(eta*x-.5*eta*eta*dt**(2*h))*eta*dx
        for weight,eta in zip([.35,.65],[.3,.8])))
    f1=100*np.exp(-.5*.04*dt+math.sqrt(.04*dt)*z0)
    std=np.sqrt(v*dt);d1=np.log(f1/100)/std+.5*std;d2=d1-std
    cdf=lambda a: np.fromiter((.5*math.erfc(-float(x)/math.sqrt(2)) for x in a.flat),float,count=a.size).reshape(a.shape)
    price=f1*cdf(d1)-100*cdf(d2)
    var_derivative=f1*np.exp(-.5*d1*d1)/math.sqrt(2*math.pi)*dt/(2*std)
    quadrature=w0*w1*w2
    return np.array([np.sum(quadrature*price)]+[np.sum(quadrature*var_derivative*d) for d in dv]).tolist()


def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    rows=[]
    for h in [.1,.3,.5]:
        a=reference(h,32);b=reference(h,48);c=reference(h,64)
        gap=max(abs(x-y) for x,y in zip(b,c))
        if gap>1e-7:raise AssertionError((h,'48/64 quadrature difference',gap))
        rows.append(dict(hurst=h,price=c[0],parameter_adjoints=c[1:],order32=a,order48=b,order64=c,max_refinement_difference=gap))
    if args.write:
        FILE.write_text(json.dumps({'method':'independent-two-step-conditional-black-gh','dt':.5,'spot':100.,'strike':100.,
        'initial_variance':.04,'half_time_forward_variance':.05,'weights':[.35,.65],'etas':[.3,.8],'rho':-.6,'rows':rows},indent=2)+'\n')
    else:
        retained=json.loads(FILE.read_text())['rows']
        for old,new in zip(retained,rows,strict=True):
            assert old['hurst']==new['hurst']
            for x,y in zip(old['order64'],new['order64']):assert abs(x-y)<=1e-10*(1+abs(y))
    print(json.dumps({'reference_rows':rows,'status':'passed'}))

if __name__=='__main__':main()
