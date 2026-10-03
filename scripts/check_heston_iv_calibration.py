"""Independent Black/SSVI target references; never imports the pricing extension.

Black uses erfc and Brent root finding. SSVI uses SciPy PCHIP and direct formulas.
Markov prices are the unchanged independently computed parent Fourier fixture.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / 'fixtures/rough-volatility/iv-calibration.json'
PARENT = ROOT / 'fixtures/rough-volatility/fourier.json'
PROTOCOL = {
    'time_steps': 128, 'integration_intervals': 512, 'cutoff': 128.0,
    'fine_time_steps': 256, 'fine_integration_intervals': 1024, 'fine_cutoff': 256.0,
    'max_iterations': 60, 'max_evaluations': 100, 'residual_tolerance': 2e-6,
    'fine_iv_budget': 0.0005, 'ssvi_objective_ratio': 0.5,
    'ssvi_max_iv_residual': 0.005, 'iv_reference_tolerance': 2e-8,
    'black_price_tolerance': 2e-12,
    'parameters': [0.04, 0.7, 0.055, 0.18, -0.65, 0.1],
    'starts': [[0.05, 0.9, 0.05, 0.23, -0.5, 0.15], [0.035, 0.5, 0.07, 0.14, -0.75, 0.07]],
    'bounds': [[0.01, 0.1, 0.04], [0.1, 2.0, 0.7], [0.015, 0.12, 0.05],
               [0.03, 0.4, 0.2], [-0.95, -0.05, 0.5], [0.02, 0.49, 0.2]],
    'lift_weights': [0.2, 0.4, 0.5], 'lift_rates': [0.1, 1.0, 8.0],
    'synthetic_maturities': [0.25, 0.75, 1.5], 'synthetic_strikes': [85., 100., 115.],
    'ssvi_fit_theta_times': [0.25, 0.75, 1.5], 'ssvi_fit_theta_values': [0.01, 0.03, 0.06],
    'ssvi_fit_terminal_slope': 0.04, 'ssvi_fit_rho': -0.5,
    'ssvi_fit_eta': 0.35, 'ssvi_fit_gamma': 0.5,
}

def cdf(x):
    return math.erfc(-x / math.sqrt(2)) / 2

def black(f, k, d, t, sigma, call=True):
    a = sigma * math.sqrt(t)
    d1 = math.log(f / k) / a + a / 2
    sign = 1 if call else -1
    return d * sign * (f * cdf(sign * d1) - k * cdf(sign * (d1 - a)))

def implied(price, f, k, d, t, call=True):
    return brentq(lambda sigma: black(f,k,d,t,sigma,call)-price,
                  1e-8, 8/math.sqrt(t), xtol=1e-14)

PARENT_SHA256 = '8e47af65d9262c5a29f321199a0c2629b3e1a50a77c6bfc674a75dc5ba4db1ad'

def references():
    if hashlib.sha256(PARENT.read_bytes()).hexdigest()!=PARENT_SHA256:
        raise ValueError("parent independent fixture changed")
    black_rows = []
    for t in [.1,.5,2.]:
        for f in [73.,100.]:
            for m in [.9,1.,1.1]:
                for sigma in [.15,.3]:
                    k=f*m; d=.97
                    d1=math.log(f/k)/(sigma*math.sqrt(t))+sigma*math.sqrt(t)/2
                    black_rows.append(dict(maturity=t,forward=f,strike=k,discount=d,volatility=sigma,
                        call=black(f,k,d,t,sigma),put=black(f,k,d,t,sigma,False),
                        vega=d*f*math.exp(-d1*d1/2)/math.sqrt(2*math.pi)*math.sqrt(t)))
    ssvi = dict(theta_times=[.25,.75,1.5],theta_values=[.012,.035,.072],terminal_theta_slope=.048,
                rho=-.5, eta=.35,gamma=.5,lambda_=1.0)
    curve=PchipInterpolator(ssvi['theta_times'],ssvi['theta_values'])
    rows=[]
    for kind in ['power_law','heston_like']:
        for t in [.125,.25,.5,.75,1.,1.5,3.]:
            if t<.25:theta=t*.012/.25
            elif t>1.5:theta=.072+(t-1.5)*.048
            else:theta=float(curve(t))
            if kind=='power_law':phi=.35/(theta**.5*(1+theta)**.5)
            else:
                # Direct high-precision expression avoids the small-z cancellation.
                import mpmath as mp
                with mp.workdps(50):
                    z=mp.mpf(theta);phi=float((z-1+mp.exp(-z))/(z*z))
            for klog in [-.2,0.,.2]:
                w=theta/2*(1+ssvi['rho']*phi*klog+math.sqrt((phi*klog+ssvi['rho'])**2+1-ssvi['rho']**2))
                f=107.;k=f*math.exp(klog)
                rows.append(dict(kind=kind,maturity=t,forward=f,strike=k,discount=.97,
                                 target_volatility=math.sqrt(w/t), is_call=k>=f))
    parent=json.loads(PARENT.read_text())
    markov=[]
    for row in parent['prices']:
        markov.append(row|{'target_volatility':implied(row['call'],parent['forward'],row['strike'],parent['discount'],row['maturity'])})
    return dict(version=1,protocol=PROTOCOL,parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),
                black=black_rows,ssvi=ssvi,ssvi_quotes=rows,markov_quotes=markov)

def check(data):
    expected=references()
    def walk(a,b,path=''):
        if isinstance(b,dict):
            if not isinstance(a,dict) or a.keys()!=b.keys():raise ValueError('keys '+path)
            for k in b:walk(a[k],b[k],path+'/'+k)
        elif isinstance(b,list):
            if not isinstance(a,list) or len(a)!=len(b):raise ValueError('length '+path)
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,path+'/'+str(i))
        elif isinstance(b,float) and '/protocol/' not in path and '/ssvi/' not in path:
            if not isinstance(a,(int,float)) or not math.isfinite(a) or abs(a-b)>2e-12*max(1.,abs(b)):
                raise ValueError('value '+path)
        elif a!=b:raise ValueError('constant '+path)
    walk(data,expected)
    for row in data['black']:
        for side in ['call','put']:
            v=implied(row[side],row['forward'],row['strike'],row['discount'],row['maturity'],side=='call')
            if abs(v-row['volatility'])>2e-11:raise ValueError('Black independent round trip')

def main():
    check(json.loads(FILE.read_text()))
    print('PASS: 36 independent Black rows, 42 SSVI queries, 12 Markov IV targets and fixed protocol')

if __name__=='__main__':
    if sys.argv[1:]==['--write-reference']:
        FILE.write_text(json.dumps(references(),indent=2)+'\n')
    else:main()
