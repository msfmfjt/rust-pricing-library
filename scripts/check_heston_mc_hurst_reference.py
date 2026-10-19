"""Independent 70-digit finite-grid drift/Black references; no pricing extension."""
import json
from pathlib import Path
import mpmath as mp
mp.mp.dps = 70
ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'fixtures/rough-volatility/mc-hurst.json'

def kernel(h, lo, hi):
    a = h + mp.mpf('.5')
    return (hi**a-lo**a)/(a*mp.gamma(a)*(hi-lo))

def integrated_variance(h):
    ts = [mp.mpf(i)/4 for i in range(5)]
    v = [mp.mpf('.04')]
    for i in range(1,5):
        v.append(v[0]+sum((ts[j+1]-ts[j])*kernel(h,ts[i]-ts[j+1],ts[i]-ts[j])
                          *mp.mpf('.7')*(mp.mpf('.055')-v[j]) for j in range(i)))
    assert min(v)>0
    return sum(v[:-1])/4

def references():
    weights = []
    for h in ['.001','.1','.3','.5']:
        for lo, hi in [('0','.25'),('.01','.26'),('10','10.00000001')]:
            h0,l,u=map(mp.mpf,(h,lo,hi))
            weights.append(dict(h=float(h0),lo=float(l),hi=float(u),
                # Reference uses the exact binary-f64 lag operands for the tiny cell.
                derivative=float(mp.diff(lambda x: kernel(x,mp.mpf(float(l)),mp.mpf(float(u))),mp.mpf(float(h0))))))
    black=[]
    for h in ['.1','.3','.5']:
        h0=mp.mpf(float(h)); q=integrated_variance(h0);dq=mp.diff(integrated_variance,h0)
        d1=mp.sqrt(q)/2
        price=100*mp.erf(d1/mp.sqrt(2))
        risk=100*mp.exp(-d1*d1/2)/mp.sqrt(2*mp.pi)*dq/(2*mp.sqrt(q))
        black.append(dict(h=float(h0),integrated_variance=float(q),q_hurst=float(dq),price=float(price),hurst=float(risk)))
    return dict(description='Independent finite-grid deterministic-variance law, NOT continuous-time Rough Heston',kernel=weights,black=black)

def check():
    stored=json.loads(PATH.read_text());computed=references()
    for section in ['kernel','black']:
        assert len(stored[section])==len(computed[section])
        for a,b in zip(stored[section],computed[section]):
            assert a.keys()==b.keys()
            for key in b: assert abs(a[key]-b[key])<=2e-13*(1+abs(b[key])),(section,key,a,b)
    return stored

if __name__=='__main__':
    import sys
    if sys.argv[1:]==['--generate']:
        PATH.write_text(json.dumps(references(),indent=2)+'\n')
    else:
        check();print('Independent 70-digit kernel and finite-grid Black references passed')
