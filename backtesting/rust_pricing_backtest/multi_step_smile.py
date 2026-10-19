"""Terminal-quote calibration on a variable log-Euler grid.

The first reporting maturity may follow several transitions and carry a smile.
Earlier rows and quotes remain fixed; this is not an all-maturity IV chain.
"""
from copy import deepcopy
import math
import numpy as np
from .discrete_smile import _black, _cdf, TwoStepSmileCalibrator
from .domain import BacktestError, require, fingerprint


def _rule(nodes, lo, hi, order, max_width):
    require(math.isfinite(max_width) and max_width>0 and math.isfinite(lo) and math.isfinite(hi) and hi<=100 and
            hi-lo>128*np.finfo(float).eps*(1+abs(lo)+abs(hi)),
            'multi_step_precision','integration window exceeds floating-point domain')
    cuts=np.r_[lo,nodes[(nodes>lo)&(nodes<hi)],hi]
    resolved=[lo]
    for left,right in zip(cuts[:-1],cuts[1:]):
        pieces=math.ceil((right-left)/max_width)
        require((len(resolved)+pieces)*order<=8192,'multi_step_budget','quadrature node budget exceeded')
        resolved.extend(np.linspace(left,right,pieces+1)[1:])
    cuts=np.array(resolved)
    z,w=np.polynomial.legendre.leggauss(order)
    return ((cuts[:-1]+cuts[1:])[:,None]/2+np.diff(cuts)[:,None]*z/2).ravel(),(np.diff(cuts)[:,None]*w/2).ravel()


def source_marginals(spot,times,nodes,rows,order):
    """Positive Gaussian kernels with no density renormalization or moment fit."""
    v0=float(np.interp(0.,nodes,rows[0])*times[1]);sd=math.sqrt(v0)
    require(math.isfinite(sd) and 0<sd<=4,'multi_step_precision','first variance outside precision domain')
    lo,hi=-v0/2-12*sd,-v0/2+12*sd;x,w=_rule(nodes,lo,hi,order,4*sd)
    w*=np.exp(-.5*((x+v0/2)/sd)**2)/(math.sqrt(2*math.pi)*sd)
    vmax=v0;ptail=2*float(_cdf(-12.));stail=spot*float(_cdf(-12-sd)+_cdf(-12+sd))
    out=[];work=0
    for row in range(len(times)-2):
        if row:
            dt=times[row+1]-times[row];variance=np.interp(x,nodes,rows[row])*dt
            mu=x-variance/2;sd=np.sqrt(variance);vmax+=max(rows[row])*dt
            lo,hi=-vmax/2-12*math.sqrt(vmax),vmax/2+12*math.sqrt(vmax)
            y,dy=_rule(nodes,lo,hi,order,4*math.sqrt(min(rows[row])*dt));density=np.empty_like(y)
            work+=len(x)*len(y)
            require(work<=400_000_000,'multi_step_budget','integration work budget exceeded')
            for a in range(0,len(y),256):
                z=(y[a:a+256,None]-mu)/sd
                density[a:a+256]=(np.exp(-z*z/2)/(math.sqrt(2*math.pi)*sd))@w
            x,w=y,dy*density
            ptail+=2*math.exp(-72);stail+=spot*(1+math.exp(lo))*math.exp(-72)
        s=spot*np.exp(x);mass=float(sum(w));stock=float(w@s)
        require(np.isfinite(w).all() and abs(mass-1)<=1e-10+ptail and abs(stock-spot)<=1e-10*spot+stail,
                'multi_step_quadrature','source probability or martingale integration failed')
        hats=np.array([np.interp(x,nodes,e) for e in np.eye(len(nodes))]).T
        out.append((s,w,hats,dict(transition_count=row+1,log_window=[lo,hi],probability=mass,
            first_spot_moment=stock,probability_tail_bound=ptail,spot_tail_bound=stail,renormalized=False)))
    return out


class MultiStepTerminalSmileCalibrator:
    def __init__(self,payload,*,quadrature_order=32,price_tolerance=1e-10,
                 maximum_iterations=20,singular_value_cutoff=1e-10):
        self.payload=deepcopy(payload)
        require(type(quadrature_order) is int and 16<=quadrature_order<=64 and type(maximum_iterations) is int and 1<=maximum_iterations<=100,
                'multi_step_policy','invalid quadrature order or iteration budget')
        require(math.isfinite(price_tolerance) and 1e-12<=price_tolerance<=1e-6 and
                math.isfinite(singular_value_cutoff) and 1e-14<=singular_value_cutoff<=1e-3,
                'multi_step_policy','invalid price tolerance or rank cutoff')
        self.order=quadrature_order;self.tolerance=price_tolerance;self.iterations=maximum_iterations;self.cutoff=singular_value_cutoff
        try:
            model=payload['model'];market=payload['market'];g=model['local_variance_grid'];b=model['reporting_iv_basis']
            require(model['type']=='local_volatility' and market['type']=='equity' and not market.get('discrete_dividends'),
                    'multi_step_scope','equity local volatility without dividends required')
            for name in ['discount_curve','dividend_curve']:
                c=market[name];ct=np.asarray(c['times']);df=np.asarray(c['discount_factors'])
                require(ct.ndim==1 and len(ct)>=2 and ct.shape==df.shape and np.isfinite(ct).all() and np.all(np.diff(ct)>0) and np.all(df==1),
                        'multi_step_scope','unit discount and zero carry required')
            self.times=np.asarray(g['time_nodes'],dtype=float);self.nodes=np.asarray(g['log_forward_moneyness_nodes'],dtype=float)
            self.quote_nodes=np.asarray(b['log_forward_moneyness_nodes'],dtype=float);self.maturities=np.asarray(b['maturity_nodes'],dtype=float)
            require(self.times.ndim==1 and 3<=len(self.times)<=17 and self.times[0]==0 and np.isfinite(self.times).all() and np.all(np.diff(self.times)>0),
                    'multi_step_scope','2..16 ordered intervals from zero required')
            for nodes,upper in [(self.nodes,33),(self.quote_nodes,8)]:
                require(nodes.ndim==1 and 2<=len(nodes)<=upper and np.isfinite(nodes).all() and np.all(np.diff(nodes)>0),
                        'multi_step_shape','ordered local and quote nodes required')
            require(self.nodes[0]<=0<=self.nodes[-1] and self.nodes[0]<=self.quote_nodes[0] and self.quote_nodes[-1]<=self.nodes[-1],
                    'multi_step_scope','grid must contain spot and quotes')
            require(self.maturities.ndim==1 and 1<=len(self.maturities)<=16 and np.isfinite(self.maturities).all() and np.all(np.diff(self.maturities)>0)
                and self.maturities[-1]==self.times[-1] and all(t in self.times[1:] for t in self.maturities),
                'multi_step_scope','reporting maturities must be a subset of simulation knots, including terminal time')
            self.indices=[int(np.flatnonzero(self.times==t)[0]) for t in self.maturities]
            self.variance=np.asarray(g['values'],dtype=float).reshape(len(self.times),len(self.nodes))
            self.iv=np.asarray(b['implied_volatilities'],dtype=float).reshape(len(self.maturities),len(self.quote_nodes))
            require(g.get('shape',list(self.variance.shape))==list(self.variance.shape) and b.get('shape',list(self.iv.shape))==list(self.iv.shape),
                    'multi_step_shape','declared grid shape mismatch')
            self.floor=float(g['floor']);self.cap=float(g['cap']);self.spot=float(market['spot']);self.dt=self.times[-1]-self.times[-2]
            require(0<self.floor<self.cap<math.inf and 0<self.spot<math.inf and np.isfinite(self.variance).all()
                    and np.all(self.variance>=self.floor) and np.all(self.variance<=self.cap) and np.isfinite(self.iv).all() and np.all(self.iv>0),
                    'multi_step_bounds','finite bounded variances, quotes and spot required')
            self.strikes=self.spot*np.exp(self.quote_nodes)
            require(np.isfinite(self.strikes).all() and np.all(self.strikes>0),'multi_step_bounds','invalid quote strikes')
        except (KeyError,TypeError,ValueError,IndexError,OverflowError) as error:
            if isinstance(error,BacktestError):raise
            raise BacktestError('multi_step_shape','invalid market or grid') from error
        self.coarse=source_marginals(self.spot,self.times,self.nodes,self.variance,self.order)
        self.fine=source_marginals(self.spot,self.times,self.nodes,self.variance,2*self.order)
        self.source_audit=[]
        for a,b in zip(self.coarse,self.fine):
            da=abs(a[3]['probability']-b[3]['probability']);ds=abs(a[3]['first_spot_moment']-b[3]['first_spot_moment'])
            require(da<=1e-10 and ds<=1e-10*self.spot,'multi_step_quadrature','source refinement failed')
            self.source_audit.append(dict(b[3],probability_refinement_difference=da,spot_refinement_difference=ds))
        self.integration=self.coarse[-1];self.verification_integration=self.fine[-1]
        self.tail_bound=self.source_audit[-1]['spot_tail_bound']
        for j,index in enumerate(self.indices):
            if index==1:
                prices,_=_black(self.spot,self.strikes,self.times[1]*np.interp(0.,self.nodes,self.variance[0]));error=0.
            else:
                a=self._row_prices(index,self.variance[index-1],self.coarse[index-2])[0]
                prices=self._row_prices(index,self.variance[index-1],self.fine[index-2])[0]
                error=max(abs(prices-a))+self.fine[index-2][3]['spot_tail_bound']
            target,_=_black(self.spot,self.strikes,self.maturities[j]*self.iv[j]**2)
            require(error<=self.tolerance and max(abs(prices-target))+error<=self.tolerance,
                    'multi_step_base_calibration','reported quotes do not match refined multi-step prices')

    def _row_prices(self,index,local,integration):
        spots,weights,hats,_=integration;dt=self.times[index]-self.times[index-1]
        prices,derivative=_black(spots[:,None],self.strikes[None,:],(hats@local)[:,None]*dt)
        return weights@prices,(weights[:,None]*derivative).T@hats*dt

    def _prices_jacobian(self,local,integration):
        return self._row_prices(len(self.times)-1,local,integration)

    def terminal_iv_tangent(self):
        _,j=self._prices_jacobian(self.variance[-2],self.integration)
        _,d=_black(self.spot,self.strikes,self.times[-1]*self.iv[-1]**2);vegas=2*self.times[-1]*self.iv[-1]*d
        require(np.isfinite(vegas).all() and np.all(vegas>1e-12),'multi_step_quote','unstable terminal Vega')
        tangent,singular=TwoStepSmileCalibrator._inverse(self,j,vegas)
        _,fine=self._prices_jacobian(self.variance[-2],self.verification_integration)
        error=float(np.max(abs(fine@tangent/vegas[:,None]-np.eye(len(vegas)))))
        require(error<=1e-9,'multi_step_quadrature','terminal tangent refinement failed')
        return tangent,dict(method='multi_step_terminal_iv_tangent_v1',iv_response_residual=error,
            singular_values=singular.tolist(),source_audit=self.source_audit,earlier_variance_rows='fixed')

    def calibrate(self,terminal_iv_shifts):
        try:shifts=np.asarray(terminal_iv_shifts,dtype=float)
        except (ValueError,TypeError) as e:raise BacktestError('multi_step_quote','invalid terminal shifts') from e
        require(shifts.shape==self.iv[-1].shape and np.isfinite(shifts).all(),'multi_step_quote','one finite shift per terminal quote required')
        iv=self.iv[-1]+shifts;require(np.all(iv>0),'multi_step_quote','positive shifted IV required')
        target,d=_black(self.spot,self.strikes,self.times[-1]*iv**2);vegas=2*self.times[-1]*iv*d
        slopes=np.diff(target)/np.diff(self.strikes)
        previous=self.verification_integration[1]@np.maximum(self.verification_integration[0][:,None]-self.strikes,0.)
        require(np.isfinite(target).all() and np.isfinite(vegas).all() and np.all(vegas>1e-12) and np.all(target>=previous)
            and np.all(target<self.spot) and np.all(slopes>=-1) and np.all(slopes<=0) and np.all(np.diff(slopes)>=0),
            'multi_step_arbitrage','terminal quotes violate bounds, convexity or preceding marginal')
        local=self.variance[-2].copy();history=[]
        for iteration in range(self.iterations+1):
            prices,j=self._prices_jacobian(local,self.integration);residual=prices-target
            inverse,singular=TwoStepSmileCalibrator._inverse(self,j,vegas)
            history.append(dict(iteration=iteration,maximum_price_error=float(max(abs(residual))),singular_values=singular.tolist()))
            if max(abs(residual))<=self.tolerance/2:break
            require(iteration<self.iterations,'multi_step_nonconvergence','calibration iteration limit reached')
            step=-inverse@(residual/vegas);accepted=False
            for backtrack in range(30):
                alpha=2.**(-backtrack);trial=local+alpha*step
                if np.all(trial>=self.floor) and np.all(trial<=self.cap):
                    price,_=self._prices_jacobian(trial,self.integration)
                    if np.linalg.norm((price-target)/vegas)<np.linalg.norm(residual/vegas):
                        local=trial;accepted=True;history[-1]['step_fraction']=alpha;break
            require(accepted,'multi_step_nonconvergence','no admissible improving calibration step')
        verified,_=self._prices_jacobian(local,self.verification_integration)
        error=float(max(abs(verified-target)))+self.tail_bound
        require(error<=self.tolerance,'multi_step_quadrature','refined quote prices failed tolerance')
        result=deepcopy(self.payload);values=self.variance.copy();values[-2:]=local
        result['model']['local_variance_grid']['values']=values.ravel().tolist();quotes=self.iv.copy();quotes[-1]=iv
        result['model']['reporting_iv_basis']['implied_volatilities']=quotes.ravel().tolist()
        return result,dict(method='multi_step_terminal_smile_calibration_v1',diagnostic_only=True,earlier_variance_rows='fixed',
            source_request_hash=fingerprint(self.payload),recalibrated_request_hash=fingerprint(result),source_audit=self.source_audit,
            maximum_verified_price_error_bound=error,terminal_iv_shifts=shifts.tolist(),target_prices=target.tolist(),verified_prices=verified.tolist(),
            iteration_history=history,clipping_applied=False,full_market_iv_bucket_acceptance='not_established')
