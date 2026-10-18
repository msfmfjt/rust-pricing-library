"""Simultaneous quote calibration in an explicit linear variance extension.

The extension fixes the parameter metric: each Newton step has minimum
Euclidean parameter norm. This is not a claim of a globally nearest fitted
surface or a unique market-implied exotic risk. All transition rows, including
row zero, may move; the unused terminal grid row stays fixed.
"""
from copy import deepcopy
import json
import math
import numpy as np
from .discrete_smile import _black
from .domain import BacktestError, fingerprint, require
from .multi_step_smile import MultiStepTerminalSmileCalibrator, source_marginals, _rule


class MultiStepExtensionCalibrator(MultiStepTerminalSmileCalibrator):
    def __init__(self, payload, *, extension="all_rows", variance_basis=None, **kwargs):
        super().__init__(payload, **kwargs)
        self.intervals=len(self.times)-1
        self.calibrated_rows=list(range(self.intervals))
        require(self.indices[0]>=2 and self.iv.size<=32, 'extension_scope',
                'first reporting maturity must follow two transitions; at most 32 quotes')
        require(extension in ('all_rows','linear_time','custom'), 'extension_policy', 'unknown variance extension')
        m=len(self.nodes);n=self.intervals
        require((extension=='custom') == (variance_basis is not None), 'extension_policy',
                'custom extension requires an explicit basis; presets do not accept one')
        if extension=='all_rows':basis=np.eye(n*m)
        elif extension=='linear_time':
            midpoints=(self.times[:-1]+self.times[1:])/2
            time_basis=np.array([np.interp(midpoints,self.maturities,e) for e in np.eye(len(self.maturities))]).T
            basis=np.kron(time_basis,np.eye(m))
        else:
            try:basis=np.array(variance_basis,dtype=float,copy=True)
            except (TypeError,ValueError) as error:raise BacktestError('extension_shape','invalid variance basis') from error
        require(basis.ndim==2 and basis.shape[0]==n*m and 1<=basis.shape[1]<=n*m
                and np.isfinite(basis).all(), 'extension_shape', 'finite (transition cells, parameters) basis required')
        self.basis=basis.copy();self.basis.setflags(write=False)
        self.extension=extension
        self.extension_audit=dict(method='minimum_parameter_norm_newton_increments_v1',extension=extension,
            basis_shape=list(basis.shape),basis_hash=fingerprint(basis.tolist()),
            variance_coordinate_order='transition_row_then_log_moneyness',unused_terminal_row='fixed')

    def _surface_prices(self, rows, order, *, jacobian=False):
        """Fixed-domain lognormal score derivatives, including the initial law."""
        states=source_marginals(self.spot,self.times,self.nodes,rows,order)
        n,m,k=self.intervals,len(self.nodes),len(self.quote_nodes)
        prices=np.zeros(self.iv.size);j=np.zeros((self.iv.size,n*m)) if jacobian else None
        dxs=[]
        if jacobian:
            for row,(_,_,_,audit) in enumerate(states):
                dt=self.times[row+1]-self.times[row]
                width=4*math.sqrt((np.interp(0,self.nodes,rows[0]) if row==0 else min(rows[row]))*dt)
                dxs.append(_rule(self.nodes,*audit['log_window'],order,width)[1])
        tails=[]
        for maturity,index in enumerate(self.indices):
            block=slice(maturity*k,(maturity+1)*k)
            source,weights,hats,_=states[index-2]
            dt=self.times[index]-self.times[index-1]
            value,derivative=_black(source[:,None],self.strikes[None,:],(hats@rows[index-1])[:,None]*dt)
            prices[block]=weights@value;tails.append(states[index-2][3]['spot_tail_bound'])
            if not jacobian:continue
            j[block,(index-1)*m:index*m]=(weights[:,None]*derivative).T@hats*dt
            for row in range(index-2,0,-1):
                previous,weights,hats,_=states[row-1];following=states[row][0]
                dt=self.times[row+1]-self.times[row];variance=(hats@rows[row])*dt
                continuation=np.empty((len(previous),k));local=np.zeros((k,m))
                for start in range(0,len(previous),128):
                    stop=start+128;v=variance[start:stop]
                    z=(np.log(following[:,None]/previous[None,start:stop])+v/2)/np.sqrt(v)
                    kernel=np.exp(-z*z/2)/np.sqrt(2*np.pi*v)*dxs[row][:,None]
                    continuation[start:stop]=kernel.T@value
                    score=dt*((z*z-1)/(2*v)-z/(2*np.sqrt(v)))
                    local+=((kernel*score).T@value).T@(weights[start:stop,None]*hats[start:stop])
                j[block,row*m:(row+1)*m]=local;value=continuation
            source,weights,_,_=states[0];dt=self.times[1];v=np.interp(0.,self.nodes,rows[0])*dt
            z=(np.log(source/self.spot)+v/2)/math.sqrt(v)
            score=dt*((z*z-1)/(2*v)-z/(2*math.sqrt(v)))
            initial_hats=np.array([np.interp(0.,self.nodes,e) for e in np.eye(m)])
            j[block,:m]=np.outer((weights*score)@value,initial_hats)
        require(np.isfinite(prices).all() and (j is None or np.isfinite(j).all()),
                'extension_precision','nonfinite prices or Jacobian')
        return prices,j,max(tails)

    def _vegas(self, iv):
        vegas=np.concatenate([2*t*v*_black(self.spot,self.strikes,t*v*v)[1] for t,v in zip(self.maturities,iv)])
        require(np.isfinite(vegas).all() and np.all(vegas>1e-12),'extension_quote','unstable market Vega')
        return vegas

    def _inverse(self, jacobian, vegas):
        with np.errstate(over="ignore",invalid="ignore"):
            reduced=jacobian@self.basis
        require(np.isfinite(reduced).all(),"extension_precision","nonfinite parameter Jacobian")
        require(reduced.shape[1]>=reduced.shape[0],'extension_rank','basis cannot span all market quotes')
        with np.errstate(over="ignore",invalid="ignore"):
            scaled=reduced/vegas[:,None]
        require(np.isfinite(scaled).all(),"extension_precision","nonfinite scaled parameter Jacobian")
        try:u,s,vh=np.linalg.svd(scaled,full_matrices=False)
        except np.linalg.LinAlgError as error:raise BacktestError("extension_precision","parameter Jacobian decomposition failed") from error
        require(np.count_nonzero(s>s[0]*self.cutoff)==len(vegas),'extension_rank',
                'variance extension does not span all market quotes at the declared cutoff')
        return (vh.T/s)@u.T,s

    def all_maturity_iv_tangent(self):
        _,coarse,_=self._surface_prices(self.variance,self.order,jacobian=True)
        _,fine,_=self._surface_prices(self.variance,2*self.order,jacobian=True)
        vegas=self._vegas(self.iv);inverse,singular=self._inverse(coarse,vegas)
        tangent=self.basis@inverse
        error=float(np.max(abs(fine@tangent/vegas[:,None]-np.eye(len(vegas)))))
        require(error<=1e-9,'extension_precision','full-row IV tangent refinement failed')
        return tangent,dict(self.extension_audit,calibrated_variance_rows=self.calibrated_rows,
            iv_response_residual=error,maximum_price_jacobian_refinement=float(np.max(abs(fine-coarse))),
            singular_values=singular.tolist(),
            coordinate_order=[dict(maturity=float(t),log_moneyness=float(x)) for t in self.maturities for x in self.quote_nodes],
            refinement_interpretation='convergence_diagnostic_not_rigorous_total_error_bound')

    def calibrate_all_maturities(self, iv_shifts):
        try:shifts=np.asarray(iv_shifts,dtype=float)
        except (TypeError,ValueError) as error:raise BacktestError('extension_quote','invalid IV shifts') from error
        require(shifts.shape==self.iv.shape and np.isfinite(shifts).all(), 'extension_quote', 'one finite shift per reported quote required')
        target_iv=self.iv+shifts
        require(np.all(target_iv>0),'extension_quote','positive shifted IV required')
        target=np.array([_black(self.spot,self.strikes,t*v*v)[0] for t,v in zip(self.maturities,target_iv)])
        slopes=np.diff(target,axis=1)/np.diff(self.strikes)
        require(np.all(target>=np.maximum(self.spot-self.strikes,0.)) and np.all(target<self.spot)
            and np.all(slopes>=-1) and np.all(slopes<=0) and np.all(np.diff(slopes,axis=1)>=0)
            and np.all(np.diff(target,axis=0)>=0), 'extension_arbitrage', 'quote bounds, strike convexity or calendar ordering failed')
        vegas=self._vegas(target_iv);target=target.ravel();parameters=np.zeros(self.basis.shape[1]);history=[]
        rows=self.variance.copy()
        for iteration in range(self.iterations+1):
            prices,j,_=self._surface_prices(rows,self.order,jacobian=True);residual=prices-target
            inverse,singular=self._inverse(j,vegas)
            history.append(dict(iteration=iteration,maximum_price_error=float(max(abs(residual))),singular_values=singular.tolist()))
            if max(abs(residual))<=self.tolerance/2:break
            require(iteration<self.iterations,'extension_nonconvergence','calibration iteration limit reached')
            step=-inverse@(residual/vegas);accepted=False
            for backtrack in range(30):
                alpha=2.**(-backtrack);candidate=parameters+alpha*step;trial=self.variance.copy()
                trial[:-1]+= (self.basis@candidate).reshape(self.intervals,len(self.nodes))
                if np.all(trial>=self.floor) and np.all(trial<=self.cap):
                    trial_prices,_,_=self._surface_prices(trial,self.order)
                    if np.linalg.norm((trial_prices-target)/vegas)<np.linalg.norm(residual/vegas):
                        parameters,rows=candidate,trial;accepted=True;history[-1]['step_fraction']=alpha;break
            require(accepted,'extension_nonconvergence','no admissible improving calibration step')
        verified,_,tail=self._surface_prices(rows,2*self.order)
        error=float(max(abs(verified-target)))+tail
        require(error<=self.tolerance,'extension_precision','refined quote prices failed tolerance')
        result=deepcopy(self.payload);result['model']['local_variance_grid']['values']=rows.ravel().tolist()
        result['model']['reporting_iv_basis']['implied_volatilities']=target_iv.ravel().tolist()
        return result,dict(self.extension_audit,diagnostic_only=True,source_request_hash=fingerprint(self.payload),
            recalibrated_request_hash=fingerprint(result),iv_shifts=shifts.tolist(),parameter_increments=parameters.tolist(),
            maximum_verified_price_error_bound=error,target_prices=target.tolist(),verified_prices=verified.tolist(),
            iteration_history=history,clipping_applied=False,full_market_iv_bucket_acceptance='not_established')


def evaluate_extended_market_gamma(payload, gamma_bin_boundaries, *, extension='all_rows',variance_basis=None,
        subdivisions=512,relative_tolerance=1e-6,gamma_relative_standard_error_limit=.25,worker_threads=1):
    import rust_pricing as rp
    c=MultiStepExtensionCalibrator(payload,extension=extension,variance_basis=variance_basis)
    tangent,ta=c.all_maturity_iv_tangent();count=c.iv.size
    directions=tangent.T.reshape(count,c.intervals,len(c.nodes)).tolist()
    plan=rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(c.payload,allow_nan=False)),
        worker_threads=worker_threads,reduction_block_size=256)
    audit=plan.evaluate_vega_kt_multi_step_gamma_chain_audit(c.calibrated_rows,gamma_bin_boundaries,
        variance_risk_directions=directions,subdivisions=subdivisions,relative_tolerance=relative_tolerance,
        gamma_relative_standard_error_limit=gamma_relative_standard_error_limit)
    covariance=mean_covariance=None
    if audit['estimate'] is not None:
        joint=np.asarray(audit['joint_sample_covariance']).reshape(3*count,3*count)
        cov=joint[count:2*count,count:2*count];covariance=cov.tolist()
        mean_covariance=(cov/audit['independent_sampling_units']).tolist()
    return dict(document_kind='extended_multi_step_market_gamma',schema_version=1,diagnostic_only=True,
        status=audit['status'],estimate=audit['estimate'],request_hash=fingerprint(c.payload),
        extension_audit=c.extension_audit,coordinate_order=ta['coordinate_order'],iv_unit='absolute_decimal',
        calibration_tangent=tangent.tolist(),tangent_audit=ta,gamma_audit=audit,
        market_sample_covariance=covariance,market_mean_covariance=mean_covariance,
        full_market_iv_bucket_acceptance='not_established')
