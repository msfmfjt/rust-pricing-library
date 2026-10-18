"""Three log-Euler intervals: non-Gaussian source and terminal quote calibration.

Terminal-only calibration freezes the earlier transitions. Joint calibration
moves middle and terminal quotes while freezing the first marginal. Neither
operation supplies a first-maturity smile tangent.
"""
from copy import deepcopy
import math
import numpy as np
from .discrete_smile import TwoStepSmileCalibrator, _black, _cdf
from .domain import BacktestError, fingerprint, require


class ThreeStepTerminalSmileCalibrator:
    def __init__(self, payload, *, quadrature_order=32, price_tolerance=1e-10,
                 maximum_iterations=20, singular_value_cutoff=1e-10):
        self._initialize(payload, quadrature_order=quadrature_order,
                         price_tolerance=price_tolerance, maximum_iterations=maximum_iterations,
                         singular_value_cutoff=singular_value_cutoff, verify_base=True)

    def _initialize(self, payload, *, quadrature_order, price_tolerance,
                    maximum_iterations, singular_value_cutoff, verify_base):
        self.payload=deepcopy(payload)
        try:
            g=payload['model']['local_variance_grid'];basis=payload['model']['reporting_iv_basis']
            self.times=np.asarray(g['time_nodes'],dtype=float);self.nodes=np.asarray(g['log_forward_moneyness_nodes'],dtype=float)
            self.quote_nodes=np.asarray(basis['log_forward_moneyness_nodes'],dtype=float)
            require(self.times.shape==(4,) and np.array_equal(basis['maturity_nodes'],self.times[1:])
                and 2<=len(self.nodes)<=65 and 2<=len(self.quote_nodes)<=8,
                'three_step_scope','exactly three intervals, 2..65 local nodes and 2..8 quotes required')
            self.variance=np.asarray(g['values'],dtype=float).reshape(4,len(self.nodes));self.iv=np.asarray(basis['implied_volatilities'],dtype=float).reshape(3,len(self.quote_nodes))
            require(g.get('shape',[4,len(self.nodes)])==[4,len(self.nodes)] and basis.get('shape',[3,len(self.quote_nodes)])==[3,len(self.quote_nodes)],'three_step_shape','declared grid shape mismatch')
            require(np.isfinite(self.times).all() and np.all(np.diff(self.times)>0) and np.isfinite(self.variance).all() and np.isfinite(self.iv).all() and np.all(self.iv>0)
                and np.all(self.variance>=g['floor']) and np.all(self.variance<=g['cap']),'three_step_bounds','finite ordered times and bounded variance/quotes required')
            require(isinstance(quadrature_order,int) and 16<=quadrature_order<=64,'three_step_policy','quadrature order must be an integer in [16,64]')
            prefix=deepcopy(payload);pg=prefix['model']['local_variance_grid'];pb=prefix['model']['reporting_iv_basis']
            pg.update(time_nodes=self.times[:3].tolist(),values=self.variance[:3].ravel().tolist(),shape=[3,len(self.nodes)])
            pb.update(maturity_nodes=self.times[1:3].tolist(),implied_volatilities=self.iv[:2].ravel().tolist(),shape=[2,len(self.quote_nodes)])
            self.prefix=TwoStepSmileCalibrator(prefix,quadrature_order=quadrature_order,price_tolerance=price_tolerance,maximum_iterations=maximum_iterations,singular_value_cutoff=singular_value_cutoff)
        except (KeyError,TypeError,ValueError,IndexError,OverflowError) as error:
            if isinstance(error,BacktestError):raise
            raise BacktestError('three_step_shape','invalid market or grid shape') from error
        self.order=quadrature_order;self.tolerance=price_tolerance;self.iterations=maximum_iterations
        self.spot=self.prefix.spot;self.strikes=self.prefix.strikes;self.dt=self.times[3]-self.times[2]
        self.floor=self.prefix.floor;self.cap=self.prefix.cap
        vmax=self.prefix.v0+max(self.variance[1])*self.prefix.dt
        self.window=(-vmax/2-12*math.sqrt(vmax),vmax/2+12*math.sqrt(vmax))
        require(math.isfinite(self.window[1]) and self.window[1]<=100.,'three_step_scope','source window exceeds precision budget')
        self.tail_bound=self.prefix.tail_bound+self.spot*(1+math.exp(self.window[0]))*math.exp(-72)
        self.probability_tail_bound=2*float(_cdf(-12))+2*math.exp(-72)
        self.integration=self._source(self.order);self.verification_integration=self._source(2*self.order)
        def moments(integration):
            spots,weights,_=integration
            return np.array([sum(weights),weights@spots])
        coarse=moments(self.integration);fine=moments(self.verification_integration)
        require(np.isfinite(fine).all() and np.all(abs(fine-coarse)<=1e-10*np.array([1,self.spot]))
            and np.all(abs(fine-[1,self.spot])<=1e-10*np.array([1,self.spot])+[self.probability_tail_bound,self.tail_bound]),
            'three_step_quadrature','source probability or martingale refinement failed')
        self.source_audit=dict(source='two_log_euler_transition_mixture',quadrature_order=2*self.order,
            source_log_window=list(self.window),source_probability=float(fine[0]),source_first_spot_moment=float(fine[1]),
            source_probability_tail_bound=self.probability_tail_bound,source_spot_tail_bound=self.tail_bound,
            probability_refinement_difference=float(abs(fine[0]-coarse[0])),spot_refinement_difference=float(abs(fine[1]-coarse[1])),renormalized=False)
        prices,_=self._prices_jacobian(self.variance[2],self.verification_integration)
        targets,_=_black(self.spot,self.strikes,self.times[3]*self.iv[2]**2)
        if verify_base:
            require(max(abs(prices-targets))+self.tail_bound<=self.tolerance,'three_step_base_calibration','base terminal quotes do not match three-step prices')

    def _source(self,order):
        s1,p1,_=self.prefix._integration(order);x1=np.log(s1/self.spot)
        variance=np.interp(x1,self.nodes,self.variance[1])*self.prefix.dt;sd=np.sqrt(variance);mu=x1-variance/2
        lo,hi=self.window;cuts=np.r_[lo,self.nodes[(self.nodes>lo)&(self.nodes<hi)],hi];z,w=np.polynomial.legendre.leggauss(order)
        x=((cuts[:-1]+cuts[1:])[:,None]/2+np.diff(cuts)[:,None]/2*z).ravel();dx=(np.diff(cuts)[:,None]/2*w).ravel()
        density=np.empty_like(x)
        for a in range(0,len(x),256):
            normalized=(x[a:a+256,None]-mu)/sd
            density[a:a+256]=(np.exp(-normalized**2/2)/(math.sqrt(2*math.pi)*sd))@p1
        hats=np.array([np.interp(x,self.nodes,unit) for unit in np.eye(len(self.nodes))]).T
        return self.spot*np.exp(x),density*dx,hats

    def _prices_jacobian(self,variances,integration):
        spots,weights,hats=integration;calls,derivative=_black(spots[:,None],self.strikes[None,:],(hats@variances)[:,None]*self.dt)
        return weights@calls,(weights[:,None]*derivative).T@hats*self.dt

    def terminal_iv_tangent(self):
        _,j=self._prices_jacobian(self.variance[2],self.integration)
        _,d=_black(self.spot,self.strikes,self.times[3]*self.iv[2]**2);vegas=d*2*self.times[3]*self.iv[2]
        require(np.isfinite(vegas).all() and np.all(vegas>1e-12),'three_step_quote','unstable terminal quote Vega')
        tangent,singular=self.prefix._inverse(j,vegas)
        _,fine=self._prices_jacobian(self.variance[2],self.verification_integration)
        error=float(np.max(abs(fine@tangent/vegas[:,None]-np.eye(len(vegas)))))
        require(error<=1e-9,'three_step_quadrature','terminal tangent did not converge')
        return tangent,dict(method='three_step_terminal_iv_tangent_v1',iv_response_residual=error,singular_values=singular.tolist(),rank=len(singular),earlier_variance_rows='fixed',source_audit=self.source_audit)

    def calibrate(self,terminal_iv_shifts):
        try:shifts=np.asarray(terminal_iv_shifts,dtype=float)
        except (TypeError,ValueError) as error:raise BacktestError('three_step_quote','invalid quote shifts') from error
        require(shifts.shape==self.iv[2].shape and np.isfinite(shifts).all(),'three_step_quote','one finite shift per terminal quote required')
        iv=self.iv[2]+shifts;require(np.all(iv>0),'three_step_quote','positive shifted IV required')
        target,d=_black(self.spot,self.strikes,self.times[3]*iv**2);vegas=2*self.times[3]*iv*d
        previous,_=_black(self.spot,self.strikes,self.times[2]*self.iv[1]**2);slopes=np.diff(target)/np.diff(self.strikes)
        require(np.isfinite(target).all() and np.isfinite(vegas).all() and np.all(vegas>1e-12)
            and np.all(target>=previous) and np.all(target<self.spot) and np.all(slopes>=-1) and np.all(slopes<=0) and np.all(np.diff(slopes)>=0),
            'three_step_arbitrage','terminal quotes violate bounds, convexity or the fixed preceding marginal')
        local=self.variance[2].copy();history=[]
        for iteration in range(self.iterations+1):
            prices,j=self._prices_jacobian(local,self.integration);residual=prices-target;inverse,singular=self.prefix._inverse(j,vegas)
            history.append(dict(iteration=iteration,maximum_price_error=float(max(abs(residual))),singular_values=singular.tolist()))
            if max(abs(residual))<=self.tolerance/2:break
            require(iteration<self.iterations,'three_step_nonconvergence','quote calibration iteration limit reached')
            step=-inverse@(residual/vegas);accepted=False
            for backtrack in range(30):
                alpha=2.**(-backtrack);trial=local+alpha*step
                if np.all(trial>=self.floor) and np.all(trial<=self.cap):
                    price,_=self._prices_jacobian(trial,self.integration)
                    if np.linalg.norm((price-target)/vegas)<np.linalg.norm(residual/vegas):
                        local=trial;accepted=True;history[-1]['step_fraction']=alpha;break
            require(accepted,'three_step_nonconvergence','no admissible residual-reducing calibration step')
        verified,_=self._prices_jacobian(local,self.verification_integration);error=float(max(abs(verified-target)))+self.tail_bound
        require(error<=self.tolerance,'three_step_quadrature','refined quote prices failed tolerance')
        result=deepcopy(self.payload);result['model']['local_variance_grid']['values']=np.vstack([self.variance[:2],local,local]).ravel().tolist()
        result['model']['reporting_iv_basis']['implied_volatilities']=np.vstack([self.iv[:2],iv]).ravel().tolist()
        return result,dict(method='three_step_terminal_smile_calibration_v1',diagnostic_only=True,earlier_variance_rows='fixed',full_market_iv_bucket_acceptance='not_established',
            source_request_hash=fingerprint(self.payload),recalibrated_request_hash=fingerprint(result),source_audit=self.source_audit,
            terminal_iv_shifts=shifts.tolist(),maximum_verified_price_error_bound=error,price_tolerance=self.tolerance,
            target_prices=target.tolist(),verified_prices=verified.tolist(),iteration_history=history,clipping_applied=False,
            extension='minimum_euclidean_norm_annual_local_variance_newton_increments')

    def calibrate_joint(self, middle_iv_shifts, terminal_iv_shifts):
        """Recalibrate middle quotes, then terminal quotes under the changed source.

        Only this private intermediate context may have unfitted terminal quotes.
        Both market slices pass their ordinary refined price gate before return.
        """
        middle, middle_audit = self.prefix.calibrate(middle_iv_shifts)
        m, k = len(self.nodes), len(self.quote_nodes)
        intermediate = deepcopy(self.payload)
        values = np.array(intermediate['model']['local_variance_grid']['values'])
        values[m:2*m] = middle['model']['local_variance_grid']['values'][m:2*m]
        intermediate['model']['local_variance_grid']['values'] = values.tolist()
        intermediate['model']['reporting_iv_basis']['implied_volatilities'][k:2*k] = (
            middle['model']['reporting_iv_basis']['implied_volatilities'][k:2*k])
        working = object.__new__(type(self))
        working._initialize(intermediate, quadrature_order=self.order,
                            price_tolerance=self.tolerance, maximum_iterations=self.iterations,
                            singular_value_cutoff=self.prefix.cutoff, verify_base=False)
        result, terminal_audit = working.calibrate(terminal_iv_shifts)
        return result, dict(method='three_step_middle_terminal_smile_calibration_v1',
                            diagnostic_only=True, first_variance_row='fixed',
                            full_market_iv_bucket_acceptance='not_established',
                            source_request_hash=fingerprint(self.payload),
                            intermediate_request_hash=fingerprint(intermediate),
                            recalibrated_request_hash=fingerprint(result),
                            middle_calibration=middle_audit, terminal_calibration=terminal_audit,
                            clipping_applied=False)

    def _middle_price_jacobian(self, order):
        """Terminal call price derivative through the conditional source density.

        This score differentiates the finite log-Euler transition, not a Gaussian
        surrogate for the combined source. It never uses recovered Gamma.
        """
        s1, p1, h1 = self.prefix._integration(order)
        x1 = np.log(s1/self.spot)
        variance = np.interp(x1, self.nodes, self.variance[1])*self.prefix.dt
        sd = np.sqrt(variance)
        lo, hi = self.window
        cuts = np.r_[lo, self.nodes[(self.nodes>lo)&(self.nodes<hi)], hi]
        z, w = np.polynomial.legendre.leggauss(order)
        x2 = ((cuts[:-1]+cuts[1:])[:,None]/2+np.diff(cuts)[:,None]*z/2).ravel()
        dx = (np.diff(cuts)[:,None]*w/2).ravel()
        derivative = np.empty((len(x2),len(self.nodes)))
        for a in range(0,len(x2),256):
            normal = (x2[a:a+256,None]-x1+variance/2)/sd
            density = np.exp(-normal**2/2)/(math.sqrt(2*math.pi)*sd)
            score = self.prefix.dt*((normal**2-1)/(2*variance)-normal/(2*sd))
            derivative[a:a+256] = (density*score*p1)@h1
        v3 = np.interp(x2,self.nodes,self.variance[2])*self.dt
        calls, _ = _black(self.spot*np.exp(x2[:,None]),self.strikes[None,:],v3[:,None])
        return calls.T@(dx[:,None]*derivative)

    def joint_iv_tangent(self):
        """2m by 2k tangent: middle quote IVs followed by terminal quote IVs.

        The first annual-variance row is frozen. A middle quote bump also moves
        terminal variance to keep all terminal quotes fixed.
        """
        middle, middle_audit = self.prefix.terminal_iv_tangent()
        terminal, terminal_audit = self.terminal_iv_tangent()
        coarse = self._middle_price_jacobian(self.order)
        fine = self._middle_price_jacobian(2*self.order)
        _, d3 = _black(self.spot,self.strikes,self.times[3]*self.iv[2]**2)
        vegas3 = d3*2*self.times[3]*self.iv[2]
        _, d2 = _black(self.spot,self.strikes,self.times[2]*self.iv[1]**2)
        vegas2 = d2*2*self.times[2]*self.iv[1]
        vmin = min(self.variance[1])*self.prefix.dt
        vmax = max(self.variance[1])*self.prefix.dt
        factor = self.prefix.dt*(1/(2*vmin)+1/(4*math.sqrt(vmin)))
        source_tail = self.spot*(1+math.exp(self.window[0]))*math.exp(-72)
        # |score| <= factor*(Z^2+1); stock-measure Cauchy-Schwarz
        # gives E[(Z^2+1)^2] = vmax^2+8*vmax+6 for the shifted normal.
        tail = factor*((vmax+2)*self.prefix.tail_bound + math.sqrt(
            self.spot*(vmax*vmax+8*vmax+6)*source_tail))
        refinement = float(np.max((abs((fine-coarse)@middle)
                                  +tail*np.sum(abs(middle),axis=0))/vegas3[:,None]))
        require(math.isfinite(refinement) and refinement<=1e-9,
                'three_step_quadrature','cross-maturity price Jacobian did not converge')
        compensation = -(coarse@middle)/vegas3[:,None]
        tangent = np.block([[middle,np.zeros_like(middle)],
                            [terminal@compensation,terminal]])
        _, j2 = self.prefix._prices_jacobian(self.variance[1],self.prefix.verification_integration)
        _, j3 = self._prices_jacobian(self.variance[2],self.verification_integration)
        response = np.vstack([j2@tangent[:len(self.nodes)]/vegas2[:,None],
            (fine@tangent[:len(self.nodes)]+j3@tangent[len(self.nodes):])/vegas3[:,None]])
        residual = float(np.max(abs(response-np.eye(2*len(self.quote_nodes)))))
        require(np.isfinite(tangent).all() and residual<=1e-9,
                'three_step_quadrature','joint market tangent did not converge')
        return tangent, dict(method='three_step_middle_terminal_iv_tangent_v1',
            first_variance_row='fixed', coordinate_order=['middle_quote_ivs','terminal_quote_ivs'],
            annual_variance_row_order=[1,2], iv_response_residual=residual,
            cross_jacobian_refinement_with_tail=refinement,
            cross_jacobian_entry_tail_bound=tail,
            terminal_compensation_quote_weights=compensation.tolist(),
            terminal_conservation_basis=terminal.T.tolist(),
            middle_tangent_audit=middle_audit,terminal_tangent_audit=terminal_audit)

    def calibrate_all_maturities(self, first_iv_shift, middle_iv_shifts, terminal_iv_shifts):
        """Move one parallel first IV and both subsequent quote slices.

        Each slice is calibrated in chronological order. Individual first-
        maturity smile quotes remain unreachable in a single first transition.
        """
        prefix, prefix_audit = self.prefix.calibrate_joint(first_iv_shift, middle_iv_shifts)
        m, k = len(self.nodes), len(self.quote_nodes)
        intermediate = deepcopy(self.payload)
        intermediate['model']['local_variance_grid']['values'][:2*m] = (
            prefix['model']['local_variance_grid']['values'][:2*m])
        intermediate['model']['reporting_iv_basis']['implied_volatilities'][:2*k] = (
            prefix['model']['reporting_iv_basis']['implied_volatilities'][:2*k])
        working = object.__new__(type(self))
        working._initialize(intermediate, quadrature_order=self.order,
                            price_tolerance=self.tolerance, maximum_iterations=self.iterations,
                            singular_value_cutoff=self.prefix.cutoff, verify_base=False)
        result, terminal_audit = working.calibrate(terminal_iv_shifts)
        return result, dict(method='three_step_all_reachable_maturities_calibration_v1',
            diagnostic_only=True, source_request_hash=fingerprint(self.payload),
            intermediate_request_hash=fingerprint(intermediate), recalibrated_request_hash=fingerprint(result),
            first_parallel_iv_shift=float(first_iv_shift), prefix_calibration=prefix_audit,
            terminal_calibration=terminal_audit, first_maturity_individual_quote_shifts='unsupported',
            full_market_iv_bucket_acceptance='not_established', clipping_applied=False)

    def _first_price_response(self, order):
        """Terminal vanilla derivative through the first Gaussian density."""
        s1, p1, _ = self.prefix._integration(order)
        x1 = np.log(s1/self.spot)
        variance = np.interp(x1,self.nodes,self.variance[1])*self.prefix.dt
        sd2 = np.sqrt(variance)
        v0 = self.prefix.v0
        z1 = (x1+v0/2)/math.sqrt(v0)
        score = self.times[1]*((z1*z1-1)/(2*v0)-z1/(2*math.sqrt(v0)))
        lo, hi = self.window
        cuts = np.r_[lo,self.nodes[(self.nodes>lo)&(self.nodes<hi)],hi]
        z, w = np.polynomial.legendre.leggauss(order)
        x2 = ((cuts[:-1]+cuts[1:])[:,None]/2+np.diff(cuts)[:,None]*z/2).ravel()
        dx = (np.diff(cuts)[:,None]*w/2).ravel()
        derivative = np.empty_like(x2)
        for a in range(0,len(x2),256):
            normal = (x2[a:a+256,None]-x1+variance/2)/sd2
            derivative[a:a+256] = (np.exp(-normal*normal/2)/(math.sqrt(2*math.pi)*sd2))@(p1*score)
        v3 = np.interp(x2,self.nodes,self.variance[2])*self.dt
        calls, _ = _black(self.spot*np.exp(x2[:,None]),self.strikes[None,:],v3[:,None])
        return (dx*derivative)@calls

    def all_maturity_iv_tangent(self):
        """(1+2m) by (1+2k): first parallel IV, middle IVs, terminal IVs."""
        later, later_audit = self.joint_iv_tangent()
        prefix, prefix_audit = self.prefix.joint_iv_tangent()
        first = [self._first_price_response(n) for n in [self.order,2*self.order]]
        middle = [self._middle_price_jacobian(n) for n in [self.order,2*self.order]]
        m, k = len(self.nodes), len(self.quote_nodes)
        _, d = _black(self.spot,self.strikes,self.times[3]*self.iv[2]**2)
        vegas = 2*self.times[3]*self.iv[2]*d
        v0 = self.prefix.v0; sd = math.sqrt(v0)
        prob = float(_cdf(-12-sd)+_cdf(-12+sd))
        moment = (v0+2)*prob+(12+sd)*(math.exp(-.5*(12-sd)**2)+math.exp(-.5*(12+sd)**2))/math.sqrt(2*math.pi)
        source_tail = self.spot*(1+math.exp(self.window[0]))*math.exp(-72)
        tail = self.times[1]*(1/(2*v0)+1/(4*sd))*(self.spot*moment + math.sqrt(
            self.spot*(v0*v0+8*v0+6)*source_tail))
        a, b = prefix[0,0], prefix[1:,0]
        responses = [first[i]*a+middle[i]@b for i in range(2)]
        allowance = abs(a)*tail + later_audit['cross_jacobian_entry_tail_bound']*sum(abs(b))
        refinement = float(max((abs(responses[1]-responses[0])+allowance)/vegas))
        require(np.isfinite(responses).all() and math.isfinite(refinement) and refinement<=1e-9,
                'three_step_quadrature','first-maturity terminal compensation did not converge')
        terminal_basis = later[m:,k:]
        first_weights = -responses[0]/vegas
        tangent = np.zeros((1+2*m,1+2*k))
        tangent[0,0]=a; tangent[1:1+m,0]=b
        tangent[1+m:,0]=terminal_basis@first_weights; tangent[1:,1:]=later
        _, j3 = self._prices_jacobian(self.variance[2],self.verification_integration)
        residual = float(max(abs(responses[1]+j3@tangent[1+m:,0])/vegas))
        require(np.isfinite(tangent).all() and residual<=1e-9,
                'three_step_quadrature','all-maturity market tangent failed verification')
        return tangent, dict(method='three_step_all_reachable_iv_tangent_v1',
            coordinate_order=['first_maturity_parallel_iv']+[f'middle_quote_{i}' for i in range(k)]+[f'terminal_quote_{i}' for i in range(k)],
            variance_coordinate_order='uniform_first_annual_variance_then_middle_nodes_then_terminal_nodes',
            first_iv_to_annual_variance=a, first_terminal_compensation_quote_weights=first_weights.tolist(),
            terminal_conservation_basis=terminal_basis.T.tolist(), first_terminal_iv_response_residual=residual,
            first_response_refinement_with_tail=refinement, first_response_tail_bound=tail,
            prefix_tangent_audit=prefix_audit, later_tangent_audit=later_audit,
            first_maturity_individual_quote_shifts='unsupported',full_market_iv_bucket_acceptance='not_established')
