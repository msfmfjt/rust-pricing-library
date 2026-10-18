"""All reported smiles on a chosen triangular local-variance extension.

Only the interval immediately preceding each reporting maturity is calibrated.
Unreported rows stay fixed; the first reporting time must follow >=2 transitions.
"""
from copy import deepcopy
import json
import math
import numpy as np
from .discrete_smile import _black, TwoStepSmileCalibrator
from .multi_step_smile import MultiStepTerminalSmileCalibrator, source_marginals, _rule
from .domain import require, fingerprint, BacktestError


class MultiStepSmileCalibrator(MultiStepTerminalSmileCalibrator):
    def __init__(self, payload, **kwargs):
        super().__init__(payload, **kwargs)
        require(self.indices[0] >= 2 and len(self.indices)*len(self.quote_nodes) <= 32,
                'multi_step_joint_scope', 'first reporting maturity must follow two transitions; at most 32 quotes')
        self.calibrated_rows = [i-1 for i in self.indices]

    def _prefix(self, payload, market_index):
        """Make a fitted prefix context after earlier rows have changed."""
        index = self.indices[market_index]
        q = deepcopy(payload)
        values = np.asarray(q['model']['local_variance_grid']['values']).reshape(self.variance.shape)
        times = self.times[:index+1]
        integration = source_marginals(self.spot, times, self.nodes, values[:index+1], 2*self.order)[-1]
        prices, _ = self._row_prices(index, values[index-1], integration)
        # Monotone Black inversion supplies the current model quotes, not target quotes.
        lo, hi = np.full(len(prices), 1e-8), np.full(len(prices), 8.)
        require(np.all(prices > np.maximum(self.spot-self.strikes, 0.)) and np.all(prices < self.spot),
                'multi_step_joint_quote', 'current prefix prices have no stable positive implied volatility')
        for _ in range(100):
            mid = (lo+hi)/2
            call, _ = _black(self.spot, self.strikes, times[-1]*mid**2)
            below = call < prices
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        current_iv = (lo+hi)/2
        g = q['model']['local_variance_grid']; b = q['model']['reporting_iv_basis']
        g.update(time_nodes=times.tolist(), values=values[:index+1].ravel().tolist(), shape=[index+1,len(self.nodes)])
        iv = np.asarray(b['implied_volatilities']).reshape(self.iv.shape)[:market_index+1].copy()
        iv[-1] = current_iv
        b.update(maturity_nodes=self.maturities[:market_index+1].tolist(), implied_volatilities=iv.ravel().tolist(), shape=list(iv.shape))
        return MultiStepTerminalSmileCalibrator(q, quadrature_order=self.order,
            price_tolerance=self.tolerance, maximum_iterations=self.iterations, singular_value_cutoff=self.cutoff)

    def calibrate_all_maturities(self, iv_shifts):
        try:
            shifts = np.asarray(iv_shifts, dtype=float)
        except (TypeError, ValueError) as error:
            raise BacktestError('multi_step_joint_quote', 'invalid reported IV shifts') from error
        require(shifts.shape == self.iv.shape and np.isfinite(shifts).all(),
                'multi_step_joint_quote', 'one finite shift for every reported maturity and quote required')
        target = self.iv+shifts
        require(np.all(target > 0), 'multi_step_joint_quote', 'positive shifted IV required')
        result = deepcopy(self.payload); audits = []
        m, k = len(self.nodes), len(self.quote_nodes)
        for j, row in enumerate(self.calibrated_rows):
            prefix = self._prefix(result, j)
            fitted, audit = prefix.calibrate(target[j]-prefix.iv[-1])
            result['model']['local_variance_grid']['values'][row*m:(row+1)*m] = fitted['model']['local_variance_grid']['values'][-2*m:-m]
            result['model']['reporting_iv_basis']['implied_volatilities'][j*k:(j+1)*k] = target[j].tolist()
            audits.append(audit)
        result['model']['local_variance_grid']['values'][-m:] = result['model']['local_variance_grid']['values'][-2*m:-m]
        verified = MultiStepSmileCalibrator(result, quadrature_order=self.order,
            price_tolerance=self.tolerance, maximum_iterations=self.iterations, singular_value_cutoff=self.cutoff)
        frozen = [i for i in range(len(self.times)-1) if i not in self.calibrated_rows]
        require(np.array_equal(verified.variance[frozen], self.variance[frozen]),
                'multi_step_joint_internal', 'unreported variance rows changed')
        return result, dict(method='multi_step_triangular_reported_smile_calibration_v1',
            diagnostic_only=True, source_request_hash=fingerprint(self.payload), recalibrated_request_hash=fingerprint(result),
            calibrated_variance_rows=self.calibrated_rows, fixed_variance_rows=frozen,
            iv_shifts=shifts.tolist(), maturity_audits=audits, clipping_applied=False,
            extension='minimum_norm_variance_increments_in_last_interval_before_each_reporting_maturity',
            full_market_iv_bucket_acceptance='not_established')

    def price_jacobian(self, order):
        """Differentiate the actual transition kernels on fixed integration domains."""
        states = source_marginals(self.spot,self.times,self.nodes,self.variance,order)
        dxs = []
        for row, (_, _, _, audit) in enumerate(states):
            dt = self.times[row+1]-self.times[row]
            width = 4*math.sqrt((np.interp(0,self.nodes,self.variance[0]) if row == 0 else min(self.variance[row]))*dt)
            _, dx = _rule(self.nodes,*audit['log_window'],order,width)
            dxs.append(dx)
        k, m, n = len(self.quote_nodes),len(self.nodes),len(self.indices)
        jacobian = np.zeros((n*k,n*m))
        for maturity, index in enumerate(self.indices):
            source, weights, hats, _ = states[index-2]
            dt = self.times[index]-self.times[index-1]
            value, derivative = _black(source[:,None],self.strikes[None,:],(hats@self.variance[index-1])[:,None]*dt)
            jacobian[maturity*k:(maturity+1)*k,maturity*m:(maturity+1)*m] = (weights[:,None]*derivative).T@hats*dt
            # Backward call continuation; the source weights for each derivative
            # are forward marginals. No recovered product Gamma enters this map.
            for row in range(index-2,0,-1):
                previous, weights, hats, _ = states[row-1]
                following = states[row][0]
                dt = self.times[row+1]-self.times[row]
                variance = (hats@self.variance[row])*dt
                continuation = np.empty((len(previous),k))
                local = np.zeros((k,m))
                for start in range(0,len(previous),128):
                    stop = start+128
                    v = variance[start:stop]
                    z = (np.log(following[:,None]/previous[None,start:stop])+v/2)/np.sqrt(v)
                    kernel = np.exp(-z*z/2)/np.sqrt(2*np.pi*v)*dxs[row][:,None]
                    continuation[start:stop] = kernel.T@value
                    if row in self.calibrated_rows:
                        score = dt*((z*z-1)/(2*v)-z/(2*np.sqrt(v)))
                        local += ((kernel*score).T@value).T@(weights[start:stop,None]*hats[start:stop])
                if row in self.calibrated_rows:
                    j = self.calibrated_rows.index(row)
                    jacobian[maturity*k:(maturity+1)*k,j*m:(j+1)*m] = local
                value = continuation
        require(np.isfinite(jacobian).all(),'multi_step_joint_precision','nonfinite price Jacobian')
        return jacobian

    def all_maturity_iv_tangent(self):
        coarse = self.price_jacobian(self.order)
        fine = self.price_jacobian(2*self.order)
        k, m, n = len(self.quote_nodes),len(self.nodes),len(self.indices)
        vegas = np.concatenate([2*t*iv*_black(self.spot,self.strikes,t*iv**2)[1]
                                for t,iv in zip(self.maturities,self.iv)])
        require(np.all(vegas > 1e-12),'multi_step_joint_quote','unstable market Vega')
        tangent = np.zeros((n*m,n*k)); eye = np.eye(n*k); singular = []
        for j in range(n):
            block = slice(j*k,(j+1)*k); local = slice(j*m,(j+1)*m)
            inverse, values = TwoStepSmileCalibrator._inverse(self,coarse[block,local],vegas[block])
            compensation = eye[block]-(coarse[block,:j*m]@tangent[:j*m])/vegas[block,None]
            tangent[local] = inverse@compensation; singular.append(values.tolist())
        error = float(np.max(abs(fine@tangent/vegas[:,None]-eye)))
        require(error <= 1e-9,'multi_step_joint_precision','all-maturity IV tangent refinement failed')
        return tangent, dict(method='multi_step_triangular_reported_smile_tangent_v1',
            calibrated_variance_rows=self.calibrated_rows, iv_response_residual=error,
            maximum_price_jacobian_refinement=float(np.max(abs(fine-coarse))), singular_values=singular,
            coordinate_order=[dict(maturity=float(t),log_moneyness=float(x)) for t in self.maturities for x in self.quote_nodes],
            extension='last_interval_before_each_reporting_maturity',
            refinement_interpretation='convergence_diagnostic_not_rigorous_total_error_bound')


def evaluate_multi_step_market_gamma(payload, gamma_bin_boundaries, *, subdivisions=64,
        relative_tolerance=1e-6, gamma_relative_standard_error_limit=.25, worker_threads=1):
    import rust_pricing as rp
    c = MultiStepSmileCalibrator(payload)
    tangent, tangent_audit = c.all_maturity_iv_tangent()
    directions = tangent.T.reshape(tangent.shape[1],len(c.calibrated_rows),len(c.nodes)).tolist()
    plan = rp.PricingPlan.compile(rp.PricingRequest.from_json(json.dumps(c.payload,allow_nan=False)),
        worker_threads=worker_threads,reduction_block_size=256)
    audit = plan.evaluate_vega_kt_multi_step_gamma_chain_audit(c.calibrated_rows,gamma_bin_boundaries,
        variance_risk_directions=directions,subdivisions=subdivisions,relative_tolerance=relative_tolerance,
        gamma_relative_standard_error_limit=gamma_relative_standard_error_limit)
    # Covariance is across the same sampling-unit readouts, in coordinate_order.
    # Failed recoveries retain raw native diagnostics but expose no market estimate.
    covariance = None
    mean_covariance = None
    if audit['estimate'] is not None:
        count = tangent.shape[1]
        joint = np.asarray(audit['joint_sample_covariance']).reshape(3*count,3*count)
        covariance = joint[count:2*count,count:2*count]
        mean_covariance = (covariance/audit['independent_sampling_units']).tolist()
        covariance = covariance.tolist()
    return dict(document_kind='multi_step_reported_market_gamma',schema_version=1,diagnostic_only=True,
        status=audit['status'],estimate=audit['estimate'],request_hash=fingerprint(c.payload),
        coordinate_order=tangent_audit['coordinate_order'],iv_unit='absolute_decimal',
        calibration_tangent=tangent.tolist(),tangent_audit=tangent_audit,gamma_audit=audit,
        market_sample_covariance=covariance,market_mean_covariance=mean_covariance,
        full_market_iv_bucket_acceptance='not_established')
