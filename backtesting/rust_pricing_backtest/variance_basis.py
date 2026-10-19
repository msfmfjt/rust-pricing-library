"""Explicit tensor hat bases for diagnostic variance-extension calibration."""
from fractions import Fraction
import math
import numpy as np
from .domain import BacktestError, fingerprint, require


def _portable_fma(a, b, c):
    """One binary64 rounding, also on Python 3.12 without math.fma."""
    return float(Fraction(a) * Fraction(b) + Fraction(c))


_fma = getattr(math, "fma", _portable_fma)


def _hat_interpolation(points, knots):
    # np.interp uses either fused or separate multiply/add depending on its
    # build. Preserve the accepted fused slope*offset+value convention explicitly.
    values = np.zeros((len(points), len(knots)))
    for row, point in enumerate(points):
        if point <= knots[0]:
            values[row, 0] = 1.
        elif point >= knots[-1]:
            values[row, -1] = 1.
        else:
            left = int(np.searchsorted(knots, point, side='right')) - 1
            offset = float(point-knots[left])
            width = float(knots[left+1]-knots[left])
            values[row, left] = _fma(-1./width, offset, 1.)
            values[row, left+1] = _fma(1./width, offset, 0.)
    return values


def tensor_variance_basis(time_nodes, variance_nodes, *, time_knots, moneyness_knots,
                          parameter_scales=None):
    """Return (transition-cell x parameter basis, audit).

    Hats interpolate transition midpoints and local log-moneyness nodes with
    constant endpoints. Parameters are ordered by time knot then moneyness knot.
    A scale multiplies a column; it therefore changes the minimum-parameter-norm
    calibration metric. It is never silently normalized or orthogonalized.
    """
    def vector(value, name, minimum, maximum):
        try:a=np.array(value,dtype=float,copy=True)
        except (TypeError,ValueError,OverflowError) as error:
            raise BacktestError('variance_basis_shape','invalid '+name) from error
        require(a.ndim==1 and minimum<=len(a)<=maximum and np.isfinite(a).all()
                and np.all(np.diff(a)>0),'variance_basis_shape','finite ordered '+name+' required')
        return a
    times=vector(time_nodes,'time nodes',3,17)
    nodes=vector(variance_nodes,'variance nodes',2,33)
    tk=vector(time_knots,'time knots',1,len(times)-1)
    kk=vector(moneyness_knots,'moneyness knots',1,len(nodes))
    require(times[0]==0 and tk[0]>=0 and tk[-1]<=times[-1],
            'variance_basis_shape','time knots must lie inside the simulation horizon')
    require(kk[0]>=nodes[0] and kk[-1]<=nodes[-1],
            'variance_basis_shape','moneyness knots must lie inside the variance grid')
    tb=_hat_interpolation((times[:-1]+times[1:])/2, tk)
    kb=_hat_interpolation(nodes, kk)
    require(np.linalg.matrix_rank(tb)==len(tk) and np.linalg.matrix_rank(kb)==len(kk),
            'variance_basis_rank','interpolation knots produce dependent or unsupported parameters')
    shape=(len(tk),len(kk))
    try:scales=np.ones(shape) if parameter_scales is None else np.array(parameter_scales,dtype=float,copy=True)
    except (TypeError,ValueError,OverflowError) as error:
        raise BacktestError('variance_basis_shape','invalid parameter scales') from error
    require(scales.shape==shape and np.isfinite(scales).all() and np.all(scales>0),
            'variance_basis_scale','one finite positive scale per time/moneyness parameter required')
    with np.errstate(over='ignore',invalid='ignore'):
        basis=np.kron(tb,kb)*scales.ravel()
    require(np.isfinite(basis).all(),'variance_basis_scale','nonfinite scaled basis')
    basis.setflags(write=False)
    audit=dict(method='tensor_linear_hats_constant_endpoints_v1',time_coordinates='transition_midpoints',
        variance_coordinate_order='transition_row_then_log_moneyness',parameter_order='time_knot_then_log_moneyness_knot',
        time_knots=tk.tolist(),moneyness_knots=kk.tolist(),parameter_scales=scales.tolist(),
        parameter_metric='euclidean_parameter_norm_without_column_normalization',
        basis_shape=list(basis.shape),basis_hash=fingerprint(basis.tolist()),
        diagnostic_only=True,full_market_iv_bucket_acceptance='not_established')
    return basis,audit
