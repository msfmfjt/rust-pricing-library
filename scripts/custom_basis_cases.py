"""Declared custom bases; neither is silently substituted for a preset."""
import numpy as np
from rust_pricing_backtest.variance_basis import tensor_variance_basis

KINDS=('row_hats','weighted_time_hats')


def basis_for(payload,kind):
    g=payload['model']['local_variance_grid'];times=np.array(g['time_nodes'])
    nodes=np.array(g['log_forward_moneyness_nodes']);knots=nodes[::2]
    if kind=='row_hats':tk=(times[:-1]+times[1:])/2;scales=None
    elif kind=='weighted_time_hats':
        tk=np.array(payload['model']['reporting_iv_basis']['maturity_nodes'])
        scales=np.linspace(.8,1.3,len(tk))[:,None]*(1.+.3*knots[None,:])
    else:raise ValueError('unknown declared basis')
    return tensor_variance_basis(times,nodes,time_knots=tk,moneyness_knots=knots,parameter_scales=scales)
