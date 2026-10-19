"""Frozen named basis constructions accepted for ledger replay."""
import numpy as np

from .domain import fingerprint, require
from .variance_basis import tensor_variance_basis, _fma


POLICIES = ("row_hats", "weighted_time_hats", "coupled_row_hats")
COUPLED_BASIS_HASHES = {
    4: "8d957c3f233540f07698515b1a985d7d2760b0b28d99e9359dddb65c8f4293e7",
    6: "821589f0d43f22d8593f009dafe268cd473a591c4f4fef6630ab807e7c775f33",
}


def _coupled_product(basis, mixing):
    # The accepted matrices used ascending-column fused accumulation. Generic
    # BLAS may split the multiply/add or reorder terms, changing the frozen hash
    # by one ULP. Sparse scalar accumulation preserves those exact matrix bits.
    result = np.zeros((basis.shape[0], mixing.shape[1]))
    for inner in range(basis.shape[1]):
        rows = np.flatnonzero(basis[:, inner])
        columns = np.flatnonzero(mixing[inner])
        for row in rows:
            for column in columns:
                result[row, column] = _fma(float(basis[row, inner]),
                    float(mixing[inner, column]), float(result[row, column]))
    return result


def named_variance_basis(payload, policy):
    """Build the declared basis, preserving its explicit parameter metric."""
    require(policy in POLICIES, "custom_basis_provider_policy", "choose a named accepted variance basis")
    grid = payload["model"]["local_variance_grid"]
    times = np.array(grid["time_nodes"], dtype=float)
    nodes = np.array(grid["log_forward_moneyness_nodes"], dtype=float)
    require(times.ndim == nodes.ndim == 1 and len(times) in (5, 7) and len(nodes) == 33,
            "custom_basis_provider_scope", "four/six intervals and 33 variance nodes required")
    knots = nodes[::2]
    if policy in ("row_hats", "coupled_row_hats"):
        time_knots = (times[:-1] + times[1:]) / 2
        scales = None
    else:
        time_knots = np.array(payload["model"]["reporting_iv_basis"]["maturity_nodes"], dtype=float)
        require(np.array_equal(time_knots, times[2::2]), "custom_basis_provider_scope",
                "reporting maturities must be every second transition")
        scales = np.linspace(.8, 1.3, len(time_knots))[:, None] * (1. + .3 * knots[None, :])
    basis, audit = tensor_variance_basis(times, nodes, time_knots=time_knots,
                                        moneyness_knots=knots, parameter_scales=scales)
    if policy == "coupled_row_hats":
        n, k = len(times)-1, len(knots)
        mixing = np.eye(n*k)
        for row in range(1, n):
            for j in range(k):
                mixing[row*k+j, (row-1)*k+j] = .2 + .1*j/(k-1) + .03*row
                if j+1 < k:
                    mixing[row*k+j, (row-1)*k+j+1] = -.12*(1+j/(k-1))
        basis = _coupled_product(basis, mixing)
        digest = fingerprint(basis.tolist())
        require(digest == COUPLED_BASIS_HASHES[n], "custom_basis_provider_scope",
                "coupled_row_hats requires the exact accepted four/six-interval matrix")
        basis.setflags(write=False)
        audit = dict(audit, method="coupled_row_hats_v1", basis_hash=digest,
                     uncoupled_basis_hash=audit["basis_hash"], parameter_mixing_hash=fingerprint(mixing.tolist()),
                     coupling_rule="identity_plus_next_row_same_and_adjacent_moneyness_v1",
                     coupling_coefficients={"same_node_constant": .2, "same_node_moneyness": .1,
                                            "same_node_time": .03, "adjacent_node": -.12},
                     accepted_matrix_hash=COUPLED_BASIS_HASHES[n])
    return basis, audit
