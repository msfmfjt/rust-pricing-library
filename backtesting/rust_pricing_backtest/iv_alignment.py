"""Audit absolute-maturity/fixed-strike IV moves; never adopt them as P&L."""

from bisect import bisect_left
import math

from .domain import BacktestError, checked_sum, fingerprint, require


POLICY = {"method": "absolute_maturity_fixed_strike_iv_audit_v1",
          "coordinate": "canonical_f_iv", "maturity_matching": "exact_absolute_date",
          "strike_matching": "opening_forward_times_exp_opening_k",
          "interpolation": "linear_iv_in_current_log_forward_moneyness",
          "extrapolation": "none", "expired_nodes": "unavailable_not_zero",
          "risk_weighting": "opening_holdings_and_raw_buckets",
          "coverage": "absolute_instrument_risk_before_netting",
          "application": "diagnostic_only_not_added_to_greek_pnl",
          "blocker": "iv_shock_interpretation_and_delta_convention_not_validated"}
GROUPS = {"targets": "target", "hedges": "option_hedge"}


def _sum(values):
    return checked_sum(values, "nonfinite_iv_alignment", "IV alignment arithmetic overflowed")


def _surface(risks):
    """Check every supplied surface, including eligible unheld instruments."""
    chosen, sources = None, []
    basis = risks["basis"]
    rows, cols = len(basis["maturity_dates"]), len(basis["log_moneyness"])
    for cid, risk in sorted(risks["instruments"].items()):
        if risk.get("reporting_iv") is None:
            continue
        surface = {"iv": risk["reporting_iv"], "forwards": risk["reporting_forwards"]}
        require(risk["basis_id"] == risks["basis_id"] and len(surface["iv"]) == rows*cols
                and len(surface["forwards"]) == rows, "iv_basis_mismatch", cid)
        require(chosen is None or chosen == surface, "iv_surface_mismatch", cid)
        chosen = surface
        sources.append(cid)
    return chosen, sources


def _node(index, maturity, k, row, previous, current, current_basis, at):
    node = {"index": index, "maturity": maturity, "opening_k": k, "opening_forward": None,
            "canonical_strike": None, "opening_iv": None, "current_k": None, "current_iv": None,
            "delta_iv": None, "current_indices": [], "weights": []}
    if previous is not None:
        f = previous["forwards"][row]
        try:
            strike = f * math.exp(k)
        except OverflowError as error:
            raise BacktestError("nonfinite_iv_alignment", "strike reconstruction overflowed") from error
        require(math.isfinite(strike) and strike > 0, "nonfinite_iv_alignment", "invalid reconstructed strike")
        node.update(opening_forward=f, canonical_strike=strike, opening_iv=previous["iv"][index])
    if maturity <= at:
        return {**node, "status": "expired_maturity"}
    if previous is None or current is None:
        return {**node, "status": "missing_opening_surface" if previous is None else "missing_current_surface"}
    if maturity not in current_basis["maturity_dates"]:
        return {**node, "status": "missing_absolute_maturity"}
    i = current_basis["maturity_dates"].index(maturity)
    # Log subtraction avoids overflow in the forward ratio. Exactly equal
    # forwards leave the original floating-point node unchanged.
    moved_k = _sum((k, math.log(f), -math.log(current["forwards"][i])))
    node["current_k"] = moved_k
    axis = current_basis["log_moneyness"]
    if moved_k < axis[0] or moved_k > axis[-1]:
        return {**node, "status": "outside_strike_domain"}
    j = bisect_left(axis, moved_k)
    if axis[j] == moved_k:
        indices, weights = [i*len(axis)+j], [1.]
    else:
        w = (moved_k-axis[j-1])/(axis[j]-axis[j-1])
        indices, weights = [i*len(axis)+j-1, i*len(axis)+j], [1.-w, w]
    value = _sum(current["iv"][n]*w for n, w in zip(indices, weights))
    return {**node, "status": "matched", "current_iv": value, "delta_iv": value-node["opening_iv"],
            "current_indices": indices, "weights": weights}


def audit(current_risks, opening_risks, opening_basis):
    current, sources = _surface(current_risks)
    output = {"at": current_risks["at"], "from_at": None,
              "application": POLICY["application"], "pnl_adopted": False,
              "blockers": [POLICY["blocker"]], "nodes": [], "groups": {},
              "opening_surface_hash": None, "current_surface_hash": fingerprint(current) if current else None,
              "current_surface_contracts": sources, "opening_surface_contracts": [],
              "opening_basis_id": None, "current_basis_id": current_risks["basis_id"]}
    if opening_risks is None:
        return {**output, "status": "no_interval"}
    require(opening_risks["at"] == opening_basis["at"], "iv_alignment_timing", "opening inventory/risk timestamps differ")
    previous, previous_sources = _surface(opening_risks)
    basis = opening_risks["basis"]
    output.update(from_at=opening_risks["at"], opening_basis_id=opening_risks["basis_id"],
                  opening_surface_hash=fingerprint(previous) if previous else None,
                  opening_surface_contracts=previous_sources,
                  risk_method=basis["method"], delta_convention=basis["delta_convention"])
    if basis["coordinate"] != POLICY["coordinate"] or current_risks["basis"]["coordinate"] != POLICY["coordinate"]:
        return {**output, "status": "unsupported_coordinate"}
    cols = len(basis["log_moneyness"])
    nodes = [_node(i*cols+j, maturity, k, i, previous, current, current_risks["basis"], current_risks["at"][:10])
             for i, maturity in enumerate(basis["maturity_dates"]) for j, k in enumerate(basis["log_moneyness"])]
    warning_codes = set()
    for book, label in GROUPS.items():
        rows = []
        for cid, position in sorted(opening_basis["positions"][book].items()):
            units = position["quantity"] * position["multiplier"]
            if units == 0:
                continue
            require(cid in opening_risks["instruments"], "missing_opening_risk", cid)
            risk = opening_risks["instruments"][cid]
            require(risk["basis_id"] == opening_risks["basis_id"] and len(risk["buckets"]) == len(nodes),
                    "iv_basis_mismatch", cid)
            warning_codes.update(risk["warning_codes"])
            rows.append([_sum((units*x,)) for x in risk["buckets"]])
        net = [_sum(r[j] for r in rows) for j in range(len(nodes))]
        gross = [_sum(abs(r[j]) for r in rows) for j in range(len(nodes))]
        matched = [n["status"] == "matched" for n in nodes]
        covered = _sum(x for x, ok in zip(gross, matched) if ok)
        missing = _sum(x for x, ok in zip(gross, matched) if not ok)
        total_risk = _sum((covered, missing))
        dot = _sum(net[j]*n["delta_iv"] for j, n in enumerate(nodes) if matched[j])
        output["groups"][label] = {"opening_raw_buckets": net, "absolute_instrument_buckets": gross,
                                    "covered_bucket_dot": dot, "complete_bucket_dot": dot if missing == 0 else None,
                                    "covered_absolute_risk": covered, "uncovered_absolute_risk": missing,
                                    "absolute_risk_coverage": covered/total_risk if total_risk else None,
                                    "status": "no_opening_risk" if not total_risk else "complete" if missing == 0 else "partial"}
    output["risk_warning_codes"] = sorted(warning_codes)
    if "unvalidated_vega_kt" in warning_codes:
        output["blockers"].append("unvalidated_vega_kt")
    if any(g["uncovered_absolute_risk"] for g in output["groups"].values()):
        output["blockers"].append("incomplete_risk_coverage")
    output.update(nodes=nodes, matched_nodes=sum(n["status"] == "matched" for n in nodes), total_nodes=len(nodes),
                  status="complete" if all(n["status"] == "matched" for n in nodes) else "partial")
    return output
