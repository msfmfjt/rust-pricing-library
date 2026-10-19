"""Lagged spot-Greek explanations, separate from exact cashflow attribution."""

from .domain import checked_sum, fingerprint, require


POLICY = {"method": "opening_holdings_delta_optional_gamma_v1",
          "scope": "spot_delta_optional_gamma", "currency": "nominal_portfolio_currency",
          "risk_timing": "previous_committed_snapshot_after_fills",
          "spot_change": "current_mid_minus_previous_mid",
          "gamma_unit": "currency_per_spot_unit_squared",
          "missing_gamma": "omit_entire_asset_group_gamma_and_report_missing_contracts",
          "vega": "not_computed_without_validated_risk_and_delta_iv_convention",
          "initial_observation": "no_interval_asset_pnl_is_unexplained",
          "residual": "actual_asset_pnl_minus_available_greek_terms",
          "reconciliation": "arithmetic_identity_not_risk_accuracy_acceptance"}
GROUPS = {"targets": "target", "hedges": "option_hedge", "stock": "stock"}
CASH_COMPONENTS = ("dividend", "funding", "stock_borrow", "entry_fees", "trading_fees", "spread_slippage")


def _sum(values):
    return checked_sum(values, "nonfinite_greek_pnl", "Greek explanation arithmetic overflowed")


def explain(snapshot, previous_snapshot, opening_basis, opening_risks, pnl, ledger, previous):
    """Use only prior committed quantities/risks; current quotes supply the shock.

    Missing Greeks stay null. A residual is intentional here and is NOT an
    accounting plug: exact component-to-ledger reconciliation occurs separately.
    """
    interval = previous_snapshot is not None
    require(not interval or (opening_risks["at"] == opening_basis["at"] == previous_snapshot.at.isoformat()),
            "greek_pnl_timing", "opening inventory/risk timestamps differ")
    shock = snapshot.spot.mid - previous_snapshot.spot.mid if interval else None
    groups, warnings = {}, set()
    for book, name in GROUPS.items():
        delta = gamma = delta_pnl = gamma_pnl = available = None
        missing = []
        positions = opening_basis["positions"][book] if interval else {}
        if interval:
            deltas, gammas = [], []
            for cid, position in sorted(positions.items()):
                units = position["quantity"] * position["multiplier"]
                if units == 0:
                    continue
                if book == "stock":
                    deltas.append(units)
                    continue
                require(cid in opening_risks["instruments"], "missing_opening_risk", cid)
                risk = opening_risks["instruments"][cid]
                deltas.append(units * risk["delta"])
                warnings.update(risk["warning_codes"])
                if risk.get("gamma") is None:
                    missing.append(cid)
                else:
                    gammas.append(units * risk["gamma"])
            delta = _sum(deltas)
            gamma = _sum(gammas) if not missing else None
            delta_pnl = _sum((delta * shock,))
            gamma_pnl = _sum((.5 * gamma * shock * shock,)) if gamma is not None else None
            available = _sum(x for x in (delta_pnl, gamma_pnl) if x is not None)
        actual = pnl[name]
        residual = _sum((actual, -(available if available is not None else 0.)))
        groups[name] = {"actual_pnl": actual, "opening_delta": delta, "opening_gamma": gamma,
                        "delta_pnl": delta_pnl, "gamma_pnl": gamma_pnl, "vega_pnl": None,
                        "gamma_missing_contracts": missing,
                        "gamma_status": "no_interval" if not interval else "missing" if missing else "available",
                        "available_greek_pnl": available, "unexplained_pnl": residual}
    cash = {key: pnl[key] for key in CASH_COMPONENTS}
    available = _sum(g["available_greek_pnl"] for g in groups.values()) if interval else None
    residual = _sum(g["unexplained_pnl"] for g in groups.values())
    cash_total = _sum(cash.values())
    total = _sum((available if available is not None else 0., residual, cash_total))
    error = total - ledger["daily_pnl"]
    prior = previous["cumulative"] if previous else {}
    cumulative = {key: _sum((prior.get(key, 0.), value)) for key, value in
                  (("available_greek_pnl", available if available is not None else 0.),
                   ("unexplained_pnl", residual), ("non_asset_pnl", cash_total))}
    cumulative_total = _sum(cumulative.values())
    cumulative_error = cumulative_total - ledger["cumulative_pnl"]
    require(abs(error) <= pnl["tolerance"] and abs(cumulative_error) <= pnl["cumulative_tolerance"],
            "greek_pnl_reconciliation", f"{ledger['at']}: daily={error}, cumulative={cumulative_error}")
    return {"at": ledger["at"], "from_at": previous_snapshot.at.isoformat() if interval else None,
            "elapsed_calendar_days": (snapshot.at.date()-previous_snapshot.at.date()).days if interval else None,
            "spot_change": shock, "risk_snapshot_hash": fingerprint(opening_risks) if interval else None,
            "inventory_snapshot_hash": fingerprint(opening_basis) if interval else None,
            "delta_convention": previous_snapshot.basis.delta_convention if interval else None,
            "risk_warning_codes": sorted(warnings), "groups": groups,
            "available_greek_pnl": available, "unexplained_pnl": residual,
            "non_asset_components": cash, "non_asset_pnl": cash_total,
            "total": total, "ledger_pnl": ledger["daily_pnl"], "reconciliation_error": error,
            "tolerance": pnl["tolerance"], "cumulative": cumulative, "cumulative_total": cumulative_total,
            "cumulative_reconciliation_error": cumulative_error, "cumulative_tolerance": pnl["cumulative_tolerance"],
            "vega_status": "not_computed_risk_and_delta_iv_convention_unvalidated"}
