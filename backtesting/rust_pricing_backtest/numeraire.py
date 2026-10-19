"""Explicit common money-market numeraire, independent of strategy funding."""

import math

from .domain import checked_sum, data, fingerprint, require


POLICY = {"method": "explicit_money_market_account_v2", "normalization": "as_supplied",
          "external_cashflows": "each_flow_at_its_snapshot_numeraire", "reconciliation_absolute_tolerance": 1e-10,
          "reconciliation_relative_tolerance": 1e-12,
          "cumulative_tolerance": "sum_daily_tolerances"}


def discounted_ledger(row, previous, observation, *, initial_cash, initial_value, external_cashflows=()):
    """Convert equity first; subtract initial capital at its own account value."""
    if observation is None:
        return {k: None for k in ("numeraire", "numeraire_hash", "discounted_equity",
                                  "discounted_daily_pnl", "discounted_cumulative_pnl",
                                  "discounted_external_net_cashflow", "discounted_cumulative_external_cashflow")}
    equity = row["equity"] / observation.value
    initial_equity = initial_cash / initial_value
    prior_equity = previous["discounted_equity"] if previous else initial_equity
    converted_flows = [f.amount / observation.value for f in external_cashflows]
    require(all(math.isfinite(x) for x in converted_flows), "nonfinite_discounted_pnl", row["at"])
    external = checked_sum(converted_flows, "nonfinite_discounted_pnl", row["at"])
    cumulative_external = checked_sum((previous["discounted_cumulative_external_cashflow"] if previous else 0.0, external),
                                       "nonfinite_discounted_pnl", row["at"])
    daily = equity - prior_equity - external
    cumulative = equity - initial_equity - cumulative_external
    require(all(math.isfinite(x) for x in (equity, initial_equity, external, cumulative_external, daily, cumulative)),
            "nonfinite_discounted_pnl", row["at"])
    return {"numeraire": data(observation), "numeraire_hash": fingerprint(observation),
            "discounted_equity": equity, "discounted_daily_pnl": daily,
            "discounted_cumulative_pnl": cumulative, "discounted_external_net_cashflow": external,
            "discounted_cumulative_external_cashflow": cumulative_external}


def discounted_attribution(row, previous, attribution, previous_attribution, *, initial_cash):
    """Reconcile raw P&L/B_t plus prior-equity conversion carry to discounted P&L.

    Dividing raw daily P&L alone by B_t omits the effect of changing numeraire
    on prior equity. No settlement/trade cashflow is subtracted a second time.
    """
    if row["numeraire"] is None:
        return None
    value = row["numeraire"]["value"]
    prior_value = previous["numeraire"]["value"] if previous else value
    prior_equity = previous["equity"] if previous else initial_cash
    prior_at_current = prior_equity / value
    prior_at_previous = prior_equity / prior_value
    converted_pnl = attribution["total"] / value
    carry = prior_at_current - prior_at_previous
    scale = max(1.0, abs(row["discounted_equity"]), abs(prior_at_current), abs(prior_at_previous))
    tolerance = (attribution["tolerance"] / value + POLICY["reconciliation_absolute_tolerance"]
                 + POLICY["reconciliation_relative_tolerance"] * scale)
    require(all(math.isfinite(x) for x in (converted_pnl, carry, tolerance)),
            "nonfinite_discounted_pnl", row["at"])
    total = math.fsum((converted_pnl, carry))
    prior = previous_attribution["discounted"] if previous_attribution else {}
    cumulative_total = math.fsum((prior.get("cumulative_total", 0.0), total))
    cumulative_tolerance = prior.get("cumulative_tolerance", 0.0) + tolerance
    error = total - row["discounted_daily_pnl"]
    cumulative_error = cumulative_total - row["discounted_cumulative_pnl"]
    require(all(math.isfinite(x) for x in (total, cumulative_total, cumulative_tolerance, error, cumulative_error)),
            "nonfinite_discounted_pnl", row["at"])
    require(abs(error) <= tolerance and abs(cumulative_error) <= cumulative_tolerance,
            "discounted_pnl_reconciliation", f"{row['at']}: daily difference={error}, tolerance={tolerance}; "
            f"cumulative difference={cumulative_error}, tolerance={cumulative_tolerance}")
    return {"pnl_at_current_numeraire": converted_pnl, "numeraire_carry": carry, "total": total,
            "ledger_pnl": row["discounted_daily_pnl"], "reconciliation_error": error, "tolerance": tolerance,
            "cumulative_total": cumulative_total, "ledger_cumulative_pnl": row["discounted_cumulative_pnl"],
            "cumulative_reconciliation_error": cumulative_error, "cumulative_tolerance": cumulative_tolerance}


def discounted_summary(rows, attribution, status):
    if not rows or rows[-1]["numeraire"] is None:
        return None
    first, final = rows[0], rows[-1]
    return {"account_id": final["numeraire"]["account_id"], "currency_id": final["numeraire"]["currency_id"],
            "initial_value": first["numeraire"]["value"], "final_value": final["numeraire"]["value"],
            "observations": len(rows), "cumulative_pnl": final["discounted_cumulative_pnl"],
            "external_net_cashflow": final["discounted_cumulative_external_cashflow"],
            "terminal_hedging_error": final["discounted_cumulative_pnl"]
            if status == "completed" and final["fully_settled"] else None,
            "reconciliation": {"passed": True,
                               "max_abs_daily_difference": max(abs(r["discounted"]["reconciliation_error"]) for r in attribution),
                               "cumulative_difference": attribution[-1]["discounted"]["cumulative_reconciliation_error"]}}
