"""Cashflow/valuation P&L attribution; no Greek approximation or balancing plug."""

import math

from .domain import require


COMPONENTS = ("target", "option_hedge", "stock", "dividend", "funding",
              "stock_borrow", "entry_fees", "trading_fees", "spread_slippage")
POLICY = {"method": "cashflow_mark_changes_v2", "external_capital": "excluded_from_pnl", "absolute_tolerance": 1e-10,
          "relative_tolerance": 1e-12, "scale": "gross_balances_and_daily_flows",
          "cumulative_tolerance": "sum_daily_tolerances"}
_CASHFLOW_KINDS = {"target_entry", "targets_settlement", "hedges_settlement",
                   "hedge_trade", "funding", "stock_borrow", "dividend", "external_capital"}


def attribute_pnl(current, previous, fills, cashflows, *, initial_cash,
                  entry_premium=0.0, entry_fees=0.0, previous_attribution=None):
    """Attribute one committed candidate row. Entry premium is signed cash.

    Premium and entry fees come from positions, trade principal/cost from fills,
    carry/payment amounts from cashflows, and marks from the valuation ledger.
    Their sum is checked against equity change, never derived from its residual.
    All returned components are signed currency contributions, including costs.
    """
    require(all(f["kind"] in _CASHFLOW_KINDS for f in cashflows),
            "pnl_cashflow_kind", "unclassified cashflow")
    previous = previous or {"cash": initial_cash}

    def change(field):
        return current.get(field, 0.0) - previous.get(field, 0.0)

    def claims(row, group):
        return math.fsum(r["present_value"] for r in row.get("receivables", {}).values()
                         if r["group"] == group)

    def cash(kind):
        return math.fsum(f["amount"] for f in cashflows if f["kind"] == kind)

    def trade_cash(spot):
        return -math.fsum(f["quantity"] * f["multiplier"] * f["mid"] for f in fills
                          if (f["contract_id"] == "__spot__") == spot)

    details = {"target_value_change": change("target_value"),
               "target_claim_change": claims(current, "targets") - claims(previous, "targets"),
               "target_settlement_cash": cash("targets_settlement"),
               "target_entry_premium": entry_premium,
               "hedge_value_change": change("hedge_value"),
               "hedge_claim_change": claims(current, "hedges") - claims(previous, "hedges"),
               "hedge_settlement_cash": cash("hedges_settlement"),
               "option_mid_trade_cash": trade_cash(False),
               "stock_value_change": change("stock_value"),
               "stock_mid_trade_cash": trade_cash(True),
               "external_net_cashflow": cash("external_capital")}
    components = {
        "target": math.fsum(details[k] for k in ("target_value_change", "target_claim_change",
                                                  "target_settlement_cash", "target_entry_premium")),
        "option_hedge": math.fsum(details[k] for k in ("hedge_value_change", "hedge_claim_change",
                                                        "hedge_settlement_cash", "option_mid_trade_cash")),
        "stock": math.fsum((details["stock_value_change"], details["stock_mid_trade_cash"])),
        "dividend": cash("dividend"), "funding": cash("funding"),
        "stock_borrow": cash("stock_borrow"), "entry_fees": -entry_fees,
        "trading_fees": -math.fsum(f["fee"] for f in fills),
        "spread_slippage": -math.fsum(f["quantity"] * f["multiplier"] * (f["price"] - f["mid"])
                                      for f in fills)}
    total = math.fsum(components.values())
    error = total - current["daily_pnl"]

    def gross_balance(row):
        return math.fsum([abs(row.get(k, 0.0)) for k in ("cash", "target_value", "hedge_value", "stock_value")]
                         + [abs(r["present_value"]) for r in row.get("receivables", {}).values()])

    scale = max(1.0, gross_balance(current), gross_balance(previous),
                math.fsum(abs(f["amount"]) for f in cashflows),
                math.fsum(abs(f["quantity"] * f["multiplier"] * f["mid"]) for f in fills),
                abs(entry_premium) + entry_fees)
    tolerance = POLICY["absolute_tolerance"] + POLICY["relative_tolerance"] * scale
    external_error = details["external_net_cashflow"] - current.get("external_net_cashflow", 0.0)
    require(math.isfinite(external_error) and abs(external_error) <= tolerance,
            "external_cashflow_reconciliation", f"{current['at']}: capital difference={external_error}, tolerance={tolerance}")
    prior = previous_attribution or {}
    cumulative = {k: math.fsum((prior.get("cumulative", {}).get(k, 0.0), components[k])) for k in COMPONENTS}
    cumulative_total = math.fsum(cumulative.values())
    cumulative_error = cumulative_total - current["cumulative_pnl"]
    cumulative_tolerance = prior.get("cumulative_tolerance", 0.0) + tolerance
    require(all(math.isfinite(x) for x in (*components.values(), total, error, tolerance,
                                          *cumulative.values(), cumulative_total, cumulative_error,
                                          cumulative_tolerance)), "nonfinite_pnl", current["at"])
    require(abs(error) <= tolerance and abs(cumulative_error) <= cumulative_tolerance,
            "pnl_reconciliation", f"{current['at']}: daily difference={error}, tolerance={tolerance}; "
            f"cumulative difference={cumulative_error}, tolerance={cumulative_tolerance}")
    return {"at": current["at"], **components, "total": total,
            "ledger_pnl": current["daily_pnl"], "reconciliation_error": error, "tolerance": tolerance,
            "cumulative": cumulative, "cumulative_total": cumulative_total,
            "ledger_cumulative_pnl": current["cumulative_pnl"],
            "cumulative_reconciliation_error": cumulative_error,
            "cumulative_tolerance": cumulative_tolerance, "external_cashflow_reconciliation_error": external_error,
            "details": details}
