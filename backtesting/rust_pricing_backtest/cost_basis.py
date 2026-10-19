"""Moving-average execution cost, with realization on close or payoff fixing."""

from copy import deepcopy
import math

from .domain import checked_sum, require


GROUPS = {"targets": "target", "hedges": "option_hedge", "stock": "stock"}
CASH_COMPONENTS = ("dividend", "funding", "stock_borrow", "entry_fees", "trading_fees")
POLICY = {"method": "moving_average_execution_price_v1", "entry_order": "portfolio_input_order",
          "fees": "expensed_separately", "spread_slippage": "included_in_execution_cost_basis",
          "expiry": "realize_fixed_face_amount_then_mark_claim_from_face_cost",
          "currency": "nominal_portfolio_currency", "quantity_absolute_tolerance": 1e-12,
          "reconciliation_tolerance": "existing_pnl_daily_and_cumulative_tolerances"}


def new_book():
    return {"positions": {group: {} for group in GROUPS}, "claims": {},
            "realized": {group: 0.0 for group in GROUPS},
            "spread_slippage": {group: 0.0 for group in GROUPS}}


def _sum(values):
    return checked_sum(values, "nonfinite_cost_basis", "cost basis arithmetic overflowed")


def _quantity_matches(a, b):
    # Match the replay engine's small-position convention; never use a relative
    # tolerance that silently permits whole missing contracts in a large book.
    return abs(a-b) <= POLICY["quantity_absolute_tolerance"]


def trade(book, group, contract_id, quantity, price, multiplier, *, mid=None):
    """Book a signed fill. Closing quantity realizes; a reversal opens a new basis."""
    require(group in GROUPS and all(math.isfinite(x) for x in (quantity, price, multiplier))
            and price >= 0 and multiplier > 0, "invalid_cost_basis_trade", contract_id)
    if quantity == 0:
        return
    positions = book["positions"][group]
    old = positions.get(contract_id)
    old_quantity, old_price = (old["quantity"], old["average_price"]) if old else (0.0, 0.0)
    require(old is None or old["multiplier"] == multiplier, "cost_basis_multiplier_mismatch", contract_id)
    new_quantity = _sum((old_quantity, quantity))
    if old_quantity == 0 or (old_quantity > 0) == (quantity > 0):
        average = _sum((abs(old_quantity)/abs(new_quantity)*old_price, abs(quantity)/abs(new_quantity)*price))
    else:
        closed = min(abs(old_quantity), abs(quantity))
        realized = math.copysign(closed, old_quantity) * multiplier * (price-old_price)
        book["realized"][group] = _sum((book["realized"][group], realized))
        average = old_price if abs(quantity) < abs(old_quantity) else price
    if new_quantity == 0:
        positions.pop(contract_id, None)
    else:
        signed_cost = new_quantity * multiplier * average
        require(math.isfinite(signed_cost), "nonfinite_cost_basis", contract_id)
        positions[contract_id] = {"quantity": new_quantity, "multiplier": multiplier,
                                  "average_price": average, "signed_cost": signed_cost}
    if mid is not None:
        book["spread_slippage"][group] = _sum((book["spread_slippage"][group], -quantity*multiplier*(price-mid)))


def fix(book, group, contract_id, quantity, amount, claim_id, *, keep_claim):
    """Realize payoff minus signed option cost; retain a claim at signed face cost."""
    position = book["positions"][group].get(contract_id)
    require(position is not None and _quantity_matches(position["quantity"], quantity),
            "cost_basis_position_mismatch", contract_id)
    book["realized"][group] = _sum((book["realized"][group], amount, -position["signed_cost"]))
    del book["positions"][group][contract_id]
    if keep_claim:
        require(claim_id not in book["claims"], "duplicate_cost_basis_claim", claim_id)
        book["claims"][claim_id] = {"group": group, "contract_id": contract_id, "signed_cost": amount}


def pay(book, claim_id, amount):
    require(claim_id in book["claims"], "missing_cost_basis_claim", claim_id)
    claim = book["claims"].pop(claim_id)
    group = claim["group"]
    book["realized"][group] = _sum((book["realized"][group], amount, -claim["signed_cost"]))


def report(book, ledger, pnl, previous, holdings):
    """Check inventory and explain both daily and cumulative nominal ledger P&L."""
    daily_tolerance, cumulative_tolerance = pnl["tolerance"], pnl["cumulative_tolerance"]
    claims = ledger["receivables"]
    require(set(book["claims"]) == set(claims), "cost_basis_claim_mismatch", ledger["at"])
    for claim_id, claim in claims.items():
        basis = book["claims"][claim_id]
        require(basis["group"] == claim["group"] and basis["contract_id"] == claim["contract_id"]
                and basis["signed_cost"] == claim["amount"], "cost_basis_claim_mismatch", claim_id)
    groups = {}
    for group, component in GROUPS.items():
        positions = book["positions"][group]
        for cid in set(holdings[group]) | set(positions):
            require(_quantity_matches(holdings[group].get(cid, 0.0), positions.get(cid, {}).get("quantity", 0.0)),
                    "cost_basis_position_mismatch", f"{group}:{cid}")
        for cid, position in positions.items():
            expected_cost = position["quantity"] * position["multiplier"] * position["average_price"]
            require(math.isfinite(expected_cost) and abs(expected_cost-position["signed_cost"]) <= cumulative_tolerance,
                    "cost_basis_position_mismatch", f"{group}:{cid}: inconsistent cost")
        open_cost = _sum(p["signed_cost"] for p in positions.values())
        claim_cost = _sum(c["signed_cost"] for c in book["claims"].values() if c["group"] == group)
        mark_field = {"targets": "target_value", "hedges": "hedge_value", "stock": "stock_value"}[group]
        value = _sum((ledger[mark_field], *[c["present_value"] for c in claims.values() if c["group"] == group]))
        unrealized = _sum((value, -open_cost, -claim_cost))
        realized = book["realized"][group]
        spread = book["spread_slippage"][group]
        total = _sum((realized, unrealized))
        expected = _sum((pnl["cumulative"][component], spread))
        prior = previous["groups"][component] if previous else {}
        daily_realized = realized - prior.get("realized_pnl", 0.0)
        unrealized_change = unrealized - prior.get("unrealized_pnl", 0.0)
        daily_total = _sum((daily_realized, unrealized_change))
        expected_daily = _sum((pnl[component], spread, -prior.get("spread_slippage", 0.0)))
        error, daily_error = total - expected, daily_total - expected_daily
        require(abs(error) <= cumulative_tolerance and abs(daily_error) <= daily_tolerance,
                "cost_basis_reconciliation", f"{ledger['at']}: {component}: daily={daily_error}, cumulative={error}")
        groups[component] = {"realized_pnl": realized, "unrealized_pnl": unrealized, "total_pnl": total,
                             "daily_realized_pnl": daily_realized, "daily_unrealized_change": unrealized_change,
                             "daily_total_pnl": daily_total, "open_cost": open_cost, "claim_cost": claim_cost,
                             "market_value": value, "spread_slippage": spread, "expected_pnl": expected,
                             "daily_reconciliation_error": daily_error, "cumulative_reconciliation_error": error}
    spread_error = _sum(book["spread_slippage"].values()) - pnl["cumulative"]["spread_slippage"]
    require(abs(spread_error) <= cumulative_tolerance, "cost_basis_reconciliation", "spread allocation mismatch")
    cash_realized = {key: pnl["cumulative"][key] for key in CASH_COMPONENTS}
    realized = _sum((*book["realized"].values(), *cash_realized.values()))
    unrealized = _sum(g["unrealized_pnl"] for g in groups.values())
    total = _sum((realized, unrealized))
    daily_realized = realized - (previous["realized_pnl"] if previous else 0.0)
    unrealized_change = unrealized - (previous["unrealized_pnl"] if previous else 0.0)
    daily_total = _sum((daily_realized, unrealized_change))
    error, daily_error = total - ledger["cumulative_pnl"], daily_total - ledger["daily_pnl"]
    require(abs(error) <= cumulative_tolerance and abs(daily_error) <= daily_tolerance,
            "cost_basis_reconciliation", f"{ledger['at']}: total: daily={daily_error}, cumulative={error}")
    return {"at": ledger["at"], "groups": groups, "cash_realized_components": cash_realized,
            "realized_pnl": realized, "unrealized_pnl": unrealized, "total_pnl": total,
            "daily_realized_pnl": daily_realized, "daily_unrealized_change": unrealized_change,
            "daily_total_pnl": daily_total, "daily_reconciliation_error": daily_error,
            "cumulative_reconciliation_error": error, "spread_reconciliation_error": spread_error,
            "daily_tolerance": daily_tolerance, "cumulative_tolerance": cumulative_tolerance,
            "positions": deepcopy(book["positions"]), "claims": deepcopy(book["claims"])}
