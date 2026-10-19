"""Dated, immutable path fixings shared by valuation and settlement."""
from dataclasses import replace
from datetime import date, datetime
from .domain import PathContract, SettlementFixing, data, fingerprint, require

POLICY = dict(method="immutable_dated_path_observations_v1", missing="fail_snapshot",
              revisions="reject_after_first_use", availability="at_or_before_snapshot",
              weighting="contract_weights_without_renormalization")


def fixing_from_data(v):
    return SettlementFixing(**{**v, "fixing_date":date.fromisoformat(v["fixing_date"]),
        "observed_at":datetime.fromisoformat(v["observed_at"]), "available_at":datetime.fromisoformat(v["available_at"])})


def observations(contract, snapshot):
    known = []
    for o in contract.observations:
        if o.observation_date > snapshot.at.date():
            continue
        f = snapshot.settlement_fixings.get(o.fixing_id)
        require(f is not None, "missing_path_fixing", f"{contract.contract_id}: {o.fixing_id}")
        f.check_expiry(o.observation_date, snapshot.at)
        known.append((o, f))
    return known


def resolve(snapshot, contracts, book):
    """Caller commits book atomically with the rest of the snapshot state."""
    for key, saved in book.items():
        current = snapshot.settlement_fixings.get(key)
        require(current is None or data(current) == saved, "path_fixing_revision", key)
    merged = {k:fixing_from_data(v) for k,v in book.items()}
    merged.update(snapshot.settlement_fixings)
    resolved = replace(snapshot, settlement_fixings=merged)
    for c in contracts:
        if not isinstance(c, PathContract) or snapshot.at.date() < c.listed_on:
            continue
        for o,f in observations(c,resolved):
            require(o.fixing_id not in book or book[o.fixing_id] == data(f), "path_fixing_revision", o.fixing_id)
            book[o.fixing_id] = data(f)
    return resolved


def native_product(contract, snapshot):
    """Construct the exact native payoff from the contract and available fixings."""
    known = {o.fixing_id:f.value for o,f in observations(contract,snapshot)}
    common = dict(type=contract.product_type, underlying_id=contract.underlying_id,
        currency_id=contract.currency_id, strike=contract.strike, notional=1.,
        side=dict(type=contract.side), payment_date=contract.payment_day.isoformat())
    if contract.product_type == "arithmetic_asian":
        return dict(**common, observations=[dict(date=o.observation_date.isoformat(),weight=o.weight,
            value=dict(type="known",fixing=known[o.fixing_id]) if o.fixing_id in known else dict(type="unknown"))
            for o in contract.observations])
    result = dict(**common, monitoring_dates=[o.observation_date.isoformat() for o in contract.observations])
    past = [known[o.fixing_id] for o in contract.observations if o.observation_date < snapshot.at.date()]
    if past:
        result['historical_extremum'] = (max if contract.side == "call" else min)(past)
    for o in contract.observations:
        if o.observation_date == snapshot.at.date():
            result['valuation_date_fixing'] = known[o.fixing_id]
    return result


def settlement(contract, snapshot, quantity):
    known = observations(contract,snapshot)
    require(len(known) == len(contract.observations), "missing_path_fixing", contract.contract_id)
    payoff = contract.observed_payoff([f.value for _,f in known])
    detail = dict(rule="path_observations", quantity=quantity, multiplier=contract.multiplier,
        expiry=contract.expiry.isoformat(), payment_date=contract.payment_day.isoformat(),
        product_type=contract.product_type, payoff_per_unit=payoff,
        observations=[dict(observation=data(o),fixing=data(f),fixing_hash=fingerprint(f)) for o,f in known])
    amount = quantity*contract.multiplier*payoff
    from .domain import finite
    finite(amount, "path settlement")
    return amount,detail


def payoff_identity(c):
    return (getattr(c,"product_type","european_vanilla"), getattr(c,"observations",()),
            c.expiry,c.strike,c.side,c.settlement_rule,c.settlement_fixing_id,c.payment_day)
