"""Read explicit backtest inputs without evaluating code or loading pickles."""

from datetime import date, datetime
import json

from .domain import (BacktestConfig, Contract, PathContract, PathObservation, ExternalCashflow, NumeraireObservation, PaymentDiscount, Position, Quote, RiskBasis, SettlementFixing, Snapshot, canonical, require)


def input_json(result) -> str:
    m = result.manifest
    return canonical({"document_kind": "hedging_backtest_request", "schema_version": m.get("input_schema_version", 1),
                      **{k: m[k] for k in ("config", "contracts", "portfolio", "market_history", "code_revision")}}) + "\n"


def load_input(payload: str) -> dict:
    doc = json.loads(payload, parse_constant=lambda value: require(False, "invalid_number", value))
    version = doc.get("schema_version")
    require(doc.get("document_kind") == "hedging_backtest_request" and version in (1, 2, 3, 4, 5, 6, 7, 8, 9),
            "schema_version", "unsupported request kind/version")
    require(set(doc) == {"document_kind", "schema_version", "config", "contracts", "portfolio", "market_history", "code_revision"},
            "schema_fields", "unexpected/missing request fields")
    require(("numeraire_id" in doc["config"]) == (version >= 4),
            "schema_fields", "config numeraire field must match request version")
    for field in ("constraint_mode", "gross_notional_limit"):
        require((field in doc["config"]) == (version >= 6), "schema_fields", "budget fields must match request version")
    require(("market_shock_log_moneyness" in doc["config"]) == (version >= 7),
            "schema_fields", "market shock field must match request version")
    if version == 7:
        require(doc["config"]["market_shock_log_moneyness"] is not None,
                "schema_fields", "request v7 requires explicit market shock nodes")
    require(('max_joint_bucket_standard_error' in doc['config']) == (version >= 9),
            'schema_fields', 'joint precision field must match request version')
    if version >= 9:
        require(doc['config']['max_joint_bucket_standard_error'] is not None,
                'schema_fields', 'request v9 requires an explicit joint precision limit')

    def quote(v):
        return Quote(**{**v, "observed_at": datetime.fromisoformat(v["observed_at"]),
                        "available_at": datetime.fromisoformat(v["available_at"])})

    for c in doc["contracts"]:
        keys = {"settlement_rule", "settlement_fixing_id"}
        require(keys <= set(c) if version >= 2 else not keys & set(c),
                "schema_fields", "contract settlement fields must match request version")
        require(("payment_date" in c) == (version >= 3), "schema_fields", "contract payment field must match request version")
    def contract(c):
        require(version >= 8 or not {"product_type","observations"}&set(c), "schema_fields", "path contracts require request v8")
        values = {**c, "expiry": date.fromisoformat(c["expiry"]),
                              "payment_date": date.fromisoformat(c["payment_date"]) if c.get("payment_date") is not None else None,
                              "listed_on": date.fromisoformat(c["listed_on"])}
        if "product_type" in c:
            require(isinstance(c.get("observations"),list), "schema_fields", "path observation array required")
            values['observations'] = tuple(PathObservation(**{**o,"observation_date":date.fromisoformat(o['observation_date'])}) for o in c['observations'])
            return PathContract(**values)
        require('observations' not in c, "schema_fields", "path product type required")
        return Contract(**values)
    contracts = [contract(c) for c in doc['contracts']]
    history = []
    for s in doc["market_history"]:
        require(("settlement_fixings" in s) == (version >= 2),
                "schema_fields", "snapshot fixing field must match request version")
        require(("payment_discounts" in s) == (version >= 3), "schema_fields", "snapshot discount field must match request version")
        require(("numeraire" in s) == (version >= 4), "schema_fields", "snapshot numeraire field must match request version")
        require(("external_cashflows" in s) == (version >= 5),
                "schema_fields", "snapshot external cashflow field must match request version")
        raw_flows = s.get("external_cashflows", [])
        require(isinstance(raw_flows, list) and all(isinstance(f, dict) for f in raw_flows),
                "schema_fields", "external_cashflows must be an array of capital movement objects")
        external_cashflows = tuple(ExternalCashflow(**{**f, "effective_at": datetime.fromisoformat(f["effective_at"]),
                                                       "available_at": datetime.fromisoformat(f["available_at"])})
                                   for f in raw_flows)
        raw_numeraire = s.get("numeraire")
        require(raw_numeraire is None or isinstance(raw_numeraire, dict),
                "schema_fields", "numeraire must be an object or null")
        numeraire = None if raw_numeraire is None else NumeraireObservation(**{
            **raw_numeraire, "valuation_date": date.fromisoformat(raw_numeraire["valuation_date"]),
            "observed_at": datetime.fromisoformat(raw_numeraire["observed_at"]),
            "available_at": datetime.fromisoformat(raw_numeraire["available_at"])})
        raw_discounts = s.get("payment_discounts", {})
        require(isinstance(raw_discounts, dict) and all(isinstance(v, dict) for v in raw_discounts.values()),
                "schema_fields", "payment_discounts must be an object of discount objects")
        discounts = {date.fromisoformat(key): PaymentDiscount(**{**v, "valuation_date": date.fromisoformat(v["valuation_date"]),
                                                                  "observed_at": datetime.fromisoformat(v["observed_at"]),
                                                                  "available_at": datetime.fromisoformat(v["available_at"])})
                     for key, v in raw_discounts.items()}
        raw_fixings = s.get("settlement_fixings", {})
        require(isinstance(raw_fixings, dict) and all(isinstance(v, dict) for v in raw_fixings.values()),
                "schema_fields", "settlement_fixings must be an object of fixing objects")
        fixings = {key: SettlementFixing(**{**v, "fixing_date": date.fromisoformat(v["fixing_date"]),
                                            "observed_at": datetime.fromisoformat(v["observed_at"]),
                                            "available_at": datetime.fromisoformat(v["available_at"])})
                   for key, v in raw_fixings.items()}
        b = s["basis"]
        basis = RiskBasis(**{**b, "valuation_date": date.fromisoformat(b["valuation_date"]),
                             "maturity_dates": tuple(date.fromisoformat(t) for t in b["maturity_dates"])})
        history.append(Snapshot(**{**s, "at": datetime.fromisoformat(s["at"]), "spot": quote(s["spot"]),
                                    "option_quotes": {cid: quote(q) for cid, q in s["option_quotes"].items()},
                                    "settlement_fixings": fixings,
                                    "payment_discounts": discounts,
                                    "numeraire": numeraire,
                                    "external_cashflows": external_cashflows,
                                    "basis": basis, "surface_available_at": datetime.fromisoformat(s["surface_available_at"])}))
    return {"config": BacktestConfig(**doc["config"]), "history": history, "contracts": contracts,
            "portfolio": [Position(**p) for p in doc["portfolio"]], "code_revision": doc["code_revision"]}
