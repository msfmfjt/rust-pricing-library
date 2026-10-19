"""Maintain the explicit backtest wire contract; never infer it from examples.

Run with --check in CI. Generated schemas are shipped in the companion wheel.
This specification uses only the standard library and does not import the engine.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path


DESTINATION = Path(__file__).resolve().parents[1] / "backtesting/rust_pricing_backtest/schemas"
DRAFT = "https://json-schema.org/draft/2020-12/schema"
BASE = "urn:rust-pricing-library:backtesting:"
BUNDLE = BASE + "definitions:v1"
N = {"type": "number"}
I = {"type": "integer"}
C = {"type": "integer", "minimum": 0}
P = {"type": "number", "exclusiveMinimum": 0}
Z = {"type": "number", "minimum": 0}
B = {"type": "boolean"}
S = {"type": "string"}
TEXT = {"type": "string", "minLength": 1, "pattern": "\\S"}
H = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
D = {"type": "string", "format": "date", "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"}
T = {"type": "string", "format": "date-time", "pattern": "T.*(Z|[+-][0-9]{2}:[0-9]{2})$"}
NULL = {"type": "null"}
RATIO = {"type": "number", "minimum": 0, "maximum": 1}
PROBABILITY = {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1}


def ref(name):
    return {"$ref": f"#/$defs/{name}"}


def nullable(schema):
    return {"anyOf": [schema, NULL]}


def array(items, minimum=0, **extra):
    return {"type": "array", "items": items, "minItems": minimum, **extra}


def mapping(items):
    return {"type": "object", "additionalProperties": items}


def enum(*values):
    return {"enum": list(values)}


def const(value):
    return {"const": value}


def obj(properties, optional=()):
    return {"type": "object", "additionalProperties": False,
            "required": [k for k in properties if k not in optional], "properties": properties}


def fields(names, schema=N):
    return {k: schema for k in names.split()}


def when(key, value, then, otherwise=None):
    result = {"if": {"properties": {key: const(value)}, "required": [key]}, "then": then}
    if otherwise is not None:
        result["else"] = otherwise
    return result


STRATEGIES = ("unhedged", "delta", "scalar_vega", "vega_kt")
STATUS = enum("completed", "partial", "failed")
COMPONENTS = "target option_hedge stock dividend funding stock_borrow entry_fees trading_fees spread_slippage"
CASH = "dividend funding stock_borrow entry_fees trading_fees"
GROUPS = ("target", "option_hedge", "stock")


def specification():
    d = {}
    d["metadata"] = {"type": "object", "$comment": "Provider and policy metadata are explicit extension points; JSON values are retained."}
    d["risk_validation"] = {"type": "object", "required": ["status"], "properties": {"status": TEXT}}
    config = {"strategy": enum(*STRATEGIES), "rebalance_every": {"type": "integer", "minimum": 1},
              **fields("initial_cash cash_floor"),
              **fields("option_position_limit option_trade_limit stock_position_limit stock_trade_limit", Z),
              "stock_lot": P, **fields("allow_option_shorts allow_stock_shorts allow_identical_hedge", B),
              **fields("option_fee stock_fee max_quote_age_seconds max_relative_spread min_vega ridge turnover_penalty", Z),
              "slippage_bps": {**Z, "exclusiveMaximum": 10000}, "solver_tolerance": P,
              "solver_iterations": {"type": "integer", "minimum": 1},
              **fields("max_abs_projection_residual max_abs_edge_sensitivity max_price_standard_error max_bucket_standard_error", Z),
              "target_mark": enum("model", "market"), "end_policy": enum("mark_to_market", "liquidate_hedges"),
              "numeraire_id": nullable(TEXT)}
    d["config"] = obj(config)
    # load_input accepts omitted dataclass defaults, except the versioned field.
    d["input_config"] = obj(config, optional=set(config)-{"numeraire_id"})
    contract = {"contract_id": {**TEXT, "not": const("__spot__")}, "expiry": D, "strike": P,
                "side": enum("call", "put"), "multiplier": P, "lot": P, "listed_on": D,
                "underlying_id": I, "currency_id": I, "settlement_rule": enum("snapshot_spot", "explicit_fixing"),
                "settlement_fixing_id": nullable(TEXT), "payment_date": nullable(D)}
    d["contract"] = obj(contract)
    d["contract"]["allOf"] = [when("settlement_rule", "explicit_fixing",
        {"properties": {"settlement_fixing_id": TEXT}}, {"properties": {"settlement_fixing_id": NULL}})]
    d["input_contract"] = deepcopy(d["contract"])
    d["input_contract"]["required"] = [k for k in contract if k not in ("side", "multiplier", "lot", "underlying_id", "currency_id")]
    d["position"] = obj({"contract_id": TEXT, "quantity": N, "entry_price": Z, "entry_fee": Z})
    d["input_position"] = deepcopy(d["position"])
    d["input_position"]["required"].remove("entry_fee")
    d["quote"] = obj({"bid": Z, "ask": Z, "observed_at": T, "available_at": T})
    provenance = {"observed_at": T, "available_at": T, "source": TEXT, "revision": TEXT}
    d["fixing"] = obj({"fixing_date": D, "value": Z, **provenance})
    d["discount"] = obj({"valuation_date": D, "factor": P, **provenance})
    d["numeraire"] = obj({"account_id": TEXT, "currency_id": I, "valuation_date": D, "value": P, **provenance})
    d["capital"] = obj({"flow_id": TEXT, "effective_at": T, "amount": {**N, "not": const(0)},
                        "currency_id": I, "available_at": T, "source": TEXT, "revision": TEXT})
    d["basis"] = obj({"valuation_date": D, "maturity_dates": array(D, 2, uniqueItems=True),
        "log_moneyness": array(N, 2, uniqueItems=True), **fields("method coordinate delta_convention", TEXT),
        "unit": const("currency_per_unit_absolute_volatility")})
    d["input_basis"] = deepcopy(d["basis"])
    d["input_basis"]["required"] = ["valuation_date", "maturity_dates", "log_moneyness"]
    d["snapshot"] = obj({"snapshot_id": TEXT, "at": T, "spot": ref("quote"),
        "option_quotes": mapping(ref("quote")), "basis": ref("basis"), "surface_available_at": T, "source": TEXT,
        "pricing_requests": mapping({**S, "contentMediaType": "application/json",
            "$comment": "JSON-encoded native requests. contentMediaType is annotation, not native schema validation."}),
        **fields("lending_rate borrowing_rate"), **fields("stock_borrow_rate dividend_per_share", Z),
        "settlement_fixings": mapping(ref("fixing")),
        "payment_discounts": {**mapping(ref("discount")), "propertyNames": D},
        "numeraire": nullable(ref("numeraire")), "external_cashflows": array(ref("capital"))})
    d["input_snapshot"] = deepcopy(d["snapshot"])
    d["input_snapshot"]["properties"]["basis"] = ref("input_basis")
    d["input_snapshot"]["required"] = [k for k in d["snapshot"]["required"] if k not in
        ("pricing_requests", "lending_rate", "borrowing_rate", "stock_borrow_rate", "dividend_per_share")]
    common_input = {"config": ref("config"), "contracts": array(ref("contract"), 1),
                    "portfolio": array(ref("position"), 1), "market_history": array(ref("snapshot"), 1), "code_revision": S}
    d["request_v5"] = obj({"document_kind": const("hedging_backtest_request"), "schema_version": const(5),
        **common_input, "config": ref("input_config"), "contracts": array(ref("input_contract"), 1),
        "portfolio": array(ref("input_position"), 1), "market_history": array(ref("input_snapshot"), 1)})
    d["order"] = obj({"decision_at": T, "desired": mapping(N), "stock": N,
        "status": enum("pending", "filled", "rejected_cash_limit"),
        "kind": enum("rebalance", "precommitted_terminal_liquidation"),
        "fill_at": T, "expired_contracts": array(TEXT)}, optional=("fill_at", "expired_contracts"))
    d["order"]["allOf"] = [when("status", "filled", {"required": ["fill_at"]})]
    d["fill"] = obj({"at": T, "decision_at": T, "contract_id": TEXT, "quantity": N,
        "multiplier": P, "price": Z, "mid": Z, "fee": Z, "cash_change": N, "execution_cost": Z, "liquidation": B})
    d["settlement_detail"] = obj({"rule": enum("snapshot_spot", "explicit_fixing"), "quantity": N,
        "multiplier": P, "expiry": D, "payment_date": D, "underlying_price": Z, "payoff_per_unit": Z,
        "fixing_id": TEXT, "fixing": ref("fixing"), "fixing_hash": H,
        "snapshot_id": TEXT, "source": TEXT, "spot_quote": ref("quote")},
        optional=("fixing_id", "fixing", "fixing_hash", "snapshot_id", "source", "spot_quote"))
    d["settlement_detail"]["allOf"] = [when("rule", "explicit_fixing",
        {"required": ["fixing_id", "fixing", "fixing_hash"]}, {"required": ["snapshot_id", "source", "spot_quote"]})]
    claim = {"claim_id": TEXT, "group": enum("targets", "hedges"), "contract_id": TEXT,
             "fixed_at": T, "payment_date": D, "amount": N, "settlement": ref("settlement_detail")}
    d["claim"] = obj(claim)
    d["claim_mark"] = obj({**claim, "discount": ref("discount"), "discount_hash": H, "present_value": N})
    d["settlement"] = obj({**claim, "at": T, "event": enum("fixed", "fixed_zero", "paid")})
    base_flow = {"at": T, "contract_id": S, "amount": N}
    d["cashflow"] = {"oneOf": [
        obj({**base_flow, "kind": enum("target_entry", "hedge_trade", "funding", "stock_borrow", "dividend")}),
        obj({**base_flow, "kind": enum("targets_settlement", "hedges_settlement"), "settlement": ref("settlement_detail")}),
        obj({**base_flow, "kind": const("external_capital"), "flow_id": TEXT,
             "external_cashflow": ref("capital"), "external_cashflow_hash": H,
             "numeraire_value": nullable(P), "numeraire_hash": nullable(H), "discounted_amount": nullable(N)})]}
    d["ledger"] = obj({"at": T, **fields("cash stock target_value hedge_value stock_value receivable_value equity daily_pnl cumulative_pnl external_net_cashflow cumulative_external_net_cashflow delta"),
        "raw_vega_l2": Z, "hedges": mapping(N), "receivables": mapping(ref("claim_mark")), "fully_settled": B,
        "numeraire": nullable(ref("numeraire")), "numeraire_hash": nullable(H),
        **fields("discounted_equity discounted_daily_pnl discounted_cumulative_pnl discounted_external_net_cashflow discounted_cumulative_external_cashflow", nullable(N))})
    d["risk"] = obj({"price": Z, "delta": N, "buckets": array(N), "basis_id": H, "signed_residual": N,
        "pre_projection": nullable(N), "price_standard_error": nullable(Z), "edge_sensitivity_abs": Z,
        "provenance": S, "bucket_standard_errors": nullable(array(Z)), "warning_codes": array(S),
        "native_policy": nullable(S), "payment_date": nullable(D), "payment_scaling_factor": P,
        "gamma": nullable(N), "gamma_method": nullable(TEXT), "gamma_spot_bump": nullable(P),
        "reporting_iv": nullable(array(P, 4)), "reporting_forwards": nullable(array(P, 2))})
    d["risk"]["allOf"] = [when("gamma", None, {"properties": {"gamma_method": NULL, "gamma_spot_bump": NULL}}),
        when("reporting_iv", None, {"properties": {"reporting_forwards": NULL}},
             {"properties": {"reporting_forwards": array(P, 2)}})]
    d["risk_row"] = obj({"at": T, "basis": ref("basis"), "basis_id": H,
        "target_buckets": array(N), "held_buckets": array(N), "target_delta": N, "held_delta": N,
        "target_projection_residual_abs": Z, "instruments": mapping(ref("risk"))})
    d["solver"] = obj({"quantities": array(N), "iterations": C, "rank": C, "singular_values": array(Z),
        "condition_number": nullable(Z), **fields("objective_before objective_after projected_gradient", Z)})
    d["decision"] = obj({"at": T, "strategy": enum(*STRATEGIES), "candidates": array(TEXT),
        "eligible": array(TEXT), "solver": ref("solver"), "matrix": array(array(N)),
        "target": array(N), "rounded_residual": array(N)}, optional=("eligible", "solver", "matrix", "target", "rounded_residual"))
    d["decision"]["allOf"] = [{"if": {"properties": {"strategy": enum("scalar_vega", "vega_kt")}},
        "then": {"required": ["eligible", "solver", "matrix", "target", "rounded_residual"]}}]
    d["components"] = obj(fields(COMPONENTS))
    d["discounted_attribution"] = obj(fields("pnl_at_current_numeraire numeraire_carry total ledger_pnl reconciliation_error cumulative_total ledger_cumulative_pnl cumulative_reconciliation_error") |
                                         fields("tolerance cumulative_tolerance", Z))
    d["pnl_row"] = obj({"at": T, **fields(COMPONENTS), "cumulative": ref("components"),
        **fields("total ledger_pnl reconciliation_error cumulative_total ledger_cumulative_pnl cumulative_reconciliation_error external_cashflow_reconciliation_error"),
        **fields("tolerance cumulative_tolerance", Z), "discounted": nullable(ref("discounted_attribution")),
        "details": obj(fields("target_value_change hedge_value_change stock_value_change target_claim_change hedge_claim_change target_entry_premium target_settlement_cash hedge_settlement_cash option_mid_trade_cash stock_mid_trade_cash external_net_cashflow"))})
    d["cost_position"] = obj({"quantity": N, "multiplier": P, "average_price": Z, "signed_cost": N})
    d["cost_claim"] = obj({"group": enum("targets", "hedges"), "contract_id": TEXT, "signed_cost": N})
    positions = obj({k: mapping(ref("cost_position")) for k in ("targets", "hedges", "stock")})
    d["cost_book"] = obj({"positions": positions, "claims": mapping(ref("cost_claim")),
        "realized": obj(fields("targets hedges stock")), "spread_slippage": obj(fields("targets hedges stock"))})
    d["cost_group"] = obj(fields("realized_pnl unrealized_pnl total_pnl daily_realized_pnl daily_unrealized_change daily_total_pnl open_cost claim_cost market_value spread_slippage expected_pnl daily_reconciliation_error cumulative_reconciliation_error"))
    cost_summary = {"groups": obj({k: ref("cost_group") for k in GROUPS}),
                    "cash_realized_components": obj(fields(CASH)), **fields("realized_pnl unrealized_pnl total_pnl")}
    d["cost_summary"] = obj(cost_summary)
    d["cost_row"] = obj({"at": T, **cost_summary, "positions": positions, "claims": mapping(ref("cost_claim")),
        **fields("daily_realized_pnl daily_unrealized_change daily_total_pnl daily_reconciliation_error cumulative_reconciliation_error spread_reconciliation_error"),
        **fields("daily_tolerance cumulative_tolerance", Z)})
    d["greek_group"] = obj({**fields("actual_pnl unexplained_pnl"),
        **fields("available_greek_pnl delta_pnl gamma_pnl opening_delta opening_gamma", nullable(N)),
        "vega_pnl": NULL, "gamma_missing_contracts": array(TEXT),
        "gamma_status": enum("available", "missing", "no_interval")})
    d["greek_cumulative"] = obj(fields("available_greek_pnl non_asset_pnl unexplained_pnl"))
    vega_status = const("not_computed_risk_and_delta_iv_convention_unvalidated")
    d["greek_row"] = obj({"at": T, "from_at": nullable(T), "elapsed_calendar_days": nullable(C),
        "spot_change": nullable(N), "risk_snapshot_hash": nullable(H), "inventory_snapshot_hash": nullable(H),
        "delta_convention": nullable(S), "risk_warning_codes": array(S),
        "groups": obj({k: ref("greek_group") for k in GROUPS}), "available_greek_pnl": nullable(N),
        **fields("unexplained_pnl non_asset_pnl total ledger_pnl reconciliation_error cumulative_total cumulative_reconciliation_error"),
        **fields("tolerance cumulative_tolerance", Z), "non_asset_components": obj(fields(CASH + " spread_slippage")),
        "cumulative": ref("greek_cumulative"), "vega_status": vega_status})
    d["iv_node"] = obj({"index": C, "maturity": D, "opening_k": N,
        **fields("opening_forward canonical_strike opening_iv current_iv", nullable(P)),
        **fields("current_k delta_iv", nullable(N)), "current_indices": array(C, maxItems=2),
        "weights": array(RATIO, maxItems=2), "status": enum("matched", "expired_maturity", "missing_opening_surface",
            "missing_current_surface", "missing_absolute_maturity", "outside_strike_domain")})
    d["iv_node"]["allOf"] = [when("status", "matched",
        {"properties": {"delta_iv": N, "current_iv": P, "current_indices": array(C, 1, maxItems=2),
                        "weights": array(RATIO, 1, maxItems=2)}}, {"properties": {"delta_iv": NULL}})]
    d["iv_group"] = obj({"opening_raw_buckets": array(N), "absolute_instrument_buckets": array(Z),
        "covered_bucket_dot": N, "complete_bucket_dot": nullable(N),
        **fields("covered_absolute_risk uncovered_absolute_risk", Z), "absolute_risk_coverage": nullable(RATIO),
        "status": enum("no_opening_risk", "complete", "partial")})
    application = const("diagnostic_only_not_added_to_greek_pnl")
    d["iv_row"] = obj({"at": T, "from_at": nullable(T), "application": application, "pnl_adopted": const(False),
        "blockers": array(S), "nodes": array(ref("iv_node")),
        "groups": obj({k: ref("iv_group") for k in ("target", "option_hedge")}, optional=("target", "option_hedge")),
        **fields("opening_surface_hash current_surface_hash opening_basis_id", nullable(H)), "current_basis_id": H,
        **fields("current_surface_contracts opening_surface_contracts", array(TEXT)),
        "status": enum("no_interval", "unsupported_coordinate", "complete", "partial"),
        **fields("risk_method delta_convention", S), **fields("matched_nodes total_nodes", C), "risk_warning_codes": array(S)},
        optional=("risk_method", "delta_convention", "matched_nodes", "total_nodes", "risk_warning_codes"))
    d["iv_row"]["allOf"] = [{"if": {"properties": {"status": enum("complete", "partial")}},
        "then": {"required": ["risk_method", "delta_convention", "matched_nodes", "total_nodes", "risk_warning_codes"],
                 "properties": {"groups": {"required": ["target", "option_hedge"]}}}}]
    table_names = {"ledger": "ledger", "orders": "order", "fills": "fill", "cashflows": "cashflow", "risks": "risk_row",
        "decisions": "decision", "settlements": "settlement", "pnl_components": "pnl_row", "cost_basis": "cost_row",
        "greek_pnl": "greek_row", "iv_alignment": "iv_row"}
    tables = {k: array(ref(v)) for k, v in table_names.items()}
    state_holdings = {"cash": N, "stock": N, "targets": mapping(N), "hedges": mapping(N), "receivables": mapping(ref("claim"))}
    d["state"] = obj({**state_holdings, "next_index": C, "pending": nullable(C), "external_cashflow_ids": mapping(H),
                      "cost_basis_book": ref("cost_book"), **tables})
    d["checkpoint_v7"] = obj({"schema_version": const(7), **fields("identity_hash history_prefix_hash state_hash", H), "state": ref("state")})
    d["failure"] = obj({"code": TEXT, "message": S, "snapshot_id": TEXT, "snapshot_index": C, "attempted_at": T,
        "last_completed_at": nullable(T), "state_boundary": const("before_failed_snapshot"),
        "remaining_state": obj(state_holdings), "pending_order": nullable(ref("order"))})
    d["reconciliation"] = obj({"observations": C, "passed": nullable(const(True)),
        **fields("max_abs_daily_difference cumulative_difference", nullable(N))})
    d["discounted_summary"] = obj({"account_id": TEXT, "currency_id": I, "initial_value": P, "final_value": P,
        "observations": C, **fields("cumulative_pnl external_net_cashflow"), "terminal_hedging_error": nullable(N),
        "reconciliation": obj({"passed": const(True), **fields("max_abs_daily_difference cumulative_difference")})})
    d["summary"] = obj({**fields("cumulative_pnl daily_mean daily_standard_deviation daily_rmse worst_day max_drawdown_currency terminal_hedging_error receivable_value", nullable(N)),
        **fields("execution_cost absolute_traded_value target_entry_fees stock_borrow_cost", Z), "funding_pnl": N,
        "fully_settled": nullable(B), **fields("observations outstanding_payments", C), "scheduled_snapshots": {**C, "minimum": 1},
        "coverage_fraction": RATIO, "status": STATUS, "last_completed_at": nullable(T), "risk_validation_status": TEXT,
        "external_capital": obj({"count": C, "net_cashflow": N, "gross_inflows": Z, "gross_outflows": Z}),
        "discounted_pnl": nullable(ref("discounted_summary")), "pnl_components": nullable(ref("components")),
        "cost_basis": nullable(ref("cost_summary")), "pnl_reconciliation": ref("reconciliation"),
        "greek_pnl": obj({"scope": const("spot_delta_optional_gamma"),
            **fields("observations intervals intervals_with_complete_gamma", C), "vega_status": vega_status,
            "cumulative": nullable(ref("greek_cumulative")),
            **fields("max_abs_reconciliation_difference cumulative_reconciliation_difference", nullable(N))}),
        "iv_alignment": obj({**fields("observations intervals intervals_with_all_nodes_matched", C),
            "application": application, "pnl_adopted": const(False), "last_status": nullable(S)})})
    d["summary"]["allOf"] = [{"if": {"properties": {"fully_settled": enum(False, None)}},
        "then": {"properties": {"terminal_hedging_error": NULL,
            "discounted_pnl": {"anyOf": [NULL, {"properties": {"terminal_hedging_error": NULL}}]}}}}]
    d["summary"]["properties"].update(fields("daily_standard_deviation daily_rmse max_drawdown_currency", nullable(Z)))
    manifest = {"document_kind": const("hedging_backtest_result"), "schema_version": const(9),
        "input_schema_version": const(5), **common_input, "provider": TEXT, "risk_validation": ref("risk_validation"),
        **fields("implementation_hash identity_hash history_hash full_history_hash", H),
        "schedule": array(T, 1, uniqueItems=True), "processed_snapshots": C, **fields("python platform numpy", TEXT),
        "execution": const("next_snapshot_bid_ask"), "funding": const("previous_rate_act365f_continuous"),
        "failure_policy": const("fail_run"), "snapshot_commit_policy": const("atomic_snapshot_v1"),
        "lot_rounding": const("eight_ulp_integer_snap_then_toward_zero_with_bounds_v1"),
        "settlement_policy": const("fix_then_discounted_claim_then_pay_v1"), "snapshot_spot_settlement_contracts": array(TEXT),
        **fields("pnl_attribution cost_basis_policy greek_pnl_policy iv_alignment_policy external_cashflow_policy numeraire_policy", ref("metadata")),
        "target_mark": enum("model", "market"), "risk_unit": const("currency_per_unit_absolute_volatility")}
    d["manifest"] = obj(manifest)
    d["result_v9"] = obj({"status": STATUS, "manifest": ref("manifest"), **tables,
                           "summary": ref("summary"), "checkpoint": ref("checkpoint_v7"), "failure": nullable(ref("failure"))})
    d["result_v9"]["allOf"] = [when("status", "failed", {"properties": {"failure": ref("failure")}},
                                        {"properties": {"failure": NULL}})]
    for status in ("completed", "partial", "failed"):
        properties = {"status": const(status)}
        if status != "completed":
            properties["terminal_hedging_error"] = NULL
            properties["discounted_pnl"] = {"anyOf": [NULL, {"properties": {"terminal_hedging_error": NULL}}]}
        d["result_v9"]["allOf"].append(when("status", status, {"properties": {"summary": {"properties": properties}}}))

    # Ensemble requests and outputs are independent of per-run schema versions.
    d["ensemble_config"] = obj({"metric": enum("nominal", "discounted"),
        "sampling_design": enum("dependent_or_unknown", "independent_simulation"),
        "loss_levels": array(PROBABILITY, 1, uniqueItems=True), "confidence_level": PROBABILITY,
        "bootstrap_resamples": {**C, "minimum": 100}, "bootstrap_seed": C,
        "minimum_ci_samples": {**C, "minimum": 2}, "minimum_tail_observations": {**C, "minimum": 1},
        "minimum_baseline_variance": Z})
    d["input_ensemble_config"] = deepcopy(d["ensemble_config"])
    d["input_ensemble_config"]["required"] = []
    d["outer_randomness"] = {"type": "object", "properties": {"algorithm": TEXT, "domain": TEXT, "seed": C}}
    scenario = {"scenario_id": TEXT, "replication_id": TEXT, "source": TEXT, "outer_randomness": nullable(ref("outer_randomness"))}
    d["scenario_request"] = obj({**scenario, "results": obj({k: TEXT for k in STRATEGIES}, optional=STRATEGIES)}, optional=("outer_randomness",))
    d["ensemble_request_v1"] = obj({"document_kind": const("hedging_backtest_ensemble_request"), "schema_version": const(1),
        "config": ref("input_ensemble_config"), "scenarios": array(ref("scenario_request"), 1)})
    d["ensemble_request_v1"]["allOf"] = [{"if": {"properties": {"config": {"properties": {
        "sampling_design": const("independent_simulation")}, "required": ["sampling_design"]}}},
        "then": {"properties": {"scenarios": {"items": {"required": ["outer_randomness"], "properties": {
            "outer_randomness": {"type": "object", "required": ["algorithm", "domain", "seed"]}}}}}}}]
    d["scenario_run"] = obj({"strategy": enum(*STRATEGIES), "status": enum(*STATUS["enum"], "missing"),
        "reason": nullable(enum("missing_strategy", "partial", "failed", "unsettled", "metric_unavailable")),
        "terminal_error": nullable(N), "observations": nullable(C), "scheduled_snapshots": nullable(C),
        "coverage_fraction": nullable(RATIO), "last_completed_at": nullable(T), "failure_code": nullable(S),
        **fields("execution_cost absolute_traded_value", nullable(Z)), "result_hash": nullable(H)})
    d["scenario_run"]["allOf"] = [{"if": {"properties": {"status": enum("missing", "partial", "failed")}},
                                     "then": {"properties": {"terminal_error": NULL}}}]
    d["scenario_output"] = obj({**scenario, "full_history_hash": nullable(H), "paired": B,
        "runs": obj({k: ref("scenario_run") for k in STRATEGIES})})
    interval = nullable(array(N, 2, maxItems=2))
    d["tail"] = obj({"level": PROBABILITY, "quantile": nullable(N), "expected_shortfall": nullable(N),
        "effective_tail_observations": Z, "status": enum("unavailable", "reference_only", "estimate"),
        "confidence_intervals": nullable(obj({"quantile": interval, "expected_shortfall": interval})),
        "confidence_interval_reason": nullable(S)})
    d["tail"]["allOf"] = [{"if": {"properties": {"status": enum("unavailable", "reference_only")}},
                            "then": {"properties": {"confidence_intervals": NULL}}}]
    d["strategy_statistics"] = obj({"sample_count": C, **fields("mean standard_deviation rmse", nullable(N)),
        "loss_tail": array(ref("tail"), 1), "run_counts": obj(fields("completed partial failed missing", C)),
        **fields("terminal_eligible_count excluded_from_pair_count", C),
        **fields("mean_execution_cost_nominal mean_absolute_traded_value_nominal", nullable(Z)),
        "confidence_intervals": nullable(obj(fields("mean standard_deviation rmse", interval))),
        "confidence_interval_reason": nullable(S), "risk_validation": ref("risk_validation")})
    d["strategy_statistics"]["properties"].update(fields("standard_deviation rmse", nullable(Z)))
    d["comparison"] = obj({"baseline": const("delta"), "sample_count": C,
        **fields("mean_error_difference variance_improvement", nullable(N)), "variance_improvement_reason": nullable(S),
        "confidence_intervals": nullable(obj(fields("mean_error_difference variance_improvement", interval))),
        "confidence_interval_reason": nullable(S), "undefined_variance_bootstrap_resamples": nullable(C),
        "variance_interval_reason": S}, optional=("variance_interval_reason",))
    shared = {k: v for k, v in manifest.items() if k not in ("market_history", "full_history_hash", "history_hash", "identity_hash", "processed_snapshots")}
    shared["config"] = obj({k: v for k, v in config.items() if k != "strategy"})
    shared["initial_numeraire"] = nullable(ref("numeraire"))
    shared["pricing_controls"] = array(obj({"basis": ref("basis"), "requests": mapping(obj({
        "engine": ref("metadata"), "risk": ref("metadata"), "model_type": TEXT, "grid": ref("metadata")}))}))
    d["shared_conditions"] = obj(shared)
    ensemble_manifest = {"document_kind": const("hedging_backtest_ensemble"), "schema_version": const(1),
        "config": ref("ensemble_config"), "status": enum("complete_cohort", "incomplete_cohort"),
        **fields("scenario_count paired_sample_count excluded_scenario_count distinct_replication_count repeated_replication_count overlapping_calendar_window_pairs", C),
        "paired_scenario_ids": array(TEXT, uniqueItems=True), "observation_schedule": array(T, 1),
        "population": const("intersection_of_four_completed_fully_settled_runs_with_requested_metric"),
        "selection_caveat": const("complete_case_statistics_can_be_biased_when_runs_fail"),
        "independence": const("caller_declared_not_inferred_from_dates_or_tested"),
        "inference_scope": const("outer_scenarios_conditional_on_fixed_pricing_randomness"),
        "bootstrap": const("paired_scenario_percentile_pcg64_linear_quantiles"),
        "tail_method": const("empirical_inverse_cdf_and_fractional_upper_tail_integral"),
        "loss_sign": const("negative_terminal_hedging_error"), "standard_deviation": const("sample_ddof_1"),
        "interpretation": const("descriptive_statistics_not_numerical_acceptance"), "risk_validation": ref("risk_validation"),
        **fields("conditions_hash input_hash implementation_hash", H), "shared_conditions": ref("shared_conditions"),
        **fields("python numpy", TEXT)}
    d["ensemble_v1"] = obj({"manifest": obj(ensemble_manifest), "scenarios": array(ref("scenario_output"), 1),
        "strategies": obj({k: ref("strategy_statistics") for k in STRATEGIES}),
        "comparisons": obj({k: ref("comparison") for k in STRATEGIES if k != "delta"})})
    return d


DOCUMENTS = {"request-v5": "request_v5", "result-v9": "result_v9", "checkpoint-v7": "checkpoint_v7",
             "ensemble-request-v1": "ensemble_request_v1", "ensemble-v1": "ensemble_v1"}
CURRENT_DOCUMENTS = {"request-v6": "request_v6", "result-v10": "result_v10",
                     "checkpoint-v8": "checkpoint_v8", "ensemble-v2": "ensemble_v2"}
MARKET_SHOCK_DOCUMENTS = {"request-v7": "request_v7", "result-v11": "result_v11",
                          "checkpoint-v9": "checkpoint_v9", "ensemble-v3": "ensemble_v3"}


def current_specification():
    # Freeze v1 above: historical documents retain their exact wire contract.
    # References within the new bundle resolve only to the copied v2 definitions.
    renamed = {"request_v5": "request_v6", "result_v9": "result_v10",
               "checkpoint_v7": "checkpoint_v8", "ensemble_v1": "ensemble_v2"}
    payload = json.dumps(specification())
    for old, new in renamed.items():
        payload = payload.replace(f'"{old}"', f'"{new}"').replace(f'#/$defs/{old}"', f'#/$defs/{new}"')
    d = json.loads(payload)
    budget_config = {"constraint_mode": enum("execution_cash", "decision_cash_gross"),
                     "gross_notional_limit": nullable(Z)}
    relation = when("constraint_mode", "decision_cash_gross",
                    {"properties": {"gross_notional_limit": Z}},
                    {"properties": {"gross_notional_limit": NULL}})
    for key in ("config", "input_config"):
        d[key]["properties"].update(budget_config)
        d[key]["required"].extend(budget_config)
        d[key]["allOf"] = [relation]
    d["request_v6"]["properties"]["schema_version"] = const(6)
    d["checkpoint_v8"]["properties"]["schema_version"] = const(8)
    d["budget_metrics"] = obj({"cash_after": N, "gross_notional": Z})
    d["budget_inputs"] = obj({"deltas": array(N),
        **fields("target_delta stock_previous stock_lower stock_upper cash cash_floor"),
        "linear": array(N, 1), "spreads": array(Z, 1), "gross_weights": array(Z, 1),
        **fields("fixed_gross gross_limit", Z)})
    d["budget_audit"] = obj({"inputs": ref("budget_inputs"), "rounded": ref("budget_metrics"), "rounded_delta": N})
    d["budget_solver"] = obj({**d["solver"]["properties"], "projected_gradient": NULL,
        "stock": N, "cash_after": N, "gross_notional": Z,
        "method": const("decision_quote_cash_gross_delta_neutral_qp_v1"), "scipy_version": TEXT,
        **fields("kkt_stationarity kkt_complementarity primal_violation", Z), "objective_scale": P,
        "optimizer_status": nullable(I), "optimizer_message": TEXT, "optimizer_success": B})
    d["solver"] = {"anyOf": [{"$ref": f"{BUNDLE}#/$defs/solver"}, ref("budget_solver")]}
    d["decision"]["properties"]["budget"] = ref("budget_audit")
    d["decision"]["properties"]["option_constraints"] = obj({k: array(N) for k in ("previous", "lower", "upper", "costs")})
    d["order"]["properties"].update({"status": enum("pending", "filled", "rejected_cash_limit", "rejected_gross_limit"),
                                       "execution_budget": ref("budget_metrics")})
    d["order"]["allOf"] = [when("status", status, {"required": ["fill_at"]})
                              for status in ("filled", "rejected_cash_limit", "rejected_gross_limit")]
    backend = obj({"policy": enum("box_l1_coordinate_descent_v1", "decision_quote_cash_gross_delta_neutral_qp_v1"),
                   "scipy": nullable(TEXT), "numpy": TEXT})
    for key in ("manifest", "shared_conditions"):
        d[key]["properties"].update({"schema_version": const(10), "input_schema_version": const(6),
                                      "optimizer_backend": backend})
        d[key]["required"].append("optimizer_backend")
    shared_config = d["shared_conditions"]["properties"]["config"]
    shared_config["properties"].update(budget_config)
    shared_config["required"].extend(budget_config)
    shared_config["allOf"] = [relation]
    # New aggregations may also consume uniformly historical result-v9 cohorts.
    d["shared_conditions"] = {"anyOf": [d["shared_conditions"], {"$ref": f"{BUNDLE}#/$defs/shared_conditions"}]}
    d["ensemble_v2"]["properties"]["manifest"]["properties"]["schema_version"] = const(2)
    return d


def market_shock_specification():
    # Both older bundles stay frozen. This version is emitted only when market
    # shock nodes are explicitly configured; ordinary replay retains v2 formats.
    payload = json.dumps(current_specification())
    for old, new in {"request_v6": "request_v7", "result_v10": "result_v11",
                     "checkpoint_v8": "checkpoint_v9", "ensemble_v2": "ensemble_v3"}.items():
        payload = payload.replace(f'"{old}"', f'"{new}"').replace(f'#/$defs/{old}"', f'#/$defs/{new}"')
    d = json.loads(payload)
    nodes = array(N, 2, uniqueItems=True)
    for key in ("config", "input_config"):
        d[key]["properties"]["market_shock_log_moneyness"] = nodes
        d[key]["required"].append("market_shock_log_moneyness")
    d["request_v7"]["properties"]["schema_version"] = const(7)
    d["checkpoint_v9"]["properties"]["schema_version"] = const(9)
    d["market_shock_policy"] = const({
        "method": "nested_log_iv_market_shocks_v1",
        "strike_policy": "configured_log_forward_moneyness_fixed_for_run",
        "maturity_policy": "exact_current_snapshot_reporting_maturities",
        "coordinate": "canonical_f_iv", "objective_application": "vega_kt_strategy_only",
        "risk_unit": "currency_per_unit_absolute_volatility",
        "audit_timing": "held_after_fills_and_proposed_after_lot_rounding",
        "complement_metric": "euclidean_reporting_quote_iv", "hedge_effectiveness": "not_established"})
    d["shock_complement"] = obj({"metric": const("euclidean_reporting_quote_iv"),
        **fields("reporting_l2 covered_l2 complement_l2 worst_complement_one_point_pnl", Z),
        "unit_complement_shock": array(N), "complement_buckets": array(N),
        "market_hedge_effectiveness": const("not_established")})
    d["shock_risk"] = obj({"market_buckets": array(N), "complement": ref("shock_complement")})
    identity = {**fields("map_id reporting_basis_id market_basis_id", H), "market_basis": ref("basis")}
    d["held_market_shocks"] = obj({**identity, "target": ref("shock_risk"), "held": ref("shock_risk")})
    d["proposed_market_shocks"] = obj({**identity, "objective": enum("fixed_market_iv", "strategy_default"),
        "proposed": ref("shock_risk"), "proposed_reporting_buckets": array(N),
        "continuous": ref("shock_risk"),
        "optimizer_input": obj({"target": array(N), "matrix": array(array(N))})},
        optional=("continuous", "optimizer_input"))
    for key, definition in (("risk_row", "held_market_shocks"), ("decision", "proposed_market_shocks")):
        d[key]["properties"]["market_shocks"] = ref(definition)
        d[key]["required"].append("market_shocks")
    d["decision"]["allOf"].append({"if": {"properties": {"strategy": enum("scalar_vega", "vega_kt")}},
        "then": {"properties": {"market_shocks": {"required": ["continuous", "optimizer_input"]}}}})
    d["decision"]["allOf"].append(when("strategy", "vega_kt",
        {"properties": {"market_shocks": {"properties": {"objective": const("fixed_market_iv")}}}},
        {"properties": {"market_shocks": {"properties": {"objective": const("strategy_default")}}}}))
    shared = d["shared_conditions"]["anyOf"][0]
    for spec in (d["manifest"], shared):
        spec["properties"].update({"schema_version": const(11), "input_schema_version": const(7),
                                    "market_shock_policy": ref("market_shock_policy")})
        spec["required"].append("market_shock_policy")
    shared_config = shared["properties"]["config"]
    shared_config["properties"]["market_shock_log_moneyness"] = nodes
    shared_config["required"].append("market_shock_log_moneyness")
    d["shared_conditions"] = {"anyOf": [shared, {"$ref": f"{BASE}definitions:v2#/$defs/shared_conditions"}]}
    d["ensemble_v3"]["properties"]["manifest"]["properties"]["schema_version"] = const(3)
    return d


PATH_DOCUMENTS = {"request-v8":"request_v8", "result-v12":"result_v12",
                  "checkpoint-v10":"checkpoint_v10", "ensemble-v4":"ensemble_v4"}


def path_specification():
    payload = json.dumps(market_shock_specification())
    for old,new in {"request_v7":"request_v8","result_v11":"result_v12",
                    "checkpoint_v9":"checkpoint_v10","ensemble_v3":"ensemble_v4"}.items():
        payload = payload.replace(f'"{old}"',f'"{new}"').replace(f'#/$defs/{old}"',f'#/$defs/{new}"')
    d = json.loads(payload)
    d['path_observation'] = obj({'observation_date':D,'fixing_id':TEXT,'weight':P})
    for key in ['contract','input_contract']:
        european = deepcopy(d[key]);path = deepcopy(european)
        path['properties'].update(product_type=enum('arithmetic_asian','fixed_lookback'),
            observations=array(ref('path_observation'),1),settlement_rule=const('path_observations'))
        path['required'].extend(['product_type','observations'])
        d[key] = {'anyOf':[european,path]}
    path_detail = obj({'rule':const('path_observations'),'quantity':N,'multiplier':P,
        'expiry':D,'payment_date':D,'product_type':enum('arithmetic_asian','fixed_lookback'),
        'payoff_per_unit':Z,'observations':array(obj({'observation':ref('path_observation'),
            'fixing':ref('fixing'),'fixing_hash':H}),1)})
    d['settlement_detail'] = {'anyOf':[d['settlement_detail'],path_detail]}
    d['state']['properties']['path_fixings'] = mapping(ref('fixing'))
    d['state']['required'].append('path_fixings')
    d['request_v8']['properties']['schema_version'] = const(8)
    d['request_v8']['properties']['contracts']['contains'] = {'required':['product_type']}
    d['checkpoint_v10']['properties']['schema_version'] = const(10)
    nodes = nullable(array(N,2,uniqueItems=True))
    for key in ['config','input_config']:
        d[key]['properties']['market_shock_log_moneyness'] = nodes
    for key in ['risk_row','decision']:
        d[key]['required'].remove('market_shocks')
    shared = d['shared_conditions']['anyOf'][0]
    for spec in [d['manifest'],shared]:
        spec['properties'].update(schema_version=const(12),input_schema_version=const(8),
            path_observation_policy=const({'method':'immutable_dated_path_observations_v1','missing':'fail_snapshot',
                'revisions':'reject_after_first_use','availability':'at_or_before_snapshot',
                'weighting':'contract_weights_without_renormalization'}))
        spec['required'].remove('market_shock_policy')
        spec['required'].append('path_observation_policy')
    shared['properties']['config']['properties']['market_shock_log_moneyness'] = nodes
    d['shared_conditions'] = {'anyOf':[shared,{'$ref':f'{BASE}definitions:v3#/$defs/shared_conditions'}]}
    d['ensemble_v4']['properties']['manifest']['properties']['schema_version'] = const(4)
    enabled = {'type':'array'}
    for key in ['market_shock_log_moneyness','config','manifest']:
        enabled = {'properties':{key:enabled}}
    present = {'required':['market_shocks']}
    absent = {'not':present}
    d['result_v12']['allOf'].append({'if':enabled,
        'then':{'properties':{'manifest':{'required':['market_shock_policy']},
            'risks':{'items':present},'decisions':{'items':present}}},
        'else':{'properties':{'manifest':{'not':{'required':['market_shock_policy']}},
            'risks':{'items':absent},'decisions':{'items':absent}}}})
    return d


JOINT_DOCUMENTS = {"request-v9":"request_v9", "result-v13":"result_v13",
                   "checkpoint-v11":"checkpoint_v11", "ensemble-v5":"ensemble_v5"}


def joint_specification():
    payload = json.dumps(path_specification())
    for old, new in {"request_v8":"request_v9", "result_v12":"result_v13",
                     "checkpoint_v10":"checkpoint_v11", "ensemble_v4":"ensemble_v5"}.items():
        payload = payload.replace(f'"{old}"', f'"{new}"').replace(f'#/$defs/{old}"', f'#/$defs/{new}"')
    d = json.loads(payload)
    for key in ('config', 'input_config'):
        d[key]['properties']['max_joint_bucket_standard_error'] = Z
        d[key]['required'].append('max_joint_bucket_standard_error')
    d['request_v9']['properties']['schema_version'] = const(9)
    d['request_v9']['properties']['contracts'].pop('contains')
    d['checkpoint_v11']['properties']['schema_version'] = const(11)
    d['state']['required'].remove('path_fixings')
    d['joint_report'] = {'type':'object', 'required':['document_kind','status','report_hash','implementation_identity',
        'instrument_order','coordinate_order','market_mean_covariance','request_hashes'],
        'properties':{'document_kind':const('portfolio_market_risk'),'status':const('passed'),
            'report_hash':H,'implementation_identity':H,'instrument_order':array(TEXT,1,uniqueItems=True),
            'coordinate_order':array(obj({'maturity':P,'log_moneyness':N}),1),
            'market_mean_covariance':array(array(N)), 'request_hashes':mapping(H)},
        '$comment':'Native audit and calibration fields are retained extension metadata; runtime verifies content hashes and moments.'}
    d['joint_gate'] = obj({'status':const('passed'), 'opening_risk_hash':nullable(H), 'quantities':mapping(N),
        'net_buckets':array(N),'bucket_standard_errors':array(Z),'market_mean_covariance':array(array(N)),
        'max_bucket_standard_error':Z,'limit':Z,
        'uncertainty_scope':const('sampling_error_only_excludes_Delta_calibration_and_discretization')})
    d['risk_row']['properties']['joint_market'] = obj({'report':nullable(ref('joint_report')), 'held':ref('joint_gate')})
    d['risk_row']['required'].append('joint_market')
    d['decision']['properties']['joint_market'] = ref('joint_gate')
    d['decision']['required'].append('joint_market')
    d['market_pnl'] = obj({'at':T, 'status':enum('no_interval','available','unavailable'),
        'opening_risk_hash':nullable(H), **fields('actual_asset_pnl available_greek_pnl unexplained_pnl non_asset_pnl total reconciliation_error cumulative_total cumulative_reconciliation_error'),
        'groups':mapping(ref('metadata')), 'reason':nullable(obj({'code':TEXT,'message':TEXT})),
        'cumulative':ref('greek_cumulative')})
    d['greek_row']['properties']['market_iv'] = ref('market_pnl')
    d['greek_row']['required'].append('market_iv')
    d['summary']['properties']['joint_market'] = obj({**fields('held_checks proposed_checks intervals covered_intervals',C),
        'limit':Z, 'pnl_status':enum('complete','partial'), 'max_abs_reconciliation_error':nullable(Z),
        'cumulative':nullable(ref('greek_cumulative'))})
    d['summary']['required'].append('joint_market')
    shared = d['shared_conditions']['anyOf'][0]
    for spec in (d['manifest'], shared):
        spec['properties'].update(schema_version=const(13), input_schema_version=const(9), joint_market_policy=ref('metadata'))
        spec['required'].remove('path_observation_policy')
        spec['required'].append('joint_market_policy')
    shared['properties']['config']['properties']['max_joint_bucket_standard_error'] = Z
    shared['properties']['config']['required'].append('max_joint_bucket_standard_error')
    d['shared_conditions'] = {'anyOf':[shared,{'$ref':f'{BASE}definitions:v4#/$defs/shared_conditions'}]}
    d['ensemble_v5']['properties']['manifest']['properties']['schema_version'] = const(5)
    return d


def artifacts():
    definitions = specification()
    yield "definitions-v1.schema.json", {"$schema": DRAFT, "$id": BUNDLE,
        "title": "Hedging backtest wire definitions v1", "$defs": definitions}
    for name, definition in DOCUMENTS.items():
        yield f"{name}.schema.json", {"$schema": DRAFT, "$id": BASE + name,
            "title": f"Hedging backtest {name}", "$ref": f"{BUNDLE}#/$defs/{definition}"}
    bundle = BASE + "definitions:v2"
    yield "definitions-v2.schema.json", {"$schema": DRAFT, "$id": bundle,
        "title": "Hedging backtest wire definitions v2", "$defs": current_specification()}
    for name, definition in CURRENT_DOCUMENTS.items():
        yield f"{name}.schema.json", {"$schema": DRAFT, "$id": BASE + name,
            "title": f"Hedging backtest {name}", "$ref": f"{bundle}#/$defs/{definition}"}
    bundle = BASE + "definitions:v3"
    yield "definitions-v3.schema.json", {"$schema": DRAFT, "$id": bundle,
        "title": "Hedging backtest wire definitions v3", "$defs": market_shock_specification()}
    for name, definition in MARKET_SHOCK_DOCUMENTS.items():
        yield f"{name}.schema.json", {"$schema": DRAFT, "$id": BASE + name,
            "title": f"Hedging backtest {name}", "$ref": f"{bundle}#/$defs/{definition}"}

    bundle = BASE + "definitions:v4"
    yield "definitions-v4.schema.json", {"$schema":DRAFT,"$id":bundle,
        "title":"Hedging backtest wire definitions v4","$defs":path_specification()}
    for name,definition in PATH_DOCUMENTS.items():
        yield f"{name}.schema.json", {"$schema":DRAFT,"$id":BASE+name,
            "title":f"Hedging backtest {name}","$ref":f"{bundle}#/$defs/{definition}"}

    bundle = BASE + "definitions:v5"
    yield "definitions-v5.schema.json", {"$schema":DRAFT,"$id":bundle,
        "title":"Hedging backtest wire definitions v5","$defs":joint_specification()}
    for name,definition in JOINT_DOCUMENTS.items():
        yield f"{name}.schema.json", {"$schema":DRAFT,"$id":BASE+name,
            "title":f"Hedging backtest {name}","$ref":f"{bundle}#/$defs/{definition}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, document in artifacts():
        payload = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        path = DESTINATION / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != payload:
                parser.exit(1, f"stale or missing backtest schema: {path}\n")
        else:
            DESTINATION.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
    print("Backtest schemas match the explicit specification" if args.check else "Generated backtest schema files")


if __name__ == "__main__":
    main()
