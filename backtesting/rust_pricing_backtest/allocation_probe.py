"""Diagnose initial-time loss using first-local-row CRN bumps, not market-IV bumps."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .domain import fingerprint
from .numerics import flat_request, threshold_check


def run_probe(policy, base):
    import rust_pricing as rp
    observations, checks, not_applicable = [], [], []

    def evaluate(request):
        return rp.PricingPlan.compile(request, worker_threads=1, reduction_block_size=256).evaluate()

    for seed in policy["seeds"]:
        for steps in policy["time_steps"]:
            kwargs = dict(seed=seed, samples=policy["sampling_units"], steps=steps, nodes=policy["strike_nodes"])
            request = flat_request(base, **kwargs)
            result = evaluate(request)
            kt = result.vega_kt
            if kt.policy_label != "local_vol_node_allocation_diagnostic_v1":
                not_applicable.append({**kwargs, "policy_label": kt.policy_label,
                    "reason": "Initial-local-row loss is a property of the legacy local-node allocation; this method has a different residual definition."})
                continue
            price_payload = json.loads(flat_request(base, **kwargs, risk=False).to_json())
            for bump in policy["local_volatility_bumps"]:
                prices, hashes = [], []
                for sign in (-1, 1):
                    payload = json.loads(json.dumps(price_payload))
                    grid = payload["model"]["local_variance_grid"]
                    # IV values, every positive-time local row and random
                    # coordinates stay fixed. Only the T=0 local-vol row moves.
                    for i in range(policy["strike_nodes"]):
                        grid["values"][i] = (math.sqrt(grid["values"][i])+sign*bump)**2
                    hashes.append(fingerprint(payload))
                    prices.append(evaluate(rp.PricingRequest.from_json(json.dumps(payload))).value)
                initial_time_crn = (prices[1]-prices[0])/(2*bump)
                residual = kt.projection.signed_residual
                observation = {**kwargs, "bump": bump, "initial_time_crn": initial_time_crn,
                               "reported_residual": residual, "bucket_sum": math.fsum(kt.raw_buckets),
                               "local_vega": result.vega_raw, "policy_label": kt.policy_label,
                               "truncation_order": kt.truncation_order,
                               "request_hash": fingerprint(json.loads(request.to_json())), "bump_hashes": hashes}
                observations.append(observation)
                checks.append(threshold_check("residual_matches_first_local_row_crn", abs(residual-initial_time_crn),
                    policy["absolute_tolerance"] + policy["relative_tolerance"]*abs(initial_time_crn),
                    seed=seed, steps=steps, bump=bump))
                checks.append({"check": "honest_native_method_label", "seed": seed, "steps": steps, "bump": bump,
                               "passed": kt.policy_label == "local_vol_node_allocation_diagnostic_v1"
                               and kt.truncation_order == "not_applicable"})
    module = getattr(rp, "rust_pricing", rp)
    status = ("failed" if any(not c["passed"] for c in checks) else
              "incomplete" if checks and not_applicable else
              "not_applicable" if not_applicable else "passed" if checks else "incomplete")
    return {"document_kind": "local_allocation_initial_time_probe", "schema_version": 2,
            "status": status, "not_applicable": not_applicable,
            "paper_vega_kt_acceptance": "not_established", "policy": policy, "base_policy": base,
            "policy_hash": fingerprint(policy), "base_policy_hash": fingerprint(base),
            "runner_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "native_binary_hash": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "observations": observations, "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    base = json.loads((args.policy.parent/policy["base_policy"]).read_text(encoding="utf-8"))
    report = run_probe(policy, base)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                      "failed": sum(not c["passed"] for c in report["checks"]),
                      "not_applicable_cases": len(report["not_applicable"]),
                      "paper_vega_kt_acceptance": report["paper_vega_kt_acceptance"]}))
    return 0 if report["status"] == "passed" else 1 if report["status"] == "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
