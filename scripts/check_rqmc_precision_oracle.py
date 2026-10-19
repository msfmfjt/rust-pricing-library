"""Refine the unchanged independent annual oracle from 2^16 to 2^18 points."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'backtesting')]
from check_portfolio_holdout import sha
from check_portfolio_market import write
from rust_pricing_backtest.domain import require,fingerprint


def run(original,output):
    require(not output.exists(),'precision_oracle_output','fresh output required')
    path=ROOT/'scripts/check_rqmc_annual_oracle.py';before={str(p):sha(p) for p in (Path(__file__),path,original)}
    spec=importlib.util.spec_from_file_location('_refined_annual_oracle',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old=module.policy_for
    module.policy_for=lambda name:dict(old(name),sobol_power=18)
    result=module.run(original,output)
    require(all(sha(p)==digest for p,digest in before.items()),'precision_oracle_source','oracle sources/input changed')
    result.pop('report_hash')
    result['refinement_driver_sha256']=before[str(Path(__file__))]
    result['refinement_scope']='Unchanged calibration/price/check kernel; same eight independent reference scrambles, four times as many points.'
    result['report_hash']=fingerprint(result);write(output,result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.input,a.output);raise SystemExit(r['status']!='passed')
