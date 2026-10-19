"""Create a PNL-04 companion report from an existing backtest result."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backtesting'))
from rust_pricing_backtest import BacktestResult
from rust_pricing_backtest.portfolio_market_risk import explain_backtest
from rust_pricing_backtest.domain import BacktestError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--extension',choices=['all_rows','linear_time'],default='all_rows')
    parser.add_argument('--workers',type=int,default=1)
    a=parser.parse_args()
    protected={a.result.resolve(),*(p.resolve() for folder in ('backtesting','scripts','crates')
        for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.rs'))}
    outputs=[a.output,a.output.with_suffix(a.output.suffix+'.tmp')]
    if any(p.resolve() in protected or (p.exists() and any(p.samefile(q) for q in protected)) for p in outputs):
        parser.error('output must not overwrite input or implementation')
    try:
        result=BacktestResult.from_json(a.result.read_text())
        report=explain_backtest(result,extension=a.extension,worker_threads=a.workers)
    except (BacktestError,ValueError) as error:
        print(str(error),file=sys.stderr);return 1
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=outputs[1];temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(a.output)
    print(json.dumps(dict(status=report['status'],covered_intervals=report['covered_intervals'],intervals=report['intervals'])))
    return 0
if __name__=='__main__':raise SystemExit(main())
