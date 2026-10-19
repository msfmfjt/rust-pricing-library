"""Replay a versioned input using the installed native pricing library."""

import argparse
import json
from pathlib import Path

from .engine import run_backtest
from .rust_adapter import RustPricingProvider
from .serialization import load_input
from .named_variance_basis import POLICIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnostic-only", action="store_true",
                        help="inspect unvalidated native VegaKT; does not imply financial acceptance")
    mode.add_argument("--validated-p0", action="store_true",
                      help="use the numerically accepted terminal scope with a per-contract market calibration check")
    mode.add_argument("--validated-multi-step", action="store_true",
                      help="use four/six-interval conditional Gamma for terminal European contracts with checked smile calibration")
    mode.add_argument("--validated-path", action="store_true",
                      help="use checked Asian/Lookback histories and conditional Gamma on four/six intervals")
    mode.add_argument("--validated-extension", choices=("all_rows", "linear_time"),
                      help="use an explicit all-transition calibration extension for checked Call-side contracts")
    mode.add_argument("--validated-put-extension", choices=("all_rows", "linear_time"),
                      help="extend the checked all-transition scope to Put-side contracts and Call hedges")
    mode.add_argument("--validated-custom-basis", choices=POLICIES,
                      help="use a named accepted variance basis for Call/Put ledgers")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--rqmc-256x16", action="store_true",
                        help="explicit runtime-checked RQMC with all_rows or coupled_row_hats; input must specify 256 points and 16 scrambles")
    sampling.add_argument("--rqmc-path-16-scrambles", action="store_true",
                         help="runtime-checked RQMC: six-interval Asian uses 128 points, other products use 256, all use 16 scrambles")
    sampling.add_argument("--rqmc-quote-control", action="store_true",
                         help="fixed virtual ATM quote control for six-interval European calls with all_rows; input requires 512 points and 24 scrambles")
    parser.add_argument("--worker-threads", type=int, default=1, help="native pricing workers (also bound into checkpoints)")
    args = parser.parse_args()
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8")) if args.checkpoint else None
    result = run_backtest(**load_input(args.input.read_text(encoding="utf-8")),
                          provider=RustPricingProvider(diagnostic_only=args.diagnostic_only,
                              worker_threads=args.worker_threads,
                              numerical_scope=("p0_terminal_replication" if args.validated_p0 else
                                               "multi_step_gamma" if args.validated_multi_step else
                                               "multi_step_path_gamma" if args.validated_path else
                                               "multi_step_extension_gamma" if args.validated_extension else
                                               "multi_step_extension_put_gamma" if args.validated_put_extension else
                                               "multi_step_custom_basis_gamma" if args.validated_custom_basis else None),
                              variance_extension=args.validated_extension or args.validated_put_extension,
                              variance_basis_policy=args.validated_custom_basis,
                              sampling_policy=("rqmc_quote_control_512x24" if args.rqmc_quote_control else
                                               "rqmc_256x16" if args.rqmc_256x16 else
                                               "rqmc_path_16_scrambles" if args.rqmc_path_16_scrambles else None)), checkpoint=checkpoint,
                          on_error="return")
    result.write(args.output)
    print(json.dumps({**result.summary, "failure": result.failure}, indent=2, allow_nan=False))
    return 1 if result.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
