"""Size USD/EUR option trades and independently reprice the April 10 JPY hedge."""
import argparse

import rust_pricing as rp
from cross_currency_autocall_quote_risk import QUOTES, seasoned_basket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note-quantity", type=float, default=1., help="signed number of notes")
    parser.add_argument("--shares-per-contract", type=float, default=100.)
    args = parser.parse_args()
    engine = rp.Engine.randomized_quasi_monte_carlo(1024, 29737, scramble_count=4,
                                                     antithetic=True, brownian_bridge=True)
    sizes = {stock: (args.shares_per_contract, args.shares_per_contract) for stock in QUOTES}
    hedge = seasoned_basket().bergomi_quote_hedge(
        QUOTES, engine, note_quantity=args.note_quantity, contract_sizes=sizes, whole_contracts=True,
        bump=.005, replications=8, worker_threads=4)
    print(f"2026-04-10  notes {args.note_quantity:+g}  shares/contract {args.shares_per_contract:g}")
    print("Positive quantities buy options. All vegas are JPY per volatility point.")
    before = hedge.unhedged.vegas
    fractional, whole = hedge.theoretical_hedged.vegas, hedge.hedged.vegas
    for stock, ccy in [(10, "USD"), (20, "EUR")]:
        for j, (expiry, side, strike, _, _) in enumerate(QUOTES[stock]):
            q = hedge.quantities[stock][j]
            theoretical = hedge.theoretical_quantities[stock][j]
            sizing_se = hedge.quantity_standard_errors[stock][j]
            unit = hedge.unit_vegas[stock][j]
            old = before[stock][j]
            ideal, actual = fractional[stock][j], whole[stock][j]
            print(f"{ccy} {expiry} {side:4} K={strike:g}  unit vega {unit:.2f}  "
                  f"theoretical {theoretical:+.5f} contracts (SE {sizing_se:.5f}, {theoretical*args.shares_per_contract:+.4f} shares)  "
                  f"whole trade {q:+.0f} contracts  "
                  f"before {old.value:+.2f} (SE {old.standard_error:.2f})  "
                  f"fractional residual {ideal.value:+.2f} (SE {ideal.standard_error:.2f})  "
                  f"whole residual {actual.value:+.2f} (SE {actual.standard_error:.2f})")


if __name__ == "__main__":
    main()
