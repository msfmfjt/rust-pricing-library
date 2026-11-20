import PricingAudit.Barrier
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Tactic.FieldSimp

/-! For a zero-carry, zero-volatility up-and-out call with no rebate, the
initial and terminal observations have the same spot. The deterministic bridge
contributes one in the open transition band. -/
namespace PricingAudit.Repair
open Filter Topology

theorem quintic_hasDerivAt (u : ℝ) :
    HasDerivAt (fun t : ℝ => quintic t) (30 * u ^ 2 * (1 - u) ^ 2) u := by
  have hx : HasDerivAt (fun t : ℝ => t) 1 u := hasDerivAt_id u
  convert ((hx.mul hx).mul hx).mul ((hx.mul ((hx.const_mul 6).const_add (-15))).const_add 10)
    using 1
  ring

/-- Delta of the actual piecewise payoff on the ITM open transition band.
v is the reflected transition coordinate and q is the survival factor per endpoint. -/
theorem zero_vol_up_out_delta (h barrier strike s : ℝ) (hh : 0 < h) (itm : strike < s)
    (lo : barrier - h < s) (hi : s < barrier + h) :
    let v := (barrier - s + h) / (2 * h)
    let q := quintic v
    HasDerivAt (fun z : ℝ => max (z - strike) 0 * (1 - indicator h (z - barrier)) ^ 2)
      (q ^ 2 - 30 * (s - strike) * q * v ^ 2 * (1 - v) ^ 2 / h) s := by
  let v := fun z : ℝ => (barrier - z + h) / (2 * h)
  have hs : HasDerivAt (fun z : ℝ => z) 1 s := hasDerivAt_id' s
  have hv : HasDerivAt v (-1 / (2 * h)) s := by
    exact HasDerivAt.div_const (HasDerivAt.add_const h (HasDerivAt.const_sub barrier hs)) (2 * h)
  have hq := (quintic_hasDerivAt (v s)).comp s hv
  have hpay := HasDerivAt.mul (HasDerivAt.sub_const strike hs) (hq.pow 2)
  have hlocal :
      (fun z : ℝ => max (z - strike) 0 * (1 - indicator h (z - barrier)) ^ 2) =ᶠ[𝓝 s]
        (fun z : ℝ => (z - strike) * quintic (v z) ^ 2) := by
    filter_upwards [lt_mem_nhds itm, lt_mem_nhds lo, gt_mem_nhds hi] with z hK hL hU
    rw [max_eq_left (by linarith), indicator_complement hh, neg_sub]
    simp [indicator, v, show ¬ barrier - z ≤ -h by linarith,
      show ¬ h ≤ barrier - z by linarith]
  apply HasDerivAt.congr_of_eventuallyEq _ hlocal
  convert hpay using 1
  dsimp [v, Function.comp_def]
  field_simp [ne_of_gt hh]
  ring

end PricingAudit.Repair
