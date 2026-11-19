import PricingAudit.Market
import Mathlib.Data.Real.Pi.Bounds

noncomputable section
namespace PricingAudit.Market
open MeasureTheory

/-- A quantitative lower bound for both unit ATM Black-76 options. -/
theorem black_atm_lower_bound (s : ℝ) (hs : 0 < s) (hmax : s ≤ 1) : s/8 ≤ atmBlack 1 1 1 s := by
  let density := fun x : ℝ => Real.exp (-x^2/2) / Real.sqrt (2*Real.pi)
  have hc : Continuous density := by dsimp [density]; fun_prop
  have sq := Real.sq_sqrt (show 0 ≤ 2*Real.pi by positivity)
  have hr : 0 < Real.sqrt (2*Real.pi) := by positivity
  have small : Real.sqrt (2*Real.pi) ≤ 4 := by nlinarith [Real.pi_lt_four]
  have bound (x : ℝ) (hx : x ∈ Set.Icc (-(s/2)) (s/2)) : (1/8 : ℝ) ≤ density x := by
    obtain ⟨lo,hi⟩ := hx
    have xa : 0 ≤ x + 1/2 := by linarith
    have xb : 0 ≤ 1/2 - x := by linarith
    have e := Real.add_one_le_exp (-x^2/2)
    apply (le_div_iff₀ hr).mpr
    nlinarith [mul_nonneg xa xb]
  have hle : -(s/2) ≤ s/2 := by linarith
  have h := intervalIntegral.integral_mono_on (μ := volume) hle (intervalIntegrable_const (c := (1/8 : ℝ)))
    (hc.intervalIntegrable _ _) bound
  rw [intervalIntegral.integral_const, smul_eq_mul] at h
  simp only [atmBlack, one_mul]
  dsimp [density] at h
  linarith

theorem black_witness_is_normal_scale : (2:ℝ)^(-63:ℤ) ≤ atmBlack 1 1 1 ((2:ℝ)^(-60:ℤ)) := by
  have h := black_atm_lower_bound ((2:ℝ)^(-60:ℤ)) (by norm_num) (by norm_num)
  convert h using 1
  norm_num

end PricingAudit.Market
