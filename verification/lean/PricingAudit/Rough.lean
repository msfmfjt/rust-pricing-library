import PricingAudit.Dynamics
import Mathlib.Analysis.SpecialFunctions.Log.Basic

noncomputable section
namespace PricingAudit.Rough

theorem truncation_nonnegative (raw : ℝ) :
    0 ≤ max raw 0 ∧ (max raw 0 = raw ↔ 0 ≤ raw) := ⟨le_max_right _ _, max_eq_left_iff⟩

def lifted (factor common rate dt : ℝ) := (factor + common) / (1 + rate * dt)

theorem lifted_nonexpansive (a b common rate dt : ℝ) (hr : 0 ≤ rate) (ht : 0 ≤ dt) :
    |lifted a common rate dt - lifted b common rate dt| ≤ |a-b| := by
  have hd : 0 < 1 + rate * dt := by nlinarith [mul_nonneg hr ht]
  unfold lifted
  rw [← sub_div, show (a + common) - (b + common) = a-b by ring,
    abs_div, abs_of_pos hd, div_le_iff₀ hd]
  nlinarith [mul_nonneg (abs_nonneg (a-b)) (mul_nonneg hr ht)]

def quadratic (a shift floor z : ℝ) := a * (z-shift)^2 + floor

theorem quadratic_floor (a shift floor z : ℝ) (ha : 0 ≤ a) :
    floor ≤ quadratic a shift floor z ∧ quadratic a shift floor shift = floor := by
  constructor
  · exact le_add_of_nonneg_left (mul_nonneg ha (sq_nonneg _))
  · simp [quadratic]

theorem quadratic_derivative (a shift floor z : ℝ) :
    HasDerivAt (quadratic a shift floor) (2*a*(z-shift)) z := by
  convert ((((hasDerivAt_id' z).sub_const shift).pow 2).const_mul a).add_const floor using 1
  simp [quadratic]
  ring

def component (eta x q : ℝ) := Real.exp (eta*x - (1/2)*eta^2*q)

theorem log_component_formula (xi weight eta x q : ℝ) (hx : 0 < xi) (hw : 0 < weight) :
    Real.exp (Real.log xi + Real.log weight + eta*x - (1/2)*eta^2*q) =
      xi * weight * component eta x q := by
  rw [show Real.log xi + Real.log weight + eta*x - (1/2)*eta^2*q =
    (Real.log xi + Real.log weight) + (eta*x - (1/2)*eta^2*q) by ring,
    Real.exp_add, Real.exp_add, Real.exp_log hx, Real.exp_log hw]
  rfl

def mixed {n : ℕ} (xi x q : ℝ) (weight eta : Fin n → ℝ) :=
  xi * ∑ i, weight i * component (eta i) x q

theorem mixed_nonnegative {n : ℕ} (xi x q : ℝ) (weight eta : Fin n → ℝ)
    (hx : 0 ≤ xi) (hw : ∀ i, 0 ≤ weight i) : 0 ≤ mixed xi x q weight eta :=
  mul_nonneg hx (Finset.sum_nonneg fun i _ => mul_nonneg (hw i) (Real.exp_pos _).le)

theorem mixed_flat_vol_of_vol {n : ℕ} (xi x q : ℝ) (weight : Fin n → ℝ)
    (hw : ∑ i, weight i = 1) : mixed xi x q weight (fun _ => 0) = xi := by
  simp [mixed, component, hw]

theorem scalar_variance_positive (xi eta x q : ℝ) (hx : 0 < xi) :
    0 < xi * component eta x q := mul_pos hx (Real.exp_pos _)

end PricingAudit.Rough
