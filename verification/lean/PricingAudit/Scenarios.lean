import PricingAudit.Dynamics

noncomputable section
namespace PricingAudit.Scenarios

open Finset

def value {m : ℕ} (weight payoff : Fin m → ℝ) : ℝ := ∑ w, weight w * payoff w

theorem price_monotone {m : ℕ} (weight a b : Fin m → ℝ)
    (hp : ∀ w, 0 ≤ weight w) (hab : ∀ w, a w ≤ b w) : value weight a ≤ value weight b :=
  Finset.sum_le_sum fun w _ => mul_le_mul_of_nonneg_left (hab w) (hp w)

theorem price_linear {m : ℕ} (weight a b : Fin m → ℝ) (alpha beta : ℝ) :
    value weight (fun w => alpha*a w + beta*b w) =
      alpha*value weight a + beta*value weight b := by
  simp only [value, Finset.mul_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun w _ => by ring

/-- Finite-scenario L² error of a mean, given the common centered second moment
and zero cross moments. This applies to any sample size greater than zero. -/
theorem mean_square_error {m n : ℕ} (hn : 0 < n) (p : Fin m → ℝ)
    (z : Fin n → Fin m → ℝ) (v : ℝ)
    (cov : ∀ i j, (∑ w, p w*z i w*z j w) = if i = j then v else 0) :
    (∑ w, p w * ((∑ i, z i w)/(n:ℝ))^2) = v / n := by
  have expand (w : Fin m) : p w * ((∑ i, z i w)/(n:ℝ))^2 =
      (∑ i, ∑ j, p w*z i w*z j w)/(n:ℝ)^2 := by
    have factor : (∑ i, ∑ j, p w*z i w*z j w) =
        p w*((∑ i, z i w)*(∑ i, z i w)) := by
      simp_rw [← Finset.mul_sum]
      rw [← Finset.sum_mul, ← Finset.mul_sum, mul_assoc]
    rw [factor, div_pow, ← pow_two (∑ i, z i w), mul_div_assoc']
  have swap : (∑ w, ∑ i, ∑ j, p w*z i w*z j w) =
      ∑ i, ∑ j, ∑ w, p w*z i w*z j w := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ => Finset.sum_comm
  have hnr : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  simp_rw [expand]
  rw [← Finset.sum_div, swap]
  simp [cov]
  field_simp [hnr]
  ring

/-- Markov's inequality applied to squared error on the actual finite scenario set. -/
theorem tail_mass_bound {m : ℕ} (weights err : Fin m → ℝ) (hp : ∀ w, 0 ≤ weights w)
    (epsilon : ℝ) (he : 0 < epsilon) :
    epsilon^2 * (∑ w, if epsilon ≤ |err w| then weights w else 0) ≤
      ∑ w, weights w * (err w)^2 := by
  rw [Finset.mul_sum]
  apply Finset.sum_le_sum
  intro w _
  split_ifs with hit
  · have hs := mul_self_le_mul_self he.le hit
    rw [abs_mul_abs_self] at hs
    simpa only [pow_two, mul_comm (weights w)] using mul_le_mul_of_nonneg_right hs (hp w)
  · simp only [mul_zero]
    exact mul_nonneg (hp w) (sq_nonneg _)

theorem dual_gap_grows (candidates : List ℝ) (gap : ℝ) :
    gap ≤ candidates.foldl max gap := by
  induction candidates generalizing gap with
  | nil => exact le_rfl
  | cons x xs ih => exact le_trans (le_max_left gap x) (ih (max gap x))

/-- Before the first exercise, the cumulative correction is zero. On exercise
the new candidate is therefore zero, including the case of no prior candidates. -/
theorem dual_upper_above_policy (prior : Option ℝ) (later : List ℝ) (payoff lower : ℝ) :
    let candidate := (payoff - payoff) - 0
    let gap := prior.elim candidate (fun g => max g candidate)
    lower ≤ lower + later.foldl max gap := by
  simp only [sub_self, sub_zero, le_add_iff_nonneg_right]
  cases prior with
  | none => exact dual_gap_grows later 0
  | some g => exact le_trans (le_max_right g 0) (dual_gap_grows later (max g 0))

end PricingAudit.Scenarios
