import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.List.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-! Real-arithmetic specification of the compact indicator, the discrete
hit-state update, and the unresolved continuous-barrier payoff. Source locations
are pinned in the accompanying report. -/
namespace PricingAudit

/-- The order of operations follows the Rust polynomial. -/
def quintic {K : Type*} [CommRing K] (u : K) : K :=
  let u2 := u * u
  let u3 := u2 * u
  u3 * (10 + u * (-15 + 6 * u))

theorem quintic_complement {K : Type*} [CommRing K] (u : K) :
    1 - quintic u = quintic (1 - u) := by
  unfold quintic
  ring

private theorem quintic_factor (u : ℝ) :
    quintic u = u ^ 3 * (1 + 3 * (1 - u) + 6 * (1 - u) ^ 2) := by
  unfold quintic
  ring

private theorem quintic_nonneg {u : ℝ} (hu : 0 ≤ u) (hv : u ≤ 1) :
    0 ≤ quintic u := by
  rw [quintic_factor]
  have : 0 ≤ 1 - u := sub_nonneg.mpr hv
  positivity

theorem quintic_range {u : ℝ} (hu : 0 ≤ u) (hv : u ≤ 1) :
    0 ≤ quintic u ∧ quintic u ≤ 1 := by
  refine ⟨quintic_nonneg hu hv, ?_⟩
  have h := quintic_nonneg (u := 1 - u) (by linarith) (by linarith)
  rw [← quintic_complement] at h
  linarith

/-- The complement is strictly positive even arbitrarily close to the upper edge. -/
theorem safety_pos_inside {u : ℝ} (hu : 0 < u) (hv : u < 1) :
    0 < 1 - quintic u := by
  rw [quintic_complement, quintic_factor]
  have : 0 < 1 - u := sub_pos.mpr hv
  have : 0 < 1 - (1 - u) := by linarith
  positivity

/-- Both floating-point normalization branches represent this same real number. -/
noncomputable def indicator (h x : ℝ) : ℝ :=
  if x ≤ -h then 0 else if h ≤ x then 1 else quintic ((x + h) / (2 * h))

theorem indicator_range {h : ℝ} (hh : 0 < h) (x : ℝ) :
    0 ≤ indicator h x ∧ indicator h x ≤ 1 := by
  unfold indicator
  split
  · exact ⟨le_rfl, zero_le_one⟩
  · rename_i hleft
    split
    · exact ⟨zero_le_one, le_rfl⟩
    · rename_i hright
      apply quintic_range
      · exact div_nonneg (by linarith) (by linarith)
      · apply (div_le_iff₀ (by linarith : 0 < 2 * h)).mpr
        linarith

/-- A survival factor can be evaluated on the reflected distance. -/
theorem indicator_complement {h : ℝ} (hh : 0 < h) (x : ℝ) :
    1 - indicator h x = indicator h (-x) := by
  by_cases lo : x ≤ -h
  · have opposite : ¬ -x ≤ -h := by linarith
    simp [indicator, lo, opposite, show h ≤ -x by linarith]
  by_cases hi : h ≤ x
  · simp [indicator, lo, hi, show -x ≤ -h by linarith]
  have oppLo : ¬ -x ≤ -h := by linarith
  have oppHi : ¬ h ≤ -x := by linarith
  simp only [indicator, if_neg lo, if_neg hi, if_neg oppLo, if_neg oppHi]
  rw [quintic_complement, one_sub_div (by linarith : 2 * h ≠ 0)]
  congr 2
  ring

/-- SourceOpcode::BarrierHitState uses the previous state and one new weight. -/
def hitStep (previous weight : ℝ) : ℝ := previous + (1 - previous) * weight

theorem hitStep_bounds {previous weight : ℝ}
    (hp : 0 ≤ previous) (hp' : previous ≤ 1)
    (hw : 0 ≤ weight) (hw' : weight ≤ 1) :
    0 ≤ hitStep previous weight ∧ previous ≤ hitStep previous weight ∧
      hitStep previous weight ≤ 1 := by
  unfold hitStep
  have increase := mul_nonneg (sub_nonneg.mpr hp') hw
  have remaining := mul_nonneg (sub_nonneg.mpr hp') (sub_nonneg.mpr hw')
  exact ⟨by linarith, by linarith, by nlinarith⟩

theorem historical_hit_absorbing (weights : List ℝ) :
    weights.foldl hitStep 1 = 1 := by
  induction weights with
  | nil => rfl
  | cons w ws ih => simpa [hitStep] using ih

theorem hitFold_bounds (weights : List ℝ)
    (hw : ∀ w ∈ weights, 0 ≤ w ∧ w ≤ 1) {initial : ℝ}
    (hi : 0 ≤ initial) (hi' : initial ≤ 1) :
    initial ≤ weights.foldl hitStep initial ∧ weights.foldl hitStep initial ≤ 1 := by
  induction weights generalizing initial with
  | nil => exact ⟨le_rfl, hi'⟩
  | cons w ws ih =>
    obtain ⟨hw₀, hw₁⟩ := hw w (by simp)
    obtain ⟨h₀, hinc, h₁⟩ := hitStep_bounds hi hi' hw₀ hw₁
    obtain ⟨hlow, hhigh⟩ := ih (fun t ht => hw t (by simp [ht])) h₀ h₁
    exact ⟨hinc.trans hlow, hhigh⟩

theorem smoothed_path_bounds {h initial : ℝ} (hh : 0 < h) (hi : 0 ≤ initial)
    (hi' : initial ≤ 1) (distances : List ℝ) :
    initial ≤ (distances.map (indicator h)).foldl hitStep initial ∧
      (distances.map (indicator h)).foldl hitStep initial ≤ 1 := by
  apply hitFold_bounds _ _ hi hi'
  intro w hw
  obtain ⟨x, _, rfl⟩ := List.mem_map.mp hw
  exact indicator_range hh x

theorem survival_product (weights : List ℝ) (initial : ℝ) :
    1 - weights.foldl hitStep initial =
      (1 - initial) * (weights.map (fun w => 1 - w)).prod := by
  induction weights generalizing initial with
  | nil => simp
  | cons w ws ih =>
    simp only [List.foldl_cons, List.map_cons, List.prod_cons]
    rw [ih]
    unfold hitStep
    ring

/-- Terminal, undiscounted cashflows for the same vanilla payoff and expiry rebate. -/
def knockOut (vanilla rebate survival : ℝ) : ℝ :=
  rebate + survival * (vanilla - rebate)

def knockIn (vanilla rebate survival : ℝ) : ℝ :=
  vanilla + survival * (rebate - vanilla)

theorem rebate_parity (vanilla rebate survival : ℝ) :
    knockIn vanilla rebate survival + knockOut vanilla rebate survival = vanilla + rebate := by
  unfold knockIn knockOut
  ring

end PricingAudit
