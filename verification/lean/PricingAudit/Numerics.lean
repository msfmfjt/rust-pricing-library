import PricingAudit.Barrier
import Mathlib.Tactic.FieldSimp

noncomputable section
namespace PricingAudit.Numerics

/-- Sufficient statistics stored by the one-pass centered reduction. -/
structure Moments where
  count : ℕ
  mean : ℝ
  m2 : ℝ

def total (s : Moments) := (s.count : ℝ) * s.mean
def sumSq (s : Moments) := s.m2 + (s.count : ℝ) * s.mean ^ 2

def push (s : Moments) (x : ℝ) : Moments :=
  let m := s.mean + (x - s.mean) / ((s.count : ℝ) + 1)
  ⟨s.count + 1, m, s.m2 + (x - s.mean) * (x - m)⟩

theorem welford_push (s : Moments) (x : ℝ) :
    total (push s x) = total s + x ∧ sumSq (push s x) = sumSq s + x ^ 2 := by
  have h : (s.count : ℝ) + 1 ≠ 0 := by positivity
  dsimp [push, total, sumSq]
  push_cast
  constructor <;> field_simp [h] <;> ring

theorem welford_fold (xs : List ℝ) (s : Moments) :
    (xs.foldl push s).count = s.count + xs.length ∧
    total (xs.foldl push s) = total s + xs.sum ∧
    sumSq (xs.foldl push s) = sumSq s + (xs.map (· ^ 2)).sum := by
  induction xs generalizing s with
  | nil => simp
  | cons x xs ih =>
    obtain ⟨hc, ht, hq⟩ := ih (push s x)
    obtain ⟨ht', hq'⟩ := welford_push s x
    simp only [List.foldl_cons, List.sum_cons, List.map_cons, List.length_cons]
    refine ⟨?_, ?_, ?_⟩
    · rw [hc]; dsimp [push]; omega
    · rw [ht, ht']; ring
    · rw [hq, hq']; ring

def joinedMean (n m x y : ℝ) := x + (y - x) * m / (n + m)
def between (n m x y : ℝ) := (y - x) ^ 2 * (n * m / (n + m))

theorem chan_conserves_raw_moments (n m x y a b : ℝ) (hn : 0 < n) (hm : 0 < m) :
    (n+m) * joinedMean n m x y = n*x + m*y ∧
    a+b+between n m x y + (n+m) * (joinedMean n m x y)^2 =
      (a+n*x^2) + (b+m*y^2) := by
  have h : n + m ≠ 0 := by positivity
  dsimp [joinedMean, between]
  constructor <;> field_simp [h] <;> ring

theorem chan_between_nonnegative (n m x y : ℝ) (hn : 0 ≤ n) (hm : 0 ≤ m) :
    0 ≤ between n m x y := by unfold between; positivity

theorem covariance_merge (n m x y u v a b : ℝ) (hn : 0 < n) (hm : 0 < m) :
    a+b+(u-x)*(v-y)*(n*m/(n+m)) + (n+m)*joinedMean n m x u*joinedMean n m y v =
    (a+n*x*y)+(b+m*u*v) := by
  have h : n + m ≠ 0 := by positivity
  unfold joinedMean
  field_simp [h]
  ring

theorem compensated_residual (s x rounded : ℝ) :
    rounded + (if |x| ≤ |s| then (s-rounded)+x else (x-rounded)+s) = s+x := by
  split <;> ring

def rule (f : ℝ → ℝ) (a b : ℝ) := (b-a)*(f a + 4*f ((a+b)/2) + f b)/6

theorem simpson_cubic (a b c₀ c₁ c₂ c₃ : ℝ) :
    let f := fun x : ℝ => c₀ + c₁*x + c₂*x^2 + c₃*x^3
    let F := fun x : ℝ => c₀*x + c₁*x^2/2 + c₂*x^3/3 + c₃*x^4/4
    rule f a b = F b - F a := by dsimp [rule]; ring

theorem richardson_cancels_leading_error (truth e : ℝ) :
    (truth+e/16) + ((truth+e/16)-(truth+e))/15 = truth := by ring

theorem complex_division_reconstructs (a b x y : ℝ) (h : x^2 + y^2 ≠ 0) :
    let re := (a*x+b*y)/(x^2+y^2)
    let im := (b*x-a*y)/(x^2+y^2)
    re*x-im*y = a ∧ re*y+im*x = b := by
  dsimp
  constructor <;> field_simp [h] <;> ring

theorem complex_norm_product (a b x y : ℝ) :
    (a*x-b*y)^2 + (a*y+b*x)^2 = (a^2+b^2)*(x^2+y^2) := by ring

theorem equal_components_div_one_i (x : ℝ) :
    (x*1+x*1)/(1^2+1^2) = x ∧ (x*1-x*1)/(1^2+1^2) = 0 := by
  constructor <;> ring

theorem correlation_pivot (rho : ℝ) (lo : -1 ≤ rho) (hi : rho ≤ 1) :
    0 ≤ 1-rho*rho := by nlinarith [mul_nonneg (by linarith : 0 ≤ 1-rho) (by linarith : 0 ≤ 1+rho)]

theorem three_driver_determinant (a b r : ℝ) :
    1 + 2*a*b*r-a^2-b^2-r^2 = (1-a^2)*(1-b^2)-(r-a*b)^2 := by ring

theorem least_squares_normal_direction (j₁ j₂ x y lam : ℝ) (damping : 0 < lam)
    (nonzero : x ≠ 0 ∨ y ≠ 0) :
    0 < (j₁*x+j₂*y)^2 + lam*(x^2+y^2) := by
  have hs : 0 < x^2+y^2 := by
    rcases nonzero with hx | hy
    · nlinarith [sq_pos_of_ne_zero hx, sq_nonneg y]
    · nlinarith [sq_pos_of_ne_zero hy, sq_nonneg x]
  positivity

end PricingAudit.Numerics
