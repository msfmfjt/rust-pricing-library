import PricingAudit.Market
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Tactic.FieldSimp

noncomputable section
namespace PricingAudit.Dynamics

def bondLoading (a t : ℝ) := if a=0 then t else (1-Real.exp (-a*t))/a

theorem hull_white_loading_composes (a s t : ℝ) :
    bondLoading a (s+t) = bondLoading a s + Real.exp (-a*s)*bondLoading a t := by
  by_cases ha : a = 0
  · simp [bondLoading, ha]
  · simp only [bondLoading, if_neg ha]
    rw [show -a*(s+t) = -a*s + -a*t by ring, Real.exp_add]
    field_simp [ha]
    ring

theorem hull_white_positive_loading (a t : ℝ) (ha : 0 ≤ a) (ht : 0 < t) : 0 < bondLoading a t := by
  rcases ha.eq_or_lt with eq | pos
  · simp [bondLoading, ← eq, ht]
  · have e : Real.exp (-a*t) < 1 := Real.exp_lt_one_iff.mpr (by nlinarith [mul_pos pos ht])
    simp only [bondLoading, if_neg (ne_of_gt pos)]
    exact div_pos (sub_pos.mpr e) pos

theorem ou_loadings (dt var cov : ℝ) (hd : 0 < dt) (psd : cov^2 ≤ var*dt) :
    let l := cov/Real.sqrt dt
    let r := Real.sqrt (max (var-l^2) 0)
    Real.sqrt dt*l = cov ∧ l^2+r^2 = var := by
  have hs := Real.sqrt_pos.2 hd
  have hsq := Real.sq_sqrt hd.le
  have hn : (cov/Real.sqrt dt)^2 ≤ var := by
    rw [div_pow, hsq, div_le_iff₀ hd]; exact psd
  dsimp
  constructor
  · field_simp [ne_of_gt hs]
  · rw [max_eq_left (by linarith : 0 ≤ var-(cov/Real.sqrt dt)^2), Real.sq_sqrt (by linarith)]
    ring

theorem lsv_local_variance (target conditional : ℝ) (ht : 0 ≤ target) (hc : 0 < conditional) :
    (Real.sqrt (target/conditional))^2 * conditional = target := by
  rw [Real.sq_sqrt (div_nonneg ht hc.le)]
  field_simp [ne_of_gt hc]

theorem bridge_survival_range (left right variance : ℝ) (hl : 0 < left) (hr : 0 < right)
    (hv : 0 < variance) : 0 < 1-Real.exp (-(2*left*right/variance)) ∧
    1-Real.exp (-(2*left*right/variance)) < 1 := by
  have hp : 0 < 2*left*right/variance := by positivity
  have he := Real.exp_lt_one_iff.mpr (neg_neg_of_pos hp)
  have hpos := Real.exp_pos (-(2*left*right/variance))
  constructor <;> linarith

theorem bridge_hit_increases_with_variance (d v w : ℝ) (hd : 0 < d) (hv : 0 < v) (hw : v < w) :
    Real.exp (-d/v) < Real.exp (-d/w) := by
  apply Real.exp_lt_exp.mpr
  apply (div_lt_div_iff₀ hv (lt_trans hv hw)).mpr
  nlinarith [mul_pos hd (sub_pos.mpr hw)]

theorem log_survival_product (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    Real.exp (Real.log a + Real.log b) = a*b := by
  rw [Real.exp_add, Real.exp_log ha, Real.exp_log hb]

theorem whitened_covariance (r : ℝ) (lo : -1 ≤ r) (hi : r ≤ 1) :
    r^2+(Real.sqrt (1-r^2))^2=1 := by
  have h : 0 ≤ 1-r^2 := by nlinarith [mul_nonneg (by linarith : 0≤1-r) (by linarith : 0≤1+r)]
  rw [Real.sq_sqrt h]; ring

def stencil (f : ℝ → ℝ) (s h : ℝ) := ((f (s+h)-f s)+(f (s-h)-f s))/h^2

theorem gamma_quadratic (a b c s h : ℝ) (hh : h ≠ 0) :
    stencil (fun x => a*x^2+b*x+c) s h = 2*a := by
  unfold stencil
  field_simp [hh]
  ring

theorem gamma_quartic_bias (s h : ℝ) (hh : h ≠ 0) :
    stencil (· ^ 4) s h = 12*s^2+2*h^2 ∧
    stencil (· ^ 4) s (h/2) - stencil (· ^ 4) s h = -3*h^2/2 := by
  have half : h/2 ≠ 0 := div_ne_zero hh (by norm_num)
  unfold stencil
  constructor <;> field_simp [hh,half] <;> ring

theorem reverse_product_and_quotient (x y dx dy a : ℝ) (hy : y ≠ 0) :
    a*(dx*y+x*dy) = (a*y)*dx+(a*x)*dy ∧
    a*((dx*y-x*dy)/y^2) = (a/y)*dx+(-a*x/y^2)*dy := by
  constructor
  · ring
  · field_simp [hy]; ring

theorem exercise_envelope (immediate continuation : ℝ) :
    (if immediate > continuation then immediate else continuation) = max immediate continuation := by
  split <;> rename_i h
  · exact (max_eq_left h.le).symm
  · exact (max_eq_right (le_of_not_gt h)).symm

theorem pnl_reconciliation (delta gamma ds actual : ℝ) :
    let linear := delta*ds
    let curvature := (1/2)*gamma*ds*ds
    let residual := actual-(linear+curvature)
    linear+curvature+residual = actual := by dsimp; ring

theorem greek_pnl_exact_quadratic (a b c s shock : ℝ) :
    (a*(s+shock)^2+b*(s+shock)+c) - (a*s^2+b*s+c) =
    (2*a*s+b)*shock + (1/2)*(2*a)*shock^2 := by ring

end PricingAudit.Dynamics
