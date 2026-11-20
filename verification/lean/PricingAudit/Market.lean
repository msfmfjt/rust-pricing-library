import PricingAudit.Barrier
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Tactic.FieldSimp

noncomputable section
namespace PricingAudit.Market

open MeasureTheory

theorem forward_carry (s p q : ℝ) (hp : 0 < p) : p*(s*q/p) = s*q := by
  field_simp [ne_of_gt hp]

theorem intrinsic_parity (f k : ℝ) : max (f-k) 0 - max (k-f) 0 = f-k := by
  rcases le_total f k with h | h
  · rw [max_eq_right (by linarith), max_eq_left (by linarith)]; ring
  · rw [max_eq_left (by linarith), max_eq_right (by linarith)]; ring

theorem black_price_and_delta_parity (cdf : ℝ → ℝ) (symm : ∀ x, cdf (-x) = 1-cdf x)
    (d₁ d₂ f k df n : ℝ) :
    n*df*(f*cdf d₁-k*cdf d₂) - n*df*(k*cdf (-d₂)-f*cdf (-d₁)) = n*df*(f-k) ∧
    n*df*cdf d₁ - (-n*df*cdf (-d₁)) = n*df := by
  rw [symm, symm]
  constructor <;> ring

theorem black_vega_gamma (s sigma t q n density : ℝ) (hs : 0 < s)
    (hv : 0 < sigma) (ht : 0 < t) :
    n*s*q*density*Real.sqrt t =
      (n*q*density/(s*(sigma*Real.sqrt t)))*s^2*sigma*t := by
  have hr : 0 < Real.sqrt t := Real.sqrt_pos.2 ht
  field_simp [ne_of_gt hs, ne_of_gt hv, ne_of_gt hr]
  calc
    _ = (n*q*density*s^2*sigma) * (Real.sqrt t)^2 := by ring
    _ = _ := by rw [Real.sq_sqrt ht.le]

/-- Real ATM Black forward time value represented by the normal mass between d2 and d1. -/
def atmBlack (n df f sigmaT : ℝ) : ℝ :=
  n * df * f * ∫ x in (-(sigmaT/2))..(sigmaT/2),
    Real.exp (-x^2/2) / Real.sqrt (2*Real.pi)

theorem black_atm_time_value_positive (n df f sigmaT : ℝ) (hn : 0 < n) (hd : 0 < df)
    (hf : 0 < f) (hs : 0 < sigmaT) : 0 < atmBlack n df f sigmaT := by
  have hc : Continuous (fun x : ℝ => Real.exp (-x^2/2) / Real.sqrt (2*Real.pi)) := by
    fun_prop
  have pos := intervalIntegral.intervalIntegral_pos_of_pos
    (hc.intervalIntegrable (-(sigmaT/2)) (sigmaT/2))
    (fun x => div_pos (Real.exp_pos _) (Real.sqrt_pos.2 (by positivity))) (by linarith)
  exact mul_pos (mul_pos (mul_pos hn hd) hf) pos

def hermite (u h a b da db : ℝ) :=
  (2*u^3-3*u^2+1)*a+(u^3-2*u^2+u)*h*da+(-2*u^3+3*u^2)*b+(u^3-u^2)*h*db

def hermiteSlope (u h a b da db : ℝ) :=
  ((6*u^2-6*u)/h)*a+(3*u^2-4*u+1)*da+((-6*u^2+6*u)/h)*b+(3*u^2-2*u)*db

theorem pchip_interpolates_knots (h a b da db : ℝ) :
    hermite 0 h a b da db = a ∧ hermite 1 h a b da db = b ∧
    hermiteSlope 0 h a b da db = da ∧ hermiteSlope 1 h a b da db = db := by
  norm_num [hermite, hermiteSlope]

theorem pchip_reproduces_affine (u h a slope : ℝ) :
    hermite u h a (a+h*slope) slope slope = a + u*h*slope := by unfold hermite; ring

theorem durrleman_flat_slice (k w : ℝ) :
    (1 - k*0/(2*w))^2 - 0^2/4*(1/w+1/4) + 0/2 = 1 := by simp

theorem ssvi_atm_variance (theta rho phi : ℝ) :
    theta/2*(1+rho*phi*0+Real.sqrt ((phi*0+rho)^2+(1-rho^2))) = theta := by
  have h : (phi*0+rho)^2+(1-rho^2) = 1 := by ring
  rw [h, Real.sqrt_one]; ring

theorem bergomi_weight_variance (w r : ℝ) :
    (1-2*w)^2+2*w*(1-w)*(1+r) = (1-w)^2+w^2+2*r*w*(1-w) := by ring

theorem bergomi_normalized (w r z : ℝ) (hz : z ≠ 0)
    (norm : z^2 = (1-2*w)^2+2*w*(1-w)*(1+r)) :
    ((1-w)/z)^2+(w/z)^2+2*r*((1-w)/z)*(w/z) = 1 := by
  field_simp [hz]
  nlinarith [norm]

theorem bergomi_degeneracy (w r : ℝ) (hw : 0 ≤ w) (hw' : w ≤ 1) (hr : -1 ≤ r) :
    (1-2*w)^2 + 2*w*(1-w)*(1+r) = 0 ↔ w=1/2 ∧ r = -1 := by
  constructor
  · intro h
    have hp : 0 ≤ 2*w*(1-w)*(1+r) := by
      have h1 : 0 ≤ 1-w := by linarith
      have h2 : 0 ≤ 1+r := by linarith
      positivity
    have hs : (1-2*w)^2 = 0 := by nlinarith [sq_nonneg (1-2*w)]
    have he : w=1/2 := by rcases mul_self_eq_zero.mp (by nlinarith [hs] : (1-2*w)*(1-2*w)=0) with h0; linarith
    exact ⟨he, by rw [he] at h; linarith⟩
  · rintro ⟨rfl,rfl⟩; norm_num

end PricingAudit.Market
