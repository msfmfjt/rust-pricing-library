import PricingAudit.TailBounds
import PricingAudit.Discrete
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Analysis.SpecialFunctions.ImproperIntegrals
import Mathlib.Data.Complex.ExponentialBounds

noncomputable section
namespace PricingAudit.NumericalRepair
open PricingAudit.Witness

/-- Writing F = g exp(x/2) removes the linear log-moneyness term in Black vega. -/
theorem vega_factorization (g x v : ℝ) (hv : v ≠ 0) :
    g * Real.exp (x/2) * Real.exp (-(x/v+v/2)^2/2) =
      g * Real.exp (-(x/v)^2/2) * Real.exp (-(v^2/8)) := by
  rw [mul_assoc, ← Real.exp_add, mul_assoc, ← Real.exp_add]
  congr 2
  field_simp [hv]
  ring

/-- Differentiating the Black call in total standard deviation gives the positive
vega integrand. The CDF hypothesis specifies the standard Gaussian density. -/
theorem black_vega_derivative (cdf : ℝ → ℝ)
    (hcdf : ∀ z, HasDerivAt cdf (Real.exp (-z^2/2)/Real.sqrt (2*Real.pi)) z)
    (g x v : ℝ) (hv : v ≠ 0) :
    HasDerivAt (fun t : ℝ => g*Real.exp (x/2)*cdf (x/t+t/2) -
        g*Real.exp (-x/2)*cdf (x/t-t/2))
      (g*Real.exp (-(x/v)^2/2)*Real.exp (-(v^2/8))/Real.sqrt (2*Real.pi)) v := by
  let a := Real.sqrt (2*Real.pi)
  let m := g*Real.exp (-(x/v)^2/2)*Real.exp (-(v^2/8))/a
  have plus : g*Real.exp (x/2)*(Real.exp (-(x/v+v/2)^2/2)/a) = m := by
    rw [← mul_div_assoc, vega_factorization g x v hv]
  have minus : g*Real.exp (-x/2)*(Real.exp (-(x/v-v/2)^2/2)/a) = m := by
    have h := vega_factorization g (-x) v hv
    rw [show ((-x)/v+v/2)^2 = (x/v-v/2)^2 by ring,
        show ((-x)/v)^2 = (x/v)^2 by ring] at h
    rw [← mul_div_assoc, h]
  have center := (hasDerivAt_const v x).div (hasDerivAt_id v) hv
  have half := (hasDerivAt_id v).div_const 2
  have call := (((hcdf _).comp v (center.add half)).const_mul (g*Real.exp (x/2))).sub
    (((hcdf _).comp v (center.sub half)).const_mul (g*Real.exp (-x/2)))
  convert call using 1
  change m = g*Real.exp (x/2)*((Real.exp (-(x/v+v/2)^2/2)/a)*_) -
      g*Real.exp (-x/2)*((Real.exp (-(x/v-v/2)^2/2)/a)*_)
  rw [← mul_assoc, plus, ← mul_assoc, minus]
  ring

/-- If the lower tail argument is at least 56, even the leading normal CDF term
of the OTM Black value is below half the smallest binary64 subnormal. Subtracting
its nonnegative second term preserves the bound for every total standard deviation. -/
theorem tiny_tail_below_half_subnormal (a low sub : ℝ) (ha : 56 ≤ a) (hlo : 0 ≤ low)
    (hmax : low ≤ (2:ℝ)^1024) (hsub : 0 ≤ sub) :
    low * (∫ x : ℝ in Set.Ioi a, Real.exp (-x^2/2) / Real.sqrt (2*Real.pi)) - sub <
      (2:ℝ)^(-1075:ℤ) := by
  have hr : 1 ≤ Real.sqrt (2*Real.pi) := by
    apply (Real.le_sqrt (by norm_num) (by positivity)).mpr
    nlinarith [Real.pi_gt_three]
  let k := Real.exp (a - a^2/2)
  have hg : MeasureTheory.Integrable (fun x : ℝ => Real.exp (-x^2/2) / Real.sqrt (2*Real.pi)) := by
    have h := (integrable_exp_neg_mul_sq (b := (1:ℝ)/2) (by norm_num)).div_const (Real.sqrt (2*Real.pi))
    convert h using 1
    ext x
    congr 2
    ring
  -- Integrate the exponential envelope from a to infinity.
  have hle := MeasureTheory.setIntegral_mono_on hg.integrableOn
    ((integrableOn_exp_neg_Ioi a).const_mul k) measurableSet_Ioi (fun x (hx : a < x) => by
      calc
        Real.exp (-x^2/2) / Real.sqrt (2*Real.pi) ≤ Real.exp (-x^2/2) :=
          div_le_self (Real.exp_pos _).le hr
        _ ≤ k * Real.exp (-x) := by
          dsimp [k]
          rw [← Real.exp_add]
          exact Real.exp_le_exp.mpr (by
            nlinarith [mul_nonneg (sub_nonneg.mpr hx.le) (show 0 ≤ x+a-2 by linarith)]))
  have hk : k * Real.exp (-a) = Real.exp (-a^2/2) := by
    dsimp [k]
    rw [← Real.exp_add]
    congr 1
    ring
  rw [MeasureTheory.integral_const_mul, integral_exp_neg_Ioi, hk] at hle
  have harg : -a^2/2 < -((2099:ℝ) * Real.log 2) := by
    nlinarith [mul_nonneg (sub_nonneg.mpr ha) (show 0 ≤ a+56 by linarith), Real.log_two_lt_d9]
  have e := Real.exp_lt_exp.mpr harg
  rw [Real.exp_neg, show (2099:ℝ) = (2099:ℕ) by norm_num,
      Real.exp_nat_mul, Real.exp_log (by norm_num : (0:ℝ) < 2)] at e
  calc
    _ ≤ low * Real.exp (-a^2/2) :=
      le_trans (sub_le_self _ hsub) (mul_le_mul_of_nonneg_left hle hlo)
    _ ≤ (2:ℝ)^1024 * _ := mul_le_mul_of_nonneg_right hmax (Real.exp_pos _).le
    _ < (2:ℝ)^1024 * ((2:ℝ)^2099)⁻¹ := mul_lt_mul_of_pos_left e (by positivity)
    _ = _ := by norm_num

/-- Each continued-fraction step carries the divided difference without subtracting
two near-equal ratios. Only the two positive tail denominators must be nonzero. -/
theorem mills_divided_step (n a b l r d : ℝ) (hd : l-r = (b-a)*d)
    (ha : a+l ≠ 0) (hb : b+r ≠ 0) :
    n/(a+l)-n/(b+r) = (b-a)*(n*(1-d)/((a+l)*(b+r))) := by
  field_simp [ha, hb]
  linear_combination -n * hd

private theorem from_zero {f d : ℝ → ℝ} (hf : ∀ y, HasDerivAt f (d y) y)
    (hd : ∀ y, 0 ≤ y → 0 ≤ d y) {x : ℝ} (hx : 0 ≤ x) : f 0 ≤ f x := by
  have hm : MonotoneOn f (Set.Ici 0) :=
    monotoneOn_of_hasDerivWithinAt_nonneg (convex_Ici _)
      (fun y _ => (hf y).continuousAt.continuousWithinAt)
      (fun y _ => (hf y).hasDerivWithinAt)
      (fun y hy => hd y (interior_subset hy))
  exact hm (Set.mem_Ici.mpr le_rfl) hx hx

/-- The quadratic Taylor polynomial used under the vega integral overestimates
exp(-x). Its absolute remainder is at most x^3/6 everywhere on the positive ray. -/
theorem exp_quadratic_remainder (x : ℝ) (hx : 0 ≤ x) :
    0 ≤ (1-x+x^2/2)-Real.exp (-x) ∧ (1-x+x^2/2)-Real.exp (-x) ≤ x^3/6 := by
  have de (y : ℝ) : HasDerivAt (fun t : ℝ => Real.exp (-t)) (-Real.exp (-y)) y := by
    simpa using (((hasDerivAt_id y).neg).exp)
  have du (y : ℝ) : HasDerivAt (fun t : ℝ => 1-t+t^2/2-Real.exp (-t))
      (-1+y+Real.exp (-y)) y := by
    convert ((((hasDerivAt_const y (1:ℝ)).sub (hasDerivAt_id y)).add
      ((hasDerivAt_pow 2 y).div_const 2)).sub (de y)) using 1
    ring
  have lower (y : ℝ) (hy : 0 ≤ y) : 0 ≤ 1-y+y^2/2-Real.exp (-y) := by
    have h := from_zero du (fun z _ => by linarith [Real.add_one_le_exp (-z)]) hy
    simpa using h
  refine ⟨lower x hx, ?_⟩
  have dl (y : ℝ) : HasDerivAt (fun t : ℝ => t^3/6-(1-t+t^2/2-Real.exp (-t)))
      (1-y+y^2/2-Real.exp (-y)) y := by
    convert (((hasDerivAt_pow 3 y).div_const 6).sub (du y)) using 1
    ring
  have h := from_zero dl lower hx
  simpa [sub_nonneg] using h

/-- Uniform relative bound on each positive integrand at the Rust branch cutoff. -/
theorem black_integrand_relative_error (sd u : ℝ) (hs : 0 ≤ sd) (hmax : sd ≤ 1/256)
    (hu : 0 ≤ u) (hu1 : u ≤ 1) :
    let x := sd^2*u^2/8
    0 ≤ (1-x+x^2/2)/Real.exp (-x)-1 ∧ (1-x+x^2/2)/Real.exp (-x)-1 < 2^(-59:ℤ) := by
  let x := sd^2*u^2/8
  have hx : 0 ≤ x := by dsimp [x]; positivity
  have sd2 : sd^2 ≤ (1/256:ℝ)^2 := pow_le_pow_left₀ hs hmax 2
  have u2 : u^2 ≤ (1:ℝ)^2 := pow_le_pow_left₀ hu hu1 2
  have bound : x ≤ 1/524288 := by
    have p := mul_le_mul sd2 u2 (sq_nonneg _) (sq_nonneg _)
    dsimp [x]
    nlinarith
  have h3 := pow_le_pow_left₀ hx bound 3
  have e : 0 < Real.exp (-x) := Real.exp_pos _
  obtain ⟨lo, hi⟩ := exp_quadratic_remainder x hx
  change 0 ≤ (1-x+x^2/2) / Real.exp (-x) - 1 ∧ (1-x+x^2/2) / Real.exp (-x) - 1 < _
  constructor
  · apply sub_nonneg.mpr
    apply (one_le_div e).mpr
    linarith
  · apply sub_lt_iff_lt_add.mpr
    apply (div_lt_iff₀ e).mpr
    norm_num at h3 ⊢
    linarith [Real.add_one_le_exp (-x)]

/-- Density bounds on the symmetric ATM interval, including its leading small-vol term. -/
theorem atm_density_bounds (s : ℝ) (hs : 0 ≤ s) :
    s * ((1-s^2/8)/Real.sqrt (2*Real.pi)) ≤ Market.atmBlack 1 1 1 s ∧
    Market.atmBlack 1 1 1 s ≤ s * (1/Real.sqrt (2*Real.pi)) := by
  let density := fun x : ℝ => Real.exp (-x^2/2) / Real.sqrt (2*Real.pi)
  have hc : Continuous density := by dsimp [density]; fun_prop
  have hr : 0 < Real.sqrt (2*Real.pi) := by positivity
  have le : -(s/2) ≤ s/2 := by linarith
  have lo := intervalIntegral.integral_mono_on (μ := MeasureTheory.volume) le
    (intervalIntegrable_const (c := (1-s^2/8)/Real.sqrt (2*Real.pi)))
    (hc.intervalIntegrable _ _) (fun x hx => by
      dsimp [density]
      apply div_le_div_of_nonneg_right _ hr.le
      nlinarith [Real.add_one_le_exp (-x^2/2), mul_nonneg (sub_nonneg.mpr hx.1) (sub_nonneg.mpr hx.2)])
  have hi := intervalIntegral.integral_mono_on (μ := MeasureTheory.volume) le
    (hc.intervalIntegrable _ _) (intervalIntegrable_const (c := 1/Real.sqrt (2*Real.pi)))
    (fun x _ => by
      dsimp [density]
      apply div_le_div_of_nonneg_right _ hr.le
      exact Real.exp_le_one_iff.mpr (by nlinarith [sq_nonneg x]))
  rw [intervalIntegral.integral_const, smul_eq_mul] at lo hi
  rw [show s/2 - -(s/2) = s by ring] at lo hi
  simpa only [Market.atmBlack, one_mul] using And.intro lo hi

/-- The exact unit ATM price at sd = 2^-60 lies strictly between the binary64
midpoints around 0x3c19884533d43651 (significand 7186705221432913, exponent -114). -/
theorem atm_witness_rounding_cell :
    (2*7186705221432913-1:ℝ)/2^115 < Market.atmBlack 1 1 1 ((2:ℝ)^(-60:ℤ)) ∧
    Market.atmBlack 1 1 1 ((2:ℝ)^(-60:ℤ)) < (2*7186705221432913+1:ℝ)/2^115 := by
  have hr : 0 < Real.sqrt (2*Real.pi) := by positivity
  have lower : (2.50662827463100050241:ℝ) < Real.sqrt (2*Real.pi) := by
    apply Real.lt_sqrt_of_sq_lt
    linarith [Real.pi_gt_d20]
  have upper : Real.sqrt (2*Real.pi) < (2.50662827463100050242:ℝ) := by
    apply (Real.sqrt_lt' (by norm_num)).mpr
    linarith [Real.pi_lt_d20]
  obtain ⟨lo, hi⟩ := atm_density_bounds ((2:ℝ)^(-60:ℤ)) (by positivity)
  constructor
  · apply lt_of_lt_of_le _ lo
    rw [mul_div_assoc', lt_div_iff₀ hr]
    norm_num at lower upper ⊢
    linarith
  · apply lt_of_le_of_lt hi
    rw [mul_div_assoc', div_lt_iff₀ hr]
    norm_num at lower upper ⊢
    linarith

/-- Product alignment and quotient rescaling over the exact reals for any exponents. -/
theorem align_dot (p q : ℝ) (i j k : ℤ) :
    (p*2^(i-k) + q*2^(j-k)) * 2^k = p*2^i + q*2^j := by
  rw [zpow_sub₀ (by norm_num : (2:ℝ) ≠ 0), zpow_sub₀ (by norm_num : (2:ℝ) ≠ 0)]
  field_simp

theorem restore_quotient (p d : ℝ) (i j : ℤ) :
    (p/d) * 2^(i-j) = (p*2^i)/(d*2^j) := by
  rw [zpow_sub₀ (by norm_num : (2:ℝ) ≠ 0)]
  ring

/-- Including the low parts gives the full residual before rounding the quotient. -/
theorem quotient_residual (nh nl dh dl q : ℝ) :
    (nh-q*dh)+(nl-q*dl) = (nh+nl)-q*(dh+dl) := by ring

/-- Round the aligned mantissa sum, divide by two, restore its exponent.
Both quotient components of the largest-finite witness survive every rounding. -/
theorem repaired_max_division :
    let m : Dyadic := ⟨2^53-1, -52⟩
    (do let numerator ← Dyadic.roundNormal (m.add m)
        let q ← Dyadic.roundNormal (numerator.divPow2 1)
        Dyadic.roundNormal (q.mul ⟨1,1023⟩)) = some Discrete.maxNormal ∧
    Dyadic.roundNormal (m.add m.neg) = some (Dyadic.ofInt 0) := by decide +kernel

end PricingAudit.NumericalRepair
