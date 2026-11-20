import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-! Escrowed affine cash and proportional dividends with deterministic carry.
The discount ratio is the market's discount factor divided by its dividend
discount factor. Lists are supplied in the engine's ex-date order. -/
namespace PricingAudit.Dividends
noncomputable section

structure Event where
  cash : ℝ
  beta : ℝ
  discount : ℝ
  cash_nonneg : 0 ≤ cash
  beta_nonneg : 0 ≤ beta
  beta_lt_one : beta < 1
  discount_pos : 0 < discount

def keep (e : Event) : ℝ := 1 - e.beta

private theorem keep_pos (e : Event) : 0 < keep e := sub_pos.mpr e.beta_lt_one

structure Coordinate where
  a : ℝ
  b : ℝ

def initial (spot reserve : ℝ) : Coordinate :=
  ⟨reserve / spot, (spot - reserve) / spot⟩

def reconstruct (c : Coordinate) (spot f : ℝ) : ℝ := c.a * spot + c.b * f

def after (c : Coordinate) (e : Event) (spot : ℝ) : Coordinate :=
  ⟨keep e * c.a - e.cash / spot, keep e * c.b⟩

theorem initial_scale_positive {spot reserve : ℝ} (hs : 0 < spot) (hr : reserve < spot) :
    0 < (initial spot reserve).b := div_pos (sub_pos.mpr hr) hs

theorem scale_stays_positive (c : Coordinate) (e : Event) (spot : ℝ) (hb : 0 < c.b) :
    0 < (after c e spot).b := mul_pos (keep_pos e) hb

theorem jump_commutes (c : Coordinate) (e : Event) {spot : ℝ} (hs : 0 < spot) (f : ℝ) :
    reconstruct (after c e spot) spot f = keep e * reconstruct c spot f - e.cash := by
  unfold reconstruct after
  dsimp only
  field_simp [ne_of_gt hs]
  ring

/-- The cash term added in compile_reserve, where g is the cumulative retention
factor before this event. -/
def pvTerm (e : Event) (g : ℝ) : ℝ := e.cash / (g * keep e) * e.discount

def accumulate : List Event → ℝ → ℝ → ℝ
  | [], _, acc => acc
  | e :: rest, g, acc => accumulate rest (g * keep e) (acc + pvTerm e g)

def remainingPV : List Event → ℝ → ℝ
  | [], _ => 0
  | e :: rest, g => pvTerm e g + remainingPV rest (g * keep e)

theorem accumulate_spec (events : List Event) (g acc : ℝ) :
    accumulate events g acc = acc + remainingPV events g := by
  induction events generalizing g acc with
  | nil => simp [accumulate, remainingPV]
  | cons e rest ih => simp [accumulate, remainingPV, ih, add_assoc]

theorem remaining_nonnegative (events : List Event) {g : ℝ} (hg : 0 < g) :
    0 ≤ remainingPV events g := by
  induction events generalizing g with
  | nil => exact le_rfl
  | cons e rest ih =>
    have hk := mul_pos hg (keep_pos e)
    have hterm := mul_nonneg (div_nonneg e.cash_nonneg hk.le) e.discount_pos.le
    exact add_nonneg hterm (ih hk)

theorem zero_cash_suffix (events : List Event) (g : ℝ) (hc : ∀ e ∈ events, e.cash = 0) :
    remainingPV events g = 0 := by
  induction events generalizing g with
  | nil => rfl
  | cons e rest ih =>
    have he := hc e (by simp)
    have hs := ih (g * keep e) (fun t ht => hc t (by simp [ht]))
    simp [remainingPV, pvTerm, he, hs]

/-- Balance in currency units: carry it to the next date and apply the jump. -/
def carryAndPay (e : Event) (priorDiscount balance : ℝ) : ℝ :=
  keep e * (priorDiscount / e.discount * balance) - e.cash

def settle : List Event → ℝ → ℝ → ℝ
  | [], _, balance => balance
  | e :: rest, d, balance => settle rest e.discount (carryAndPay e d balance)

theorem funded_step (e : Event) {g d : ℝ} (hg : 0 < g) (hd : 0 < d) (later : ℝ) :
    carryAndPay e d (g / d * (pvTerm e g + later)) =
      (g * keep e) / e.discount * later := by
  unfold carryAndPay pvTerm
  field_simp [ne_of_gt hg, ne_of_gt hd, ne_of_gt (keep_pos e), ne_of_gt e.discount_pos]
  ring

/-- After paying the current dividend, the funded cash balance is nonnegative
and the positive residual-equity position makes the physical post-spot positive. -/
theorem funded_post_positive (e : Event) (rest : List Event) {g d b f : ℝ}
    (hg : 0 < g) (hd : 0 < d) (hb : 0 < b) (hf : 0 < f) :
    0 < carryAndPay e d (g / d * (pvTerm e g + remainingPV rest (g * keep e))) +
      (keep e * b) * f := by
  rw [funded_step e hg hd]
  have hk := mul_pos hg (keep_pos e)
  exact add_pos_of_nonneg_of_pos
    (mul_nonneg (div_nonneg hk.le e.discount_pos.le) (remaining_nonnegative rest hk))
    (mul_pos (mul_pos (keep_pos e) hb) hf)

theorem funded_sweep_zero (events : List Event) (g d : ℝ) (hg : 0 < g) (hd : 0 < d) :
    settle events d (g / d * remainingPV events g) = 0 := by
  induction events generalizing g d with
  | nil => simp [settle, remainingPV]
  | cons e rest ih =>
    simp only [settle, remainingPV]
    rw [funded_step e hg hd]
    exact ih _ _ (mul_pos hg (keep_pos e)) e.discount_pos

theorem initial_reserve_settles (events : List Event) :
    settle events 1 (accumulate events 1 0) = 0 := by
  rw [accumulate_spec]
  simpa using funded_sweep_zero events 1 1 zero_lt_one zero_lt_one

end
end PricingAudit.Dividends
