import PricingAudit.Market
import PricingAudit.Witness

namespace PricingAudit.Discrete

def completeYears (n : ℕ) := 365*n+n/4-n/100+n/400

def leap (year : ℕ) : Prop := year%4=0 ∧ (year%100≠0 ∨ year%400=0)
instance (y : ℕ) : Decidable (leap y) := by unfold leap; infer_instance

theorem gregorian_period (y : ℕ) : completeYears (y+400) = completeYears y + 146097 := by
  unfold completeYears
  omega

private theorem leap_repeats (y : ℕ) : leap (y+400) = leap y := by
  simp [leap, Nat.add_mod]

set_option maxRecDepth 12000 in
private theorem calendar_cycle : ∀ z : Fin 400,
    completeYears (z.val+1) = completeYears z.val + (if leap (z.val+1) then 366 else 365) := by
  decide +kernel

theorem gregorian_next_year (y : ℕ) :
    completeYears (y+1) = completeYears y + (if leap (y+1) then 366 else 365) := by
  induction y using Nat.strongRecOn with
  | ind y ih =>
    by_cases small : y < 400
    · exact calendar_cycle ⟨y,small⟩
    · obtain ⟨k,rfl⟩ : ∃ k, y=k+400 := ⟨y-400, by omega⟩
      have step := ih k (by omega)
      rw [show k+400+1=k+1+400 by omega, gregorian_period (k+1), gregorian_period k, step]
      simp only [leap_repeats]
      omega

theorem weekday_period (ordinal : ℕ) (pos : 0 < ordinal) :
    (ordinal+146097-1)%7 = (ordinal-1)%7 := by omega

theorem full_calendar_size : completeYears 9999 = 3652059 := by decide +kernel

theorem actual_daycount_additive (a b c base : ℝ) :
    (b-a)/base + (c-b)/base = (c-a)/base := by ring

def nextId (limit id : ℕ) : Option ℕ := if id+1 < limit then some (id+1) else none

theorem checked_id_success (limit id : ℕ) : (nextId limit id).isSome ↔ id+1 < limit := by
  unfold nextId
  split <;> simp_all

theorem checked_id_exhaustion (limit id : ℕ) (valid : id < limit) :
    nextId limit id = none ↔ id+1 = limit := by
  unfold nextId
  split <;> simp_all <;> omega

theorem midpoint_has_exact_short_significand (word : ℕ) (h : word < 2^32) :
    (2*word+1 < 2^53) ∧ (2*word+1)%2=1 ∧
    ((word : ℚ)+1/2)/2^32 = ((2*word+1 : ℕ) : ℚ)/2^33 := by
  refine ⟨by omega, by omega, ?_⟩
  push_cast
  ring

open PricingAudit.Witness
def maxNormal : Dyadic := ⟨2^53 - 1, 971⟩

theorem max_normal_is_representable : Dyadic.roundNormal maxNormal = some maxNormal := by decide +kernel

theorem complex_numerator_exceeds_binary64 :
    Dyadic.roundNormal (maxNormal.add maxNormal) = none := by decide +kernel

end PricingAudit.Discrete
