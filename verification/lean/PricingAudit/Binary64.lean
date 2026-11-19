import PricingAudit.Discrete

noncomputable section
namespace PricingAudit.Binary64

/-- Unsigned field split of an arbitrary 64-bit word, including every NaN payload. -/
def fields (b : BitVec 64) :=
  (b.toNat / 2^63, b.toNat / 2^52 % 2048, b.toNat % 2^52)

theorem fields_recover_bits (b : BitVec 64) :
    let (s,e,f) := fields b
    s < 2 ∧ e < 2048 ∧ f < 2^52 ∧ s*2^63 + e*2^52 + f = b.toNat := by
  have bound := b.isLt
  dsimp [fields]
  norm_num at bound ⊢
  omega

def significand (e f : ℕ) := if e = 0 then f else 2^52+f
def scale (e : ℕ) : ℝ := (2:ℝ) ^ (if e = 0 then (-1074:ℤ) else (e:ℤ)-1075)
def decodedFinite (s : Bool) (e f : ℕ) : ℝ :=
  (if s then -(significand e f : ℝ) else (significand e f : ℝ)) * scale e

theorem zero_significand (e f : ℕ) : significand e f = 0 ↔ e = 0 ∧ f = 0 := by
  by_cases he : e = 0 <;> simp [significand, he]

/-- For finite words the sign/magnitude classifier agrees with exact binary64 decoding.
Signed zero is valid in the nonnegative wrapper and excluded from the positive wrapper. -/
theorem numeric_wrapper_guards (s : Bool) (e f : ℕ) :
    ((e < 2047 ∧ s = false ∧ significand e f ≠ 0) ↔ e < 2047 ∧ 0 < decodedFinite s e f) ∧
    ((e < 2047 ∧ (s = false ∨ significand e f = 0)) ↔ e < 2047 ∧ 0 ≤ decodedFinite s e f) := by
  have hp : 0 < scale e := by unfold scale; positivity
  have hn : (0:ℝ) ≤ (significand e f : ℝ) := Nat.cast_nonneg _
  have hpos : (0:ℝ) < (significand e f : ℝ) ↔ significand e f ≠ 0 := by
    exact_mod_cast (Nat.pos_iff_ne_zero (n := significand e f))
  have hz : (significand e f : ℝ) ≤ 0 ↔ significand e f = 0 := by
    constructor
    · intro h
      exact_mod_cast (le_antisymm h hn)
    · intro h
      simp [h]
  have neg : ¬ 0 < -(significand e f : ℝ) := by linarith
  cases s <;> simp [decodedFinite, mul_pos_iff_of_pos_right hp, mul_nonneg_iff_of_pos_right hp,
    hpos, hn, hz, neg]
  intro _
  constructor
  · intro zero
    simp [zero]
  · intro nonpos
    apply hz.mp
    by_contra! positive
    exact (not_lt.mpr nonpos) (mul_pos positive hp)


end PricingAudit.Binary64
