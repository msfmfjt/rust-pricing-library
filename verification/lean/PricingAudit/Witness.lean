import PricingAudit.Barrier

/-! Each input and intermediate in these traces is a dyadic rational. The
integer implementation rounds each source operation on the normal binary64 grid
with ties to even. No machine Float primitive is used by the proofs. -/
namespace PricingAudit.Witness

/-- Represents m times two to the power e. Operations may use unnormalised m. -/
structure Dyadic where
  m : ℤ
  e : ℤ
  deriving DecidableEq, Repr

namespace Dyadic

def ofInt (n : ℤ) : Dyadic := ⟨n, 0⟩
def value (a : Dyadic) : ℚ := (a.m : ℚ) * (2 : ℚ) ^ a.e
def neg (a : Dyadic) : Dyadic := ⟨-a.m, a.e⟩
def mul (a b : Dyadic) : Dyadic := ⟨a.m * b.m, a.e + b.e⟩

/-- Align exponents before adding the signed integer coefficients. -/
def add (a b : Dyadic) : Dyadic :=
  let e := min a.e b.e
  ⟨a.m * 2 ^ (a.e - e).toNat + b.m * 2 ^ (b.e - e).toNat, e⟩

def divPow2 (a : Dyadic) (n : ℕ) : Dyadic := ⟨a.m, a.e - n⟩

private def nearestEven (n denom : ℕ) : ℕ :=
  let low := n / denom
  let twiceRem := 2 * (n % denom)
  if twiceRem > denom ∨ (twiceRem = denom ∧ low % 2 = 1) then low + 1 else low

/-- Find the leading bit of the unnormalised integer and check the bracket
explicitly. Every intermediate coefficient in these traces fits in 256 bits. -/
private def leadingBit (n : ℕ) : Option ℕ :=
  let strides : List ℕ := [128, 64, 32, 16, 8, 4, 2, 1]
  let e := strides.foldl (fun base stride =>
    if 2 ^ (base + stride) ≤ n then base + stride else base) 0
  if 2 ^ e ≤ n ∧ n < 2 ^ (e + 1) then some e else none

/-- A nonzero result has 53 significant binary digits, including its leading 1.
This partial replay handles zero and the normal input exponent range. Subnormal
inputs and overflow are outside these witnesses and return none. -/
def roundNormal (a : Dyadic) : Option Dyadic := do
  if a.m = 0 then return ofInt 0
  let magnitude := a.m.natAbs
  let top ← leadingBit magnitude
  let binade := a.e + (top : ℤ)
  if binade < -1022 ∨ 1023 < binade then none else
    let rounded := if top ≤ 52 then magnitude * 2 ^ (52 - top)
                   else nearestEven magnitude (2 ^ (top - 52))
    let carry := rounded = 2 ^ 53
    let newBinade := if carry then binade + 1 else binade
    if newBinade > 1023 then none else
      let significand : ℤ := if carry then (2 ^ 52 : ℕ) else rounded
      return ⟨if a.m < 0 then -significand else significand, newBinade - 52⟩

end Dyadic
open Dyadic

/-- With h=1 the Rust normalization divides by the exactly representable 2. -/
def roundedInterior (x : Dyadic) : Option Dyadic := do
  let numerator ← roundNormal (x.add (ofInt 1))
  let u ← roundNormal (numerator.divPow2 1)
  let u2 ← roundNormal (u.mul u)
  let u3 ← roundNormal (u2.mul u)
  let sixU ← roundNormal ((ofInt 6).mul u)
  let inner ← roundNormal ((ofInt (-15)).add sixU)
  let middle ← roundNormal (u.mul inner)
  let outer ← roundNormal ((ofInt 10).add middle)
  roundNormal (u3.mul outer)

/-- x = 1 - 2⁻²⁵. -/
def distance : Dyadic := ⟨2 ^ 25 - 1, -25⟩

theorem exact_safety :
    1 - quintic ((distance.value + 1) / 2) =
      (22517997633536003 : ℚ) / 2 ^ 129 := by norm_num [distance, value, quintic]

/-- 1 + 2⁻⁵¹, or two binary64 ulps above one. -/
theorem rounded_hit : roundedInterior distance = some ⟨2 ^ 52 + 2, -52⟩ := by decide +kernel

/-- The computed complement is exactly −2⁻⁵¹. -/
theorem rounded_safety :
    (roundedInterior distance).bind (fun hit => roundNormal ((ofInt 1).add hit.neg)) =
      some ⟨-(2 ^ 52), -103⟩ := by decide +kernel

theorem complementary_evaluation :
    roundedInterior distance.neg = some ⟨5629499408384001, -127⟩ := by decide +kernel

/-- For h=1 the repaired interior branch reflects if fl(1 − H) ≤ 32 eps.
32 eps is exactly 2⁻⁴⁷ for binary64. Compare dyadics with an exact sign test. -/
def repairedWeights (x : Dyadic) : Option (Dyadic × Dyadic) := do
  let hit ← roundedInterior x
  let safety ← roundNormal ((ofInt 1).add hit.neg)
  let cutoff : Dyadic := ⟨1, -47⟩
  if (safety.add cutoff.neg).m ≤ 0 then
    let tail ← roundedInterior x.neg
    let high ← roundNormal ((ofInt 1).add tail.neg)
    return (high, tail)
  return (hit, safety)

theorem repaired_weight_pair :
    repairedWeights distance = some (⟨2 ^ 52, -52⟩, ⟨5629499408384001, -127⟩) := by decide +kernel

/-- The computed survival differs from the exact result by one quarter of its ulp. -/
theorem repaired_safety_error :
    (⟨5629499408384001, -127⟩ : Dyadic).value - (1 - quintic ((distance.value + 1) / 2)) =
      (1 : ℚ) / 2 ^ 129 := by norm_num [distance, value, quintic]

/-- One cash dividend, zero carry, initial spot 128, cash 64 and F=2⁻⁴⁷. -/
def dividendRoutes : Option (Dyadic × Dyadic) := do
  let spot := ofInt 128
  let cash := ofInt 64
  let f : Dyadic := ⟨1, -47⟩
  let a ← roundNormal (cash.divPow2 7)
  let residual ← roundNormal (spot.add cash.neg)
  let b ← roundNormal (residual.divPow2 7)
  let reserve ← roundNormal (a.mul spot)
  let equity ← roundNormal (b.mul f)
  let before ← roundNormal (reserve.add equity)
  let fromPre ← roundNormal (before.add reserve.neg)
  return (fromPre, equity)

/-- Reconstruct-then-subtract gives zero. Post-event reconstruction gives 2⁻⁴⁸. -/
theorem dividend_routes_disagree :
    dividendRoutes = some (ofInt 0, ⟨2 ^ 52, -100⟩) := by decide +kernel

end PricingAudit.Witness
