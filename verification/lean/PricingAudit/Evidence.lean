import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.Order.Field.Rat

namespace PricingAudit.Evidence

/-- Absolute plus relative tolerance on already-represented input values. -/
def tolerance (x y a r : ℚ) := a + r * max |x| |y|

def riskGate (aad lo hi absolute relative : ℚ) : Bool := decide
  (|aad-lo| ≤ tolerance aad lo absolute relative ∧
   |aad-hi| ≤ tolerance aad hi absolute relative ∧
   |lo-hi| ≤ 2 * tolerance lo hi absolute relative)

theorem gate_requires_both_bumps (aad lo hi a r : ℚ) (ok : riskGate aad lo hi a r = true) :
    |aad-lo| ≤ tolerance aad lo a r ∧ |aad-hi| ≤ tolerance aad hi a r ∧
      |lo-hi| ≤ 2*tolerance lo hi a r := of_decide_eq_true ok

end PricingAudit.Evidence
