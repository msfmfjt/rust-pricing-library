import PricingAudit.Dynamics
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

noncomputable section
namespace PricingAudit.Graph

/-- An unfolded arithmetic payoff graph. Multiple occurrences of the same spot share its seed. -/
inductive Expr where
  | literal (value : ℝ) | spot
  | plus (a b : Expr) | minus (a b : Expr) | times (a b : Expr) | quotient (a b : Expr)
  | neg (a : Expr) | exp (a : Expr) | log (a : Expr)

def eval (e : Expr) (x : ℝ) : ℝ :=
  match e with
  | .literal c => c | .spot => x
  | .plus a b => eval a x + eval b x | .minus a b => eval a x - eval b x
  | .times a b => eval a x * eval b x | .quotient a b => eval a x / eval b x
  | .neg a => -eval a x | .exp a => Real.exp (eval a x) | .log a => Real.log (eval a x)

def regular (e : Expr) (x : ℝ) : Prop :=
  match e with
  | .literal _ | .spot => True
  | .plus a b | .minus a b | .times a b => regular a x ∧ regular b x
  | .quotient a b => regular a x ∧ regular b x ∧ eval b x ≠ 0
  | .neg a | .exp a => regular a x
  | .log a => regular a x ∧ 0 < eval a x

def tangent (e : Expr) (x : ℝ) : ℝ :=
  match e with
  | .literal _ => 0 | .spot => 1
  | .plus a b => tangent a x + tangent b x | .minus a b => tangent a x - tangent b x
  | .times a b => tangent a x * eval b x + eval a x * tangent b x
  | .quotient a b => (tangent a x * eval b x - eval a x * tangent b x) / eval b x ^ 2
  | .neg a => -tangent a x | .exp a => Real.exp (eval a x) * tangent a x
  | .log a => tangent a x / eval a x

/-- Accumulated sensitivity at all occurrences of the same input, propagating the output seed. -/
def pullback (e : Expr) (x seed : ℝ) : ℝ :=
  match e with
  | .literal _ => 0 | .spot => seed
  | .plus a b => pullback a x seed + pullback b x seed
  | .minus a b => pullback a x seed + pullback b x (-seed)
  | .times a b => pullback a x (seed * eval b x) + pullback b x (seed * eval a x)
  | .quotient a b => pullback a x (seed / eval b x) +
      pullback b x (-seed * eval a x / eval b x ^ 2)
  | .neg a => pullback a x (-seed) | .exp a => pullback a x (seed * Real.exp (eval a x))
  | .log a => pullback a x (seed / eval a x)

theorem tangent_hasDerivAt (e : Expr) (x : ℝ) (h : regular e x) :
    HasDerivAt (eval e) (tangent e x) x := by
  induction e with
  | literal c => exact hasDerivAt_const x c
  | spot => exact hasDerivAt_id' x
  | plus a b ia ib => exact (ia h.1).add (ib h.2)
  | minus a b ia ib => exact (ia h.1).sub (ib h.2)
  | times a b ia ib => exact (ia h.1).mul (ib h.2)
  | quotient a b ia ib => exact (ia h.1).div (ib h.2.1) h.2.2
  | neg a ia => exact (ia h).neg
  | exp a ia => exact (ia h).exp
  | log a ia => exact (ia h.1).log (ne_of_gt h.2)

theorem pullback_tangent (e : Expr) (x seed : ℝ) (h : regular e x) :
    pullback e x seed = seed * tangent e x := by
  induction e generalizing seed with
  | literal c => simp [pullback, tangent]
  | spot => simp [pullback, tangent]
  | plus a b ia ib | minus a b ia ib | times a b ia ib =>
    simp only [pullback, tangent, ia _ h.1, ib _ h.2]
    ring
  | quotient a b ia ib =>
    simp only [pullback, tangent, ia _ h.1, ib _ h.2.1]
    field_simp [h.2.2]
    ring
  | neg a ia | exp a ia =>
    simp only [pullback, tangent, ia _ h]
    ring
  | log a ia =>
    simp only [pullback, tangent, ia _ h.1]
    ring

theorem reverse_is_derivative (e : Expr) (x seed : ℝ) (h : regular e x) :
    pullback e x seed = seed * deriv (eval e) x := by
  rw [pullback_tangent e x seed h, (tangent_hasDerivAt e x h).deriv]

end PricingAudit.Graph
