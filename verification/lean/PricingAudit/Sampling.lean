import PricingAudit.Barrier
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.FieldSimp

namespace PricingAudit.Sampling

abbrev Word := BitVec 32
structure Counter where
  a : Word
  b : Word
  c : Word
  d : Word
  deriving Repr, DecidableEq

def m0 : Word := 0xd2511f53
def m1 : Word := 0xcd9e8d57
def upper (m x : Word) : Word := BitVec.ofNat 32 ((m.toNat*x.toNat)/2^32)

def round (k₀ k₁ : Word) (x : Counter) : Counter :=
  ⟨upper m1 x.c ^^^ x.b ^^^ k₀, m1*x.c, upper m0 x.a ^^^ x.d ^^^ k₁, m0*x.a⟩

def unround (k₀ k₁ : Word) (y : Counter) : Counter :=
  let a := (0x991a7cdb : Word)*y.d
  let c := (0x6d7cae67 : Word)*y.b
  ⟨a, upper m1 c ^^^ y.a ^^^ k₀, c, upper m0 a ^^^ y.c ^^^ k₁⟩

private theorem unm0 (x : Word) : (0x991a7cdb : Word) * (m0*x) = x := by
  rw [← BitVec.mul_assoc, show (0x991a7cdb : Word)*m0 = 1#32 by decide]
  exact BitVec.one_mul x
private theorem unm1 (x : Word) : (0x6d7cae67 : Word) * (m1*x) = x := by
  rw [← BitVec.mul_assoc, show (0x6d7cae67 : Word)*m1 = 1#32 by decide]
  exact BitVec.one_mul x
private theorem undoXor (a b c : Word) : a ^^^ (a ^^^ b ^^^ c) ^^^ c = b := by
  simp [BitVec.xor_assoc]
  rw [← BitVec.xor_assoc, BitVec.xor_self, BitVec.zero_xor]

theorem philox_round_inverse (k₀ k₁ : Word) (x : Counter) : unround k₀ k₁ (round k₀ k₁ x) = x := by
  cases x
  simp only [unround, round, unm0, unm1, undoXor]

def rounds : ℕ → Word → Word → Counter → Counter
  | 0, _, _, x => x
  | n+1, k₀, k₁, x => rounds n (k₀+0x9e3779b9) (k₁+0xbb67ae85) (round k₀ k₁ x)

theorem philox_rounds_injective (n : ℕ) (k₀ k₁ : Word) :
    Function.Injective (rounds n k₀ k₁) := by
  induction n generalizing k₀ k₁ with
  | zero => exact fun _ _ h => h
  | succ n ih =>
    intro x y eq
    have hr := ih (k₀+0x9e3779b9) (k₁+0xbb67ae85) eq
    simpa only [philox_round_inverse] using congrArg (unround k₀ k₁) hr

theorem philox_known_vector :
    rounds 10 0 0 ⟨0,0,0,0⟩ = ⟨0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8⟩ := by decide +kernel

def address (path dimension domain : ℕ) :=
  (path % (2^32), path / (2^32), dimension / 4, domain, dimension % 4)

theorem address_injective (p q d e t u : ℕ) (h : address p d t = address q e u) :
    p = q ∧ d = e ∧ t = u := by
  simp only [address, Prod.mk.injEq] at h
  omega

theorem midpoint_in_unit_interval (word : ℕ) (valid : word < 2^32) :
    0 < ((word : ℝ) + 1/2) / 2^32 ∧ ((word : ℝ)+1/2)/2^32 < 1 := by
  have h : (word : ℝ) ≤ 4294967295 := by exact_mod_cast (show word ≤ 4294967295 by omega)
  constructor
  · positivity
  · norm_num; linarith

theorem bridge_weights (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    b/(a+b) + a/(a+b) = 1 ∧ 0 < a*b/(a+b) ∧ a*b/(a+b) ≤ (a+b)/4 := by
  have h : 0 < a+b := by positivity
  refine ⟨?_, by positivity, ?_⟩
  · field_simp [ne_of_gt h]
    ring
  · apply (div_le_iff₀ h).2
    nlinarith [sq_nonneg (a-b)]

theorem bridge_reverse_duality (wl wr sd dl dr dz upstream : ℝ) :
    upstream*(wl*dl+wr*dr+sd*dz) =
      dl*(wl*upstream)+dr*(wr*upstream)+dz*(sd*upstream) := by ring

theorem antithetic_linear_cancellation (level slope z : ℝ) :
    ((level+slope*z)+(level+slope*(-z)))/2 = level := by ring

end PricingAudit.Sampling
