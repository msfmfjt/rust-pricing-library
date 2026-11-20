import Lean

namespace PricingAudit.IndexGuards

abbrev Registers (width : Nat) := Nat → BitVec width

/-- Read the index after an adjacent chain of register-to-register copies. -/
def readCopies (registers : Registers width) (source : Nat) : List Nat → BitVec width
  | [] => registers source
  | target :: tail =>
    readCopies (fun r => if r = target then registers source else registers r) target tail

theorem copies_preserve_index (registers : Registers width) (source : Nat) (copies : List Nat) :
    readCopies registers source copies = registers source := by
  induction copies generalizing registers source with
  | nil => rfl
  | cons target tail ih =>
    simp [readCopies, ih]

/-- A MIR assignment, its adjacent copies, and unsigned index < length. -/
def checkIndex (registers : Registers width) (dest : Nat) (copies : List Nat)
    (index length : BitVec width) : Bool :=
  (readCopies (fun r => if r = dest then index else registers r) dest copies).ult length

theorem literal_in_bounds (i n : Nat) (hi : i < n) (hn : n < 2^width)
    (registers : Registers width) (dest : Nat) (copies : List Nat) :
    checkIndex registers dest copies (BitVec.ofNat width i) (BitVec.ofNat width n) = true := by
  simp [checkIndex, copies_preserve_index, BitVec.ult, Nat.mod_eq_of_lt (Nat.lt_trans hi hn),
    Nat.mod_eq_of_lt hn, hi]

theorem remainder_in_bounds (d n : Nat) (hd : 0 < d) (hle : d ≤ n) (hn : n < 2^width)
    (x : BitVec width) (registers : Registers width) (dest : Nat) (copies : List Nat) :
    checkIndex registers dest copies (x % BitVec.ofNat width d) (BitVec.ofNat width n) = true := by
  have h := Nat.lt_of_lt_of_le (Nat.mod_lt x.toNat hd) hle
  simp [checkIndex, copies_preserve_index, BitVec.ult, BitVec.toNat_umod,
    Nat.mod_eq_of_lt (Nat.lt_of_le_of_lt hle hn), Nat.mod_eq_of_lt hn, h]

end PricingAudit.IndexGuards
