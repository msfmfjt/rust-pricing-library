namespace PricingAudit.Projection

/-- The single-field read and return fragment of Rust MIR for a primitive scalar. -/
inductive Instruction (n : Nat) where
  | readField (index : Fin n)
  | ret

def execute {n : Nat} {α : Type} (fields : Fin n → α) :
    List (Instruction n) → Option α → Option α
  | [], _ => none
  | .readField i :: rest, _ => execute fields rest (some (fields i))
  | .ret :: _, value => value

theorem returns_field {n : Nat} {α : Type} (fields : Fin n → α) (i : Fin n) :
    execute fields [.readField i, .ret] none = some (fields i) := rfl

end PricingAudit.Projection
