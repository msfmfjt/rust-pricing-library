import Mathlib.Data.List.Basic

namespace PricingAudit.Schema

/-- If every tag occurs once, a tag accepted by two branches identifies the same branch. -/
theorem tag_determines_branch (entries : List (String × Nat)) (tag : String) (i j : Nat)
    (distinct : (entries.map Prod.fst).Nodup) (hi : (tag,i) ∈ entries) (hj : (tag,j) ∈ entries) :
    i = j := by
  induction entries with
  | nil => simp at hi
  | cons a rest ih =>
    obtain ⟨fresh, tail⟩ := List.nodup_cons.mp distinct
    rcases List.mem_cons.mp hi with ha | htailI <;> rcases List.mem_cons.mp hj with hb | htailJ
    · exact congrArg Prod.snd (ha.trans hb.symm)
    · rw [← ha] at fresh
      exact (fresh (List.mem_map.mpr ⟨(tag,j), htailJ, rfl⟩)).elim
    · rw [← hb] at fresh
      exact (fresh (List.mem_map.mpr ⟨(tag,i), htailI, rfl⟩)).elim
    · exact ih tail htailI htailJ

def contained (keys requiredNames : List String) : Bool := requiredNames.all (keys.contains ·)

theorem lookup_exists_of_check (keys requiredNames : List String)
    (ok : contained keys requiredNames = true) {name : String} (h : name ∈ requiredNames) :
    name ∈ keys := by
  simpa using List.all_eq_true.mp ok name h

end PricingAudit.Schema
