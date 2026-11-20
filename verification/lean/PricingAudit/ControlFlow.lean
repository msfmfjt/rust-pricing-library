import Lean

open Lean

namespace PricingAudit.ControlFlow

/-- A block in a single Rust MIR body. Successors include explicit unwind edges.
Each category has a rank; zero means that the event is unreachable from this block. -/
structure Block where
  edges : List Nat
  mask : Nat
  ranks : Array Nat
  deriving Inhabited, FromJson

abbrev Graph := Array Block

def block (g : Graph) (i : Nat) : Block := g[i]?.getD default
def rank (g : Graph) (k i : Nat) : Nat := ((block g i).ranks[k]?).getD 0
def event (g : Graph) (k i : Nat) : Prop := (block g i).mask.testBit k = true

instance (g : Graph) (k i : Nat) : Decidable (event g k i) := by unfold event; infer_instance

/-- Path in the intraprocedural graph. No data feasibility assumptions are added. -/
inductive Reaches (g : Graph) (k : Nat) : Nat → Prop
  | here {i} : i < g.size → event g k i → Reaches g k i
  | step {i j} : i < g.size → j ∈ (block g i).edges → Reaches g k j → Reaches g k i

def RowOK (g : Graph) (k i : Nat) : Prop :=
  (∀ j ∈ (block g i).edges, j < g.size) ∧
  (event g k i → 0 < rank g k i) ∧
  (rank g k i = 0 → ∀ j ∈ (block g i).edges, rank g k j = 0) ∧
  (0 < rank g k i → event g k i ∨
    ∃ j ∈ (block g i).edges, 0 < rank g k j ∧ rank g k j < rank g k i)

instance (g : Graph) (k i : Nat) : Decidable (RowOK g k i) := by
  unfold RowOK
  infer_instance

/-- Independent, closed obligations suitable for small kernel reductions. -/
def boundsOK (size : Nat) (edges : List Nat) : Bool := decide (∀ j ∈ edges, j < size)

def localOK (hit : Bool) (r : Nat) (next : List Nat) : Bool := decide (
  (hit = true → 0 < r) ∧ (r = 0 → ∀ v ∈ next, v = 0) ∧
  (0 < r → hit = true ∨ ∃ v ∈ next, 0 < v ∧ v < r))

theorem row_from_local (g : Graph) (k i : Nat)
    (hb : boundsOK g.size (block g i).edges = true)
    (hc : localOK ((block g i).mask.testBit k) (rank g k i)
      ((block g i).edges.map (rank g k)) = true) : RowOK g k i := by
  have bounds : ∀ j ∈ (block g i).edges, j < g.size := of_decide_eq_true hb
  simp only [localOK, decide_eq_true_eq] at hc
  refine ⟨bounds, hc.1, ?_, ?_⟩
  · intro zero j edge
    exact hc.2.1 zero _ (List.mem_map.mpr ⟨j, edge, rfl⟩)
  · intro pos
    rcases hc.2.2 pos with hit | ⟨v, mem, pos', less⟩
    · exact Or.inl hit
    · obtain ⟨j, edge, rfl⟩ := List.mem_map.mp mem
      exact Or.inr ⟨j, edge, pos', less⟩

def Certified (g : Graph) (k : Nat) : Prop := ∀ i ∈ List.range g.size, RowOK g k i

instance (g : Graph) (k : Nat) : Decidable (Certified g k) := by unfold Certified; infer_instance

def check (g : Graph) (k : Nat) : Bool := decide (Certified g k)

theorem reachability_iff (g : Graph) (k : Nat) (ok : check g k = true)
    {i : Nat} (hi : i < g.size) : 0 < rank g k i ↔ Reaches g k i := by
  have h : Certified g k := of_decide_eq_true ok
  have valid {v : Nat} (hv : v < g.size) : RowOK g k v := h v (List.mem_range.mpr hv)
  constructor
  · intro positive
    suffices go : ∀ n v, rank g k v = n → v < g.size → 0 < n → Reaches g k v from
      go _ i rfl hi positive
    intro n
    induction n using Nat.strongRecOn with
    | ind n prev =>
      intro v eq hv hn
      rcases (valid hv).2.2.2 (eq ▸ hn) with hit | ⟨w, edge, pos, smaller⟩
      · exact .here hv hit
      · exact .step hv edge (prev _ (eq ▸ smaller) w rfl ((valid hv).1 w edge) pos)
  · intro path
    clear hi
    induction path with
    | here hv hit => exact (valid hv).2.1 hit
    | @step v w hv edge _ child =>
      by_cases hz : rank g k v = 0
      · have zero := (valid hv).2.2.1 hz w edge
        omega
      · omega

theorem zero_iff_no_path (g : Graph) (k : Nat) (ok : check g k = true)
    {i : Nat} (hi : i < g.size) : rank g k i = 0 ↔ ¬ Reaches g k i := by
  rw [← reachability_iff g k ok hi]
  omega

/-- A corrupted certificate which hides a reachable event must be rejected. -/
example : check #[⟨[1], 0, #[0]⟩, ⟨[], 1, #[1]⟩] 0 = false := by decide
/-- A cycle cannot supply a descending witness for an event that is absent. -/
example : check #[⟨[0], 0, #[1]⟩] 0 = false := by decide
example : check #[⟨[1], 0, #[2]⟩, ⟨[], 1, #[1]⟩] 0 = true := by decide

end PricingAudit.ControlFlow
