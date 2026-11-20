import PricingAudit.ControlFlow

open Lean PricingAudit.ControlFlow

private structure Body where
  name : String
  blocks : Array Block
  deriving FromJson

/-- Use the kernel's definitional equality on the closed expression. -/
private def proveTrue (env : Environment) (test : Expr) : IO Unit :=
  match Kernel.isDefEq env {} test (mkConst ``Bool.true) with
  | .ok true => pure ()
  | .ok false => throw <| IO.userError "A closed kernel obligation reduced to false"
  | .error _ => throw <| IO.userError "Lean kernel could not reduce an obligation"

def main (args : List String) : IO Unit := do
  initSearchPath (← findSysroot)
  let env ← importModules #[{module := `PricingAudit.ControlFlow}] {}
  for path in args do
    let input ← IO.FS.Handle.mk path .read
    let mut seenBounds : Std.HashSet (Nat × List Nat) := {}
    let mut seenRanks : Std.HashSet (Bool × Nat × List Nat) := {}
    let mut bodies := 0
    let mut blocks := 0
    let mut counts := #[0, 0, 0, 0, 0, 0]
    repeat
      let line ← input.getLine
      if line.isEmpty then break
      let parsed := Json.parse line >>= fromJson? (α := Body)
      let b ← IO.ofExcept parsed
      if b.blocks.isEmpty || !(b.blocks.all fun row => row.ranks.size == counts.size && row.mask < 64) then
        throw <| IO.userError s!"Invalid block layout in {b.name}"
      for k in [:counts.size] do
        unless check b.blocks k do
          throw <| IO.userError s!"Invalid reachability certificate in {b.name}, category {k}"
        if rank b.blocks k 0 > 0 then counts := counts.modify k (· + 1)
      for i in [:b.blocks.size] do
        let row := block b.blocks i
        let boundKey := (b.blocks.size, row.edges)
        unless seenBounds.contains boundKey do
          proveTrue env (mkApp2 (mkConst ``boundsOK) (toExpr b.blocks.size) (toExpr row.edges))
          seenBounds := seenBounds.insert boundKey
        for k in [:counts.size] do
          let hit := row.mask.testBit k
          let r := rank b.blocks k i
          let next := row.edges.map (rank b.blocks k)
          let rankKey := (hit, r, next)
          unless seenRanks.contains rankKey do
            proveTrue env (mkApp3 (mkConst ``localOK) (toExpr hit) (toExpr r) (toExpr next))
            seenRanks := seenRanks.insert rankKey
      bodies := bodies + 1
      blocks := blocks + b.blocks.size
    if bodies = 0 then throw <| IO.userError s!"Empty corpus: {path}"
    IO.println s!"{path}: bodies={bodies} blocks={blocks} entry_reach={counts} kernel_bounds={seenBounds.size} kernel_ranks={seenRanks.size}"
