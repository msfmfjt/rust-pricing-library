"""Balance isolated wheel modules using timing hints, never a coverage allowlist."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path


REMAINING_GROUPS = tuple(f"remaining-{index}" for index in range(1, 5))
EXAMPLES_GROUP = REMAINING_GROUPS[0]
TIMINGS_PATH = Path(__file__).with_name("remaining_timings.json")


def load_timings() -> dict:
    hints = json.loads(TIMINGS_PATH.read_text(encoding="utf-8"))
    values = [hints["examples_seconds"], *hints["modules_seconds"].values()]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in values):
        raise ValueError("wheel timing hints must be finite nonnegative seconds")
    return hints


def partition_remaining(modules: list[str], *, timings: dict | None = None) -> dict[str, list[str]]:
    """Schedule every discovered module exactly once; new modules get a default weight.

    Each module still runs in its original isolated process. The examples run
    once in the first group and contribute to that group's estimated load.
    Timings affect only placement, never whether a module is selected.
    """
    if len(set(modules)) != len(modules):
        raise RuntimeError("duplicate remaining test modules")
    if len(modules) < len(REMAINING_GROUPS):
        raise RuntimeError("too few modules for nonempty remaining groups")
    hints = load_timings() if timings is None else timings
    weights = hints["modules_seconds"]
    groups = {name: [] for name in REMAINING_GROUPS}
    loads = {name: hints["examples_seconds"] if name == EXAMPLES_GROUP else 0.0
             for name in REMAINING_GROUPS}
    ordered = sorted(modules, key=lambda name: (-weights.get(name, 1.0), name))
    for index, module in enumerate(ordered):
        empty = [name for name in REMAINING_GROUPS if not groups[name]]
        eligible = empty if len(ordered) - index <= len(empty) else REMAINING_GROUPS
        group = min(eligible, key=lambda name: (loads[name], name))
        groups[group].append(module)
        loads[group] += weights.get(module, 1.0)
    for group in groups.values():
        group.sort()
    actual = Counter(module for group in groups.values() for module in group)
    if actual != Counter(modules) or any(not group for group in groups.values()):
        raise RuntimeError("remaining wheel partition is incomplete or overlapping")
    return groups


def estimated_loads(groups: dict[str, list[str]]) -> dict[str, float]:
    """Report historical estimates, not measured candidate execution times."""
    hints = load_timings()
    return {group: sum(hints["modules_seconds"].get(module, 1.0) for module in modules)
            + (hints["examples_seconds"] if group == EXAMPLES_GROUP else 0.0)
            for group, modules in groups.items()}
