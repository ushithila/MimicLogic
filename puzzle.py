"""
Mimic Logic - core puzzle model, rules, and solver.

Shared by the generator, the solver agent and the game screen.
Basic rules only: Normal chests tell the truth, Mimics lie.

A room follows the format in the team's Demo Plan:
{
  "id": ..., "grid": {"rows", "cols"},
  "chests": [{"id", "row", "col"}],
  "statements": [{"speaker", "type", "targets", "phrase"?}],
  "hints": {"mimic_count", "show_mimic_count"},
  "solution": {chest_id: "normal" | "mimic"},
  "meta": {...}
}
"""

from __future__ import annotations

import itertools
from typing import Callable

# --------------------------------------------------------------------------
# Rules and instructions shown with every room
# --------------------------------------------------------------------------

RULES = [
    "Every chest is either a Normal chest or a Mimic.",
    "Normal chests always tell the truth.",
    "Mimics always lie.",
    "Chests that say nothing could be either kind.",
    "'Next to' means directly left, right, above or below (not diagonal).",
    "Every room has exactly one solution.",
]

INSTRUCTIONS = [
    "Read every chest's statement.",
    "Use the rules to work out which chests must be Mimics.",
    "Mark each chest as Normal or Mimic.",
    "Submit your answer. Open only the Normal chests to claim the loot!",
]

# --------------------------------------------------------------------------
# Patterns (statement type "pattern"); each takes the set of mimic
# positions {(row, col)} and returns True/False.
# --------------------------------------------------------------------------

def _same_row(cells):
    return len(cells) >= 2 and len({r for r, _ in cells}) == 1


def _same_col(cells):
    return len(cells) >= 2 and len({c for _, c in cells}) == 1


def _l_shape(cells):
    """Exactly 3 mimics filling 3 cells of a 2x2 block (any rotation)."""
    if len(cells) != 3:
        return False
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    return (len(rows) == 2 and len(cols) == 2
            and max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1)


def _diagonal(cells):
    """2+ mimics on one diagonal line, no gaps."""
    if len(cells) < 2:
        return False
    pts = sorted(cells)
    for step in ((1, 1), (1, -1)):
        r0, c0 = pts[0]
        line = [(r0 + i * step[0], c0 + i * step[1]) for i in range(len(pts))]
        if sorted(line) == pts:
            return True
    return False


def _none_adjacent(cells):
    return all(abs(r1 - r2) + abs(c1 - c2) != 1
               for (r1, c1), (r2, c2) in itertools.combinations(cells, 2))


PATTERNS: dict[str, tuple[Callable, str]] = {
    "same_row":     (_same_row,      "All the mimics are in the same row."),
    "same_col":     (_same_col,      "All the mimics are in the same column."),
    "l_shape":      (_l_shape,       "The mimics form an L shape."),
    "diagonal":     (_diagonal,      "The mimics form a diagonal line."),
    "none_adjacent": (_none_adjacent, "No two mimics are next to each other."),
}

# --------------------------------------------------------------------------
# Statement types
# --------------------------------------------------------------------------

STATEMENT_TYPES = ["is_mimic", "not_mimic", "either_one", "if_then", "same_kind", "pattern"]

# How many chest targets each type takes (pattern takes a shape name instead).
ARITY = {"is_mimic": 1, "not_mimic": 1, "either_one": 2,
         "if_then": 2, "same_kind": 2, "pattern": 0}

DIRECTIONS = {(0, 1): "right", (0, -1): "left", (1, 0): "below", (-1, 0): "above"}
DIRECTION_WORDS = {"right": "the chest to my right", "left": "the chest to my left",
                   "below": "the chest below me", "above": "the chest above me"}


def statement_truth(stmt: dict, is_mimic: dict, pos: dict) -> bool:
    """Is this statement true, given is_mimic {chest_id: bool}?"""
    t, tg = stmt["type"], stmt["targets"]
    if t == "is_mimic":
        return is_mimic[tg[0]]
    if t == "not_mimic":
        return not is_mimic[tg[0]]
    if t == "either_one":
        return is_mimic[tg[0]] != is_mimic[tg[1]]
    if t == "if_then":
        return (not is_mimic[tg[0]]) or is_mimic[tg[1]]
    if t == "same_kind":
        return is_mimic[tg[0]] == is_mimic[tg[1]]
    if t == "pattern":
        cells = {pos[c] for c, m in is_mimic.items() if m}
        return PATTERNS[tg[0]][0](cells)
    raise ValueError(f"unknown statement type {t}")


def statement_vars(stmt: dict, chest_ids: list) -> list:
    """Chests whose type this statement's truth depends on (speaker included)."""
    if stmt["type"] == "pattern":
        return list(chest_ids)
    return [stmt["speaker"]] + list(stmt["targets"])


def statement_text(stmt: dict) -> str:
    """Readable wording, e.g. 'The chest to my right is a mimic.'"""
    t, tg, phrase = stmt["type"], stmt["targets"], stmt.get("phrase")

    def name(i):
        if i == 0 and phrase and ARITY[t] == 1:
            return DIRECTION_WORDS[phrase]
        return f"Chest {tg[i]}"

    def cap(s):
        return s[0].upper() + s[1:]

    if t == "is_mimic":
        return cap(f"{name(0)} is a mimic.")
    if t == "not_mimic":
        return cap(f"{name(0)} is not a mimic.")
    if t == "either_one":
        return f"Exactly one of Chest {tg[0]} and Chest {tg[1]} is a mimic."
    if t == "if_then":
        return f"If Chest {tg[0]} is a mimic, then Chest {tg[1]} is a mimic."
    if t == "same_kind":
        return f"Chest {tg[0]} and Chest {tg[1]} are the same kind."
    if t == "pattern":
        return PATTERNS[tg[0]][1]
    raise ValueError(t)


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

def _constraints(room: dict):
    """Turn a room into (variables, [constraint(vars, check)])."""
    ids = [c["id"] for c in room["chests"]]
    pos = {c["id"]: (c["row"], c["col"]) for c in room["chests"]}
    cons = []
    for s in room["statements"]:
        def check(m, s=s):  # a statement is true exactly when its speaker is Normal
            return statement_truth(s, m, pos) == (not m[s["speaker"]])
        cons.append((statement_vars(s, ids), check, s))
    h = room.get("hints", {})
    if h.get("show_mimic_count"):
        k = h["mimic_count"]
        cons.append((ids, lambda m, k=k: sum(m.values()) == k, None))
    return ids, cons


def solve_all(room: dict, limit: int | None = None) -> list[dict]:
    """Brute force: every assignment that fits. Ground truth for testing."""
    ids, cons = _constraints(room)
    out = []
    for bits in itertools.product((False, True), repeat=len(ids)):
        m = dict(zip(ids, bits))
        if all(check(m) for _, check, _ in cons):
            out.append({c: ("mimic" if v else "normal") for c, v in m.items()})
            if limit and len(out) >= limit:
                break
    return out


def solve_with_trace(room: dict) -> dict:
    """
    Human-style solver: forced deductions first, case split only when stuck.
    Returns solution(s), a readable trace and metrics (nodes, case splits).
    """
    ids, cons = _constraints(room)
    stats = {"nodes": 0, "case_splits": 0, "max_depth": 0}
    trace: list[str] = []
    solutions: list[dict] = []

    def kind(v):
        return "a Mimic" if v else "Normal"

    def label(s):
        return f"{s['speaker']}'s statement" if s else "the mimic-count hint"

    def consistent(assign):
        for vs, check, _ in cons:
            if all(v in assign for v in vs) and not check({v: assign[v] for v in vs}):
                return False
        return True

    def propagate(assign, depth):
        """Repeatedly apply constraints with exactly one unknown chest."""
        changed = True
        while changed:
            changed = False
            # mimic-count hint: once the count is reached (or every unknown
            # chest is needed to reach it), the rest follow at once
            h = room.get("hints", {})
            if h.get("show_mimic_count"):
                k = h["mimic_count"]
                found = sum(assign.get(v, False) for v in ids)
                unknown = [v for v in ids if v not in assign]
                if found > k or found + len(unknown) < k:
                    trace.append("  " * depth + "Contradiction in the mimic-count hint.")
                    return False
                if unknown and (found == k or found + len(unknown) == k):
                    val = found < k
                    for v in unknown:
                        assign[v] = val
                    trace.append("  " * depth + f"From the mimic-count hint: "
                                 f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} "
                                 f"{'Mimic' if val else 'Normal'}{'s' if val and len(unknown) > 1 else ''}"
                                 ".")
                    changed = True
            for vs, check, s in cons:
                unknown = [v for v in vs if v not in assign]
                if len(unknown) != 1:
                    continue
                u = unknown[0]
                ok = [val for val in (False, True)
                      if check({**{v: assign[v] for v in vs if v in assign}, u: val})]
                if not ok:
                    trace.append("  " * depth + f"Contradiction in {label(s)}.")
                    return False
                if len(ok) == 1:
                    assign[u] = ok[0]
                    trace.append("  " * depth + f"From {label(s)}: {u} is {kind(ok[0])}.")
                    changed = True
            if not consistent(assign):
                trace.append("  " * depth + "Contradiction.")
                return False
        return True

    def search(assign, depth):
        stats["nodes"] += 1
        if not propagate(assign, depth):
            return
        unknown = [v for v in ids if v not in assign]
        if not unknown:
            solutions.append({c: ("mimic" if v else "normal") for c, v in assign.items()})
            trace.append("  " * depth + "Everything fits.")
            return
        # split on the chest that appears in the most constraints
        u = max(unknown, key=lambda v: sum(v in vs for vs, _, _ in cons))
        stats["case_splits"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth + 1)  # nested assumptions
        for val in (True, False):
            trace.append("  " * depth + f"Assume {u} is {kind(val)}:")
            search(dict(assign) | {u: val}, depth + 1)
            if len(solutions) > 1:
                return

    search({}, 0)
    return {"solutions": solutions, "trace": trace, **stats}


def check_answer(room: dict, answer: dict) -> tuple[bool, str]:
    """Is a player's answer right? If not, name the first rule it breaks."""
    ids, cons = _constraints(room)
    m = {c: answer.get(c) == "mimic" for c in ids}
    for vs, check, s in cons:
        if not check({v: m[v] for v in vs}):
            if s is None:
                return False, "The number of mimics doesn't match the hint."
            who = "Normal" if not m[s["speaker"]] else "a Mimic"
            truth = "false" if not m[s["speaker"]] else "true"
            return False, (f"You marked {s['speaker']} as {who}, but its statement "
                           f"\"{statement_text(s)}\" would be {truth}.")
    if answer != room["solution"]:
        return False, "This fits the clues, but it isn't the room's answer."
    return True, "Correct!"


def render(room: dict) -> str:
    """Plain-text view of a room for the terminal."""
    rows, cols = room["grid"]["rows"], room["grid"]["cols"]
    at = {(c["row"], c["col"]): c["id"] for c in room["chests"]}
    lines = ["RULES"] + [f"  - {r}" for r in room.get("rules", RULES)]
    lines += ["", "HOW TO PLAY"] + [f"  {i}. {s}" for i, s in
                                    enumerate(room.get("instructions", INSTRUCTIONS), 1)]
    lines += ["", f"ROOM {room['id']}", ""]
    for r in range(rows):
        lines.append("    " + "   ".join(f"[{at[(r, c)]}]" if (r, c) in at else " . "
                                          for c in range(cols)))
    lines.append("")
    said = {s["speaker"]: s for s in room["statements"]}
    for c in room["chests"]:
        s = said.get(c["id"])
        lines.append(f"  {c['id']}: " + (f"\"{statement_text(s)}\"" if s else "(says nothing)"))
    h = room.get("hints", {})
    if h.get("show_mimic_count"):
        n = h["mimic_count"]
        lines += ["", f"  Hint: this room has {n} mimic{'s' if n != 1 else ''}."]
    return "\n".join(lines)
