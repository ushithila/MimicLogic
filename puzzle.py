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
  "rules_version": RULES_VERSION,
  "meta": {...}
}
"""

from __future__ import annotations

import itertools
import secrets
import time
from typing import Callable

# Bump this whenever a rule changes what a statement means. Rooms saved under an
# older version (e.g. in the room bank) are discarded instead of being shown.
#   1: first version
#   2: "diagonal" means a complete edge-to-edge diagonal; no giveaway or duplicate statements
RULES_VERSION = 2


def make_room_id(level: int | None = None) -> str:
    """A unique room id, e.g. 'L2-20261104-130712-a3f9c1' (level, time built, random)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    prefix = f"L{level}-" if level else ""
    return f"{prefix}{stamp}-{secrets.token_hex(3)}"

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

# Exact meaning of each pattern statement, shown with the rules.
PATTERN_RULES = [
    "Same row / same column: there are at least 2 mimics, all in one row (or column). "
    "Gaps between them are fine.",
    "Diagonal: the mimics fill one complete diagonal, corner to corner or edge to edge "
    "(at least 2 chests long), and there are no other mimics.",
    "L shape: exactly 3 mimics, filling 3 of the 4 cells of a 2x2 square.",
    "No two mimics next to each other: no mimic is directly left, right, above or below "
    "another mimic.",
]

INSTRUCTIONS = [
    "Read every chest's statement.",
    "Use the rules to work out which chests must be Mimics.",
    "Mark each chest as Normal or Mimic.",
    "Submit your answer. Open only the Normal chests to claim the loot!",
]

# --------------------------------------------------------------------------
# Patterns (statement type "pattern"); each takes the set of mimic
# positions {(row, col)} and the grid size (rows, cols), and returns True/False.
# --------------------------------------------------------------------------

def _same_row(cells, grid):
    return len(cells) >= 2 and len({r for r, _ in cells}) == 1


def _same_col(cells, grid):
    return len(cells) >= 2 and len({c for _, c in cells}) == 1


def _l_shape(cells, grid):
    """Exactly 3 mimics filling 3 cells of a 2x2 block (any rotation)."""
    if len(cells) != 3:
        return False
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    return (len(rows) == 2 and len(cols) == 2
            and max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1)


def full_diagonals(grid) -> list[set]:
    """Every complete diagonal of the grid (edge to edge, 2+ cells), both directions."""
    rows, cols = grid
    lines = []
    for k in range(-(cols - 1), rows):             # down-right: row - col == k
        lines.append({(r, r - k) for r in range(rows) if 0 <= r - k < cols})
    for k in range(rows + cols - 1):               # down-left: row + col == k
        lines.append({(r, k - r) for r in range(rows) if 0 <= k - r < cols})
    return [d for d in lines if len(d) >= 2]


def _diagonal(cells, grid):
    """The mimics fill one complete diagonal (edge to edge, 2+ long) and nothing else.
    A diagonal with an empty grid cell can never be filled, so it never counts."""
    return any(set(cells) == d for d in full_diagonals(grid))


def _none_adjacent(cells, grid):
    return all(abs(r1 - r2) + abs(c1 - c2) != 1
               for (r1, c1), (r2, c2) in itertools.combinations(cells, 2))


PATTERNS: dict[str, tuple[Callable, str]] = {
    "same_row":     (_same_row,      "All the mimics are in the same row."),
    "same_col":     (_same_col,      "All the mimics are in the same column."),
    "l_shape":      (_l_shape,       "The mimics form an L shape."),
    "diagonal":     (_diagonal,      "The mimics fill a diagonal line."),
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


def grid_of(room: dict) -> tuple[int, int]:
    return room["grid"]["rows"], room["grid"]["cols"]


def statement_truth(stmt: dict, is_mimic: dict, pos: dict, grid=None) -> bool:
    """Is this statement true, given is_mimic {chest_id: bool}?
    grid = (rows, cols); needed by the diagonal pattern (inferred from pos if omitted)."""
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
        if grid is None:
            grid = (max(r for r, _ in pos.values()) + 1, max(c for _, c in pos.values()) + 1)
        return PATTERNS[tg[0]][0](cells, grid)
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
# Statement quality: duplicates and giveaways
# --------------------------------------------------------------------------

SYMMETRIC = {"either_one", "same_kind"}


def statement_key(stmt: dict) -> tuple:
    """What a statement says, ignoring who says it ('A and B' == 'B and A')."""
    tg = sorted(stmt["targets"]) if stmt["type"] in SYMMETRIC else list(stmt["targets"])
    return (stmt["type"], tuple(tg))


def is_giveaway(stmt: dict, room: dict) -> bool:
    """True if the statement's truth is settled before solving anything: it is the same
    in every arrangement allowed by the mimic-count hint (or in every arrangement at all,
    when the count is hidden). E.g. 'The mimics fill a diagonal line' with 1 mimic is
    always false, so the speaker is obviously a Mimic."""
    ids = [c["id"] for c in room["chests"]]
    pos = {c["id"]: (c["row"], c["col"]) for c in room["chests"]}
    grid = grid_of(room)
    h = room.get("hints", {})
    if h.get("show_mimic_count"):
        sets = itertools.combinations(ids, h["mimic_count"])
    else:
        sets = (combo for k in range(1, len(ids) + 1)
                for combo in itertools.combinations(ids, k))
    seen = set()
    for mimics in sets:
        m = {c: c in mimics for c in ids}
        seen.add(statement_truth(stmt, m, pos, grid))
        if len(seen) == 2:
            return False
    return True


def design_problems(room: dict) -> list[str]:
    """Statements that make a room worse without making it wrong."""
    problems, said = [], {}
    for s in room["statements"]:
        key = statement_key(s)
        if key in said:
            problems.append(f"{s['speaker']} and {said[key]} make the same statement "
                            f"(\"{statement_text(s)}\").")
        said.setdefault(key, s["speaker"])
        if is_giveaway(s, room):
            problems.append(f"{s['speaker']}'s statement \"{statement_text(s)}\" is settled by "
                            "the mimic count alone, which gives the answer away.")
    return problems


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------

def _constraints(room: dict):
    """Turn a room into (variables, [constraint(vars, check)])."""
    ids = [c["id"] for c in room["chests"]]
    pos = {c["id"]: (c["row"], c["col"]) for c in room["chests"]}
    grid = grid_of(room)
    cons = []
    for s in room["statements"]:
        def check(m, s=s):  # a statement is true exactly when its speaker is Normal
            return statement_truth(s, m, pos, grid) == (not m[s["speaker"]])
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
                return False, (f"You marked {sum(m.values())} mimic(s), but this room has "
                               f"{room['hints']['mimic_count']}.")
            who = "Normal" if not m[s["speaker"]] else "a Mimic"
            truth = "false" if not m[s["speaker"]] else "true"
            return False, (f"You marked {s['speaker']} as {who}, but its statement "
                           f"\"{statement_text(s)}\" would be {truth}.")
    if answer != room["solution"]:
        return False, "This fits the clues, but it isn't the room's answer."
    return True, "Correct!"


HIDDEN_COUNT_LINES = [
    "The mimics have aggressively taken over the hint system to hide their numbers.",
    "The mimics chewed up the hint board. You're on your own.",
    "Someone (something?) has scratched out the mimic count.",
    "The mimic count was here. It has been eaten.",
]


def mimic_label(room: dict) -> str:
    """The 'Mimics:' line shown above the grid."""
    h = room.get("hints", {})
    if h.get("show_mimic_count"):
        return f"Mimics: {h['mimic_count']}"
    line = HIDDEN_COUNT_LINES[sum(map(ord, str(room.get("id", "")))) % len(HIDDEN_COUNT_LINES)]
    return f"Mimics: ???  ({line})"


def rules_text() -> list[str]:
    lines = ["RULES"] + [f"  - {r}" for r in RULES]
    lines += ["", "PATTERNS"] + [f"  - {r}" for r in PATTERN_RULES]
    lines += ["", "HOW TO PLAY"] + [f"  {i}. {s}" for i, s in enumerate(INSTRUCTIONS, 1)]
    return lines


def render(room: dict, show_rules: bool = True, title: str | None = None) -> str:
    """Plain-text view of a room for the terminal.
    show_rules: include the rules first. title: header line (default 'ROOM <id>')."""
    rows, cols = room["grid"]["rows"], room["grid"]["cols"]
    at = {(c["row"], c["col"]): c["id"] for c in room["chests"]}
    lines = rules_text() + [""] if show_rules else []
    lines += [title or f"ROOM {room['id']}", mimic_label(room), ""]
    for r in range(rows):
        lines.append("    " + "   ".join(f"[{at[(r, c)]}]" if (r, c) in at else " . "
                                          for c in range(cols)))
    lines.append("")
    said = {s["speaker"]: s for s in room["statements"]}
    for c in room["chests"]:
        s = said.get(c["id"])
        lines.append(f"  {c['id']}: " + (f"\"{statement_text(s)}\"" if s else "(says nothing)"))
    return "\n".join(lines)
