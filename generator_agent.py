"""
Mimic Logic - Puzzle Generator Agent.

The agent runs a simple loop:
  1. DECIDE  - turn the player profile into generation settings
  2. BUILD   - pick a hidden answer, then write statements that fit it
  3. VERIFY  - ask the solver whether the room has exactly one answer
  4. REVISE  - if ambiguous, rewrite a statement from a chest the competing
               answers disagree on, and verify again
  5. TAG     - record skills used and solver metrics in the room's meta

Usage:
  python generator_agent.py                       # one room, default settings
  python generator_agent.py --chests 5 --grid 3x3 --count 3
  python generator_agent.py --focus if_then --hide-count
  python generator_agent.py --json rooms.json     # also save as JSON
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field

from puzzle import (ARITY, DIRECTIONS, INSTRUCTIONS, PATTERNS, RULES, RULES_VERSION,
                    STATEMENT_TYPES, is_giveaway, make_room_id, render, solve_all,
                    solve_with_trace, statement_key, statement_truth)


@dataclass
class Settings:
    rows: int = 2
    cols: int = 2
    num_chests: int = 4                 # must fit in rows x cols
    allowed_types: list = field(default_factory=lambda: list(STATEMENT_TYPES))
    focus_type: str | None = None       # statement type to lean on (adaptive hook)
    focus_weight: float = 4.0           # how strongly to prefer the focus type
    show_mimic_count: bool = True
    silent_chests: int = 0              # chests that say nothing
    min_mimics: int = 1
    max_mimics: int | None = None       # default: about half the chests
    minimize: bool = False              # drop statements that aren't needed
    max_repairs: int = 12
    max_attempts: int = 400


class GeneratorAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.level = None
        self._giveaway_cache = {}

    # ---------------------------------------------------------------- DECIDE
    def decide_settings(self, profile: dict | None = None) -> Settings:
        """
        Map a simple player profile to settings.
        profile = {"level": 1-5, "weak_type": "if_then" | None}
        """
        profile = profile or {}
        level = max(1, min(5, profile.get("level", 1)))
        table = {  # level: (rows, cols, chests, types, show_count, silent)
            1: (2, 2, 3, ["is_mimic", "not_mimic"], True, 0),
            2: (2, 2, 4, ["is_mimic", "not_mimic", "either_one", "same_kind"], True, 0),
            3: (2, 3, 5, ["is_mimic", "not_mimic", "either_one", "same_kind", "if_then"], True, 0),
            4: (3, 3, 5, list(STATEMENT_TYPES), True, 1),
            5: (3, 3, 6, list(STATEMENT_TYPES), False, 1),
        }
        rows, cols, n, types, show, silent = table[level]
        weak = profile.get("weak_type")
        if weak and weak not in types:
            types = types + [weak]
        return Settings(rows=rows, cols=cols, num_chests=n, allowed_types=types,
                        focus_type=weak, show_mimic_count=show, silent_chests=silent,
                        minimize=level >= 3)

    # ---------------------------------------------------------------- BUILD
    def _layout(self, s: Settings):
        cells = [(r, c) for r in range(s.rows) for c in range(s.cols)]
        chosen = sorted(self.rng.sample(cells, s.num_chests))
        ids = [chr(ord("A") + i) for i in range(s.num_chests)]
        return [{"id": i, "row": r, "col": c} for i, (r, c) in zip(ids, chosen)]

    def _pick_answer(self, chests, s: Settings):
        n = len(chests)
        hi = s.max_mimics if s.max_mimics is not None else max(1, n // 2)
        k = self.rng.randint(s.min_mimics, max(s.min_mimics, hi))
        mimics = set(self.rng.sample([c["id"] for c in chests], k))
        return {c["id"]: c["id"] in mimics for c in chests}

    def _candidates(self, speaker, chests, answer, pos, s: Settings, room=None, taken=()):
        """Every statement this speaker could make that fits the answer, skipping
        statements another chest already makes (taken) and giveaways: statements
        whose truth is settled by the mimic count alone."""
        others = [c["id"] for c in chests if c["id"] != speaker]
        by_pos = {(c["row"], c["col"]): c["id"] for c in chests}
        must_be_true = not answer[speaker]
        grid = (s.rows, s.cols)
        out = []
        for t in s.allowed_types:
            if t == "pattern":
                options = [[name] for name in PATTERNS]
            elif ARITY[t] == 1:
                options = [[o] for o in others]
            else:
                options = [[a, b] for a in others for b in others if a != b
                           and not (t in ("either_one", "same_kind") and a > b)]
            for tg in options:
                stmt = {"speaker": speaker, "type": t, "targets": tg}
                if statement_truth(stmt, answer, pos, grid) != must_be_true:
                    continue
                key = statement_key(stmt)
                if key in taken:
                    continue
                if room is not None:
                    if key not in self._giveaway_cache:
                        self._giveaway_cache[key] = is_giveaway(stmt, room)
                    if self._giveaway_cache[key]:
                        continue
                if ARITY[t] == 1:  # use a direction word when the target is next door
                    r, c = pos[speaker]
                    for (dr, dc), word in DIRECTIONS.items():
                        if by_pos.get((r + dr, c + dc)) == tg[0]:
                            stmt["phrase"] = word
                out.append(stmt)
        return out

    def _choose(self, cands, s: Settings):
        weights = [s.focus_weight if c["type"] == s.focus_type else 1.0 for c in cands]
        # prefer each type equally, not each target combination
        per_type = {}
        for c in cands:
            per_type[c["type"]] = per_type.get(c["type"], 0) + 1
        weights = [w / per_type[c["type"]] for w, c in zip(weights, cands)]
        return self.rng.choices(cands, weights=weights)[0]

    # ---------------------------------------------------------------- GENERATE
    def generate(self, settings: Settings | None = None, profile: dict | None = None) -> dict:
        s = settings or self.decide_settings(profile)
        self.level = (profile or {}).get("level")
        if s.num_chests > s.rows * s.cols:
            raise ValueError("More chests than grid cells.")
        for attempt in range(1, s.max_attempts + 1):
            room = self._attempt(s)
            if room:
                room["meta"]["attempts"] = attempt
                return room
        raise RuntimeError("No uniquely solvable room found; loosen the settings.")

    def _attempt(self, s: Settings):
        chests = self._layout(s)
        pos = {c["id"]: (c["row"], c["col"]) for c in chests}
        answer = self._pick_answer(chests, s)
        speakers = [c["id"] for c in chests]
        silent = set(self.rng.sample(speakers, min(s.silent_chests, len(speakers) - 1)))
        room = {
            "id": make_room_id(self.level),
            "grid": {"rows": s.rows, "cols": s.cols},
            "chests": chests,
            "statements": [],
            "hints": {"mimic_count": sum(answer.values()),
                      "show_mimic_count": s.show_mimic_count},
            "solution": {k: ("mimic" if v else "normal") for k, v in answer.items()},
            "rules_version": RULES_VERSION,
            "rules": RULES,
            "instructions": INSTRUCTIONS,
            "meta": {},
        }
        self._giveaway_cache = {}
        stmts = {}
        for sp in speakers:
            if sp in silent:
                continue
            taken = {statement_key(x) for x in stmts.values()}
            cands = self._candidates(sp, chests, answer, pos, s, room, taken)
            if cands:
                stmts[sp] = self._choose(cands, s)
        room["statements"] = [stmts[k] for k in speakers if k in stmts]

        # VERIFY + REVISE
        repairs = 0
        while True:
            sols = solve_all(room, limit=2)
            if len(sols) == 1:
                break
            if repairs >= s.max_repairs or not stmts:
                return None
            repairs += 1
            # rewrite a statement from a chest the two answers disagree on
            a, b = sols[0], sols[1]
            disputed = [c for c in speakers if a[c] != b[c] and c in stmts and c not in silent]
            pick = self.rng.choice(disputed or list(stmts))
            taken = {statement_key(x) for k, x in stmts.items() if k != pick}
            cands = self._candidates(pick, chests, answer, pos, s, room, taken)
            if not cands:
                return None
            stmts[pick] = self._choose(cands, s)
            room["statements"] = [stmts[k] for k in speakers if k in stmts]

        if s.minimize:
            # never strip a room below "one more silent chest than asked for"
            floor = max(2, s.num_chests - s.silent_chests - 1)
            self._minimize(room, floor)

        # TAG
        result = solve_with_trace(room)
        types = sorted({st["type"] for st in room["statements"]})
        room["meta"] = {
            "skills": types,
            "focus_type": s.focus_type,
            "repairs": repairs,
            "assumption_depth": result["max_depth"],
            "case_splits": result["case_splits"],
            "search_nodes": result["nodes"],
            "difficulty": self._difficulty(room, result),
        }
        return room

    def _minimize(self, room, floor: int):
        """Drop statements one at a time while the room stays uniquely solvable."""
        order = list(range(len(room["statements"])))
        self.rng.shuffle(order)
        keep = list(room["statements"])
        for i in order:
            if len(keep) <= floor:
                break
            trial = [st for st in keep if st is not room["statements"][i]]
            if len(trial) == len(keep):
                continue
            if len(solve_all({**room, "statements": trial}, limit=2)) == 1:
                keep = trial
        room["statements"] = keep

    @staticmethod
    def _difficulty(room, result) -> int:
        """
        Rough 1-5 score. The main signal is how many nested assumptions
        ("assume A is a mimic... then assume C is...") the solver needed.
        Compound statements, silent chests, a hidden mimic count and
        bigger rooms add a little on top.
        """
        compound = sum(st["type"] in ("either_one", "if_then", "same_kind", "pattern")
                       for st in room["statements"])
        silent = len(room["chests"]) - len(room["statements"])
        hidden = 0 if room["hints"]["show_mimic_count"] else 1
        big = 1 if len(room["chests"]) >= 5 else 0
        score = (result["max_depth"] + 0.25 * compound + 0.5 * silent
                 + 0.5 * hidden + 0.5 * big)
        return max(1, min(5, round(score)))


# -------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description="Generate Mimic Logic rooms.")
    p.add_argument("--level", type=int, help="1-5; picks grid, chests and statement types")
    p.add_argument("--grid", default=None, help="e.g. 2x2, 2x3, 3x3")
    p.add_argument("--chests", type=int, default=None)
    p.add_argument("--types", default=None,
                   help="comma list from: " + ",".join(STATEMENT_TYPES))
    p.add_argument("--focus", default=None, help="statement type to lean on")
    p.add_argument("--hide-count", action="store_true", help="don't show the mimic count")
    p.add_argument("--silent", type=int, default=None, help="chests that say nothing")
    p.add_argument("--minimize", action="store_true", help="remove unneeded statements")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", default=None, help="save rooms to this JSON file")
    p.add_argument("--answers", action="store_true", help="print answers and solver trace")
    a = p.parse_args()

    agent = GeneratorAgent(seed=a.seed)
    s = agent.decide_settings({"level": a.level or 2, "weak_type": a.focus})
    if a.grid:
        s.rows, s.cols = map(int, a.grid.lower().split("x"))
    if a.chests:
        s.num_chests = a.chests
    if a.types:
        s.allowed_types = a.types.split(",")
    if a.hide_count:
        s.show_mimic_count = False
    if a.silent is not None:
        s.silent_chests = a.silent
    if a.minimize:
        s.minimize = True

    rooms = [agent.generate(s) for _ in range(a.count)]
    for room in rooms:
        print(render(room))
        m = room["meta"]
        print(f"\n  [difficulty {m['difficulty']}/5 | case splits {m['case_splits']} | "
              f"skills: {', '.join(m['skills'])}]")
        if a.answers:
            sol = room["solution"]
            print("  Answer: " + ", ".join(f"{k}={v}" for k, v in sol.items()))
            print("  Solver trace:")
            for line in solve_with_trace(room)["trace"]:
                print("    " + line)
        print("\n" + "=" * 60 + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rooms, f, indent=2)
        print(f"Saved {len(rooms)} room(s) to {a.json}")


if __name__ == "__main__":
    main()
