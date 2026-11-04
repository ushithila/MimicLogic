"""
Mimic Logic - the ReAct puzzle generator agent as an importable module.

GENERATED from Sections 1-4 of Mimic_Logic_ReAct_Generator.ipynb, so that scripts
(play.py) and the game can use the same agent as the notebook. If you change the
agent in the notebook, make the same change here (or regenerate this file).

Usage:
    from mimic_agent import agent, generate_room
    room, trace = generate_room(agent, level=3, show_trace=False)

Set MIMIC_OFFLINE=1 to use the scripted stand-in instead of the model (no API key).
"""

import os
from dotenv import load_dotenv

load_dotenv()
USE_OFFLINE_MODEL = os.getenv("MIMIC_OFFLINE") == "1"
if not USE_OFFLINE_MODEL and not os.getenv("OPENROUTER_API_KEY"):
    raise RuntimeError("Missing OPENROUTER_API_KEY. Add it to a .env file next to this "
                       "file, or set MIMIC_OFFLINE=1 to use the offline stand-in.")

# Standard library
import os
import json
import random
from typing import Literal

# Shared Mimic Logic code (must be next to this notebook)
from puzzle import (ARITY, DIRECTIONS, INSTRUCTIONS, PATTERN_RULES, PATTERNS, RULES,
                    RULES_VERSION, STATEMENT_TYPES, check_answer, design_problems, grid_of,
                    is_giveaway, make_room_id, render, solve_all, solve_with_trace,
                    statement_key, statement_text, statement_truth)
from generator_agent import GeneratorAgent


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4.1-mini"
TEMPERATURE = 0.7


LEVELS = {
    1: {"grids": ["2x2"], "chests": (3, 3), "types": ["is_mimic", "not_mimic"],
        "show_count": True, "silent": (0, 0), "target_difficulty": (1, 2)},
    2: {"grids": ["2x2", "2x3"], "chests": (4, 4),
        "types": ["is_mimic", "not_mimic", "either_one", "same_kind"],
        "show_count": True, "silent": (0, 0), "target_difficulty": (2, 3)},
    3: {"grids": ["2x3", "3x2"], "chests": (4, 5),
        "types": ["is_mimic", "not_mimic", "either_one", "same_kind", "if_then"],
        "show_count": True, "silent": (0, 1), "target_difficulty": (2, 4)},
    4: {"grids": ["3x3", "2x3"], "chests": (5, 6), "types": list(STATEMENT_TYPES),
        "show_count": True, "silent": (0, 1), "target_difficulty": (3, 5)},
    5: {"grids": ["3x3"], "chests": (5, 7), "types": list(STATEMENT_TYPES),
        "show_count": False, "silent": (1, 1), "target_difficulty": (4, 5)},
}

# (twist, statement type it needs, or None)
TWISTS = [
    ("Two chests accuse each other, and only one of them is right.", "is_mimic"),
    ("The final deduction should come from the mimic-count hint.", None),
    ("Make a corner chest the key to the whole room.", None),
    ("No chest directly calls another chest a mimic.", "either_one"),
    ("A Normal chest should look the most suspicious at first glance.", None),
    ("Use direction words ('the chest below me') for most statements.", None),
    ("Include a conditional whose 'if' part turns out false.", "if_then"),
    ("The mimics should form a recognisable shape on the grid.", "pattern"),
    ("Two chests of the same kind should be far apart on the grid.", "same_kind"),
    ("Let one chest's statement be true but useless on its own.", None),
]

def make_brief(level: int, rng: random.Random) -> dict:
    level = max(1, min(5, int(level)))
    g = LEVELS[level]
    types = list(g["types"])
    n = rng.randint(*g["chests"])
    grids = [gr for gr in g["grids"] if int(gr[0]) * int(gr[2]) >= n] or ["3x3"]
    twists = [t for t, needs in TWISTS if needs is None or needs in types]
    return {
        "level": level,
        "grid": rng.choice(grids),
        "num_chests": n,
        "num_mimics": rng.randint(1, max(1, n // 2)),
        "allowed_types": types,
        "show_mimic_count": g["show_count"],
        "silent_chests": rng.randint(*g["silent"]),
        "target_difficulty": list(g["target_difficulty"]),
        "twist": rng.choice(twists),
    }


SESSION = {}

def start_session(level: int, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    SESSION.clear()
    SESSION.update(brief=make_brief(level, rng), rng=rng,
                   draft=None, draft_unique=False, final=None, checks=0)
    return SESSION["brief"]


def get_design_brief() -> dict:
    return SESSION["brief"]


def _normalize(grid_rows, grid_cols, chests, mimics, statements, show_mimic_count):
    # Validate a draft and convert it to the shared room format. Returns (room, errors).
    errors = []
    brief = SESSION["brief"]
    ids = [c["id"] for c in chests]
    if len(set(ids)) != len(ids):
        errors.append("Chest ids must be unique.")
    cells = [(c["row"], c["col"]) for c in chests]
    if len(set(cells)) != len(cells):
        errors.append("Two chests share a grid cell.")
    for c in chests:
        if not (0 <= c["row"] < grid_rows and 0 <= c["col"] < grid_cols):
            errors.append(f"Chest {c['id']} at row {c['row']}, col {c['col']} is outside the "
                          f"{grid_rows}x{grid_cols} grid (rows and cols start at 0).")
    if not mimics:
        errors.append("A room needs at least one mimic.")
    for m in mimics:
        if m not in ids:
            errors.append(f"Mimic {m} is not one of the chests.")
    speakers = [s["speaker"] for s in statements]
    if len(set(speakers)) != len(speakers):
        errors.append("Each chest can make at most one statement.")
    for s in statements:
        t, tg, sp = s["type"], s["targets"], s["speaker"]
        if sp not in ids:
            errors.append(f"Unknown speaker {sp}.")
        if t not in brief["allowed_types"]:
            errors.append(f"Statement type {t} isn't allowed in this brief "
                          f"(allowed: {brief['allowed_types']}).")
        if t == "pattern":
            if len(tg) != 1 or tg[0] not in PATTERNS:
                errors.append(f"{sp}: a pattern statement's target must be one of {list(PATTERNS)}.")
            continue
        if len(tg) != ARITY[t]:
            errors.append(f"{sp}: {t} takes {ARITY[t]} target(s), got {len(tg)}.")
        if any(x not in ids for x in tg):
            errors.append(f"{sp}'s statement names a chest that doesn't exist.")
        if sp in tg:
            errors.append(f"{sp} talks about itself, which isn't allowed.")
        if len(set(tg)) != len(tg):
            errors.append(f"{sp} names the same chest twice.")
    if errors:
        return None, errors

    # Single-target statements about a neighbour get direction words ("the chest to my right")
    pos = {c["id"]: (c["row"], c["col"]) for c in chests}
    at = {p: i for i, p in pos.items()}
    stmts = []
    for s in statements:
        s = {"speaker": s["speaker"], "type": s["type"], "targets": list(s["targets"])}
        if ARITY[s["type"]] == 1:
            r, c = pos[s["speaker"]]
            for (dr, dc), word in DIRECTIONS.items():
                if at.get((r + dr, c + dc)) == s["targets"][0]:
                    s["phrase"] = word
        stmts.append(s)
    order = {c: i for i, c in enumerate(ids)}
    stmts.sort(key=lambda s: order[s["speaker"]])
    solution = {c: ("mimic" if c in mimics else "normal") for c in ids}
    room = {"id": None, "grid": {"rows": grid_rows, "cols": grid_cols},
            "chests": [dict(c) for c in chests], "statements": stmts,
            "hints": {"mimic_count": len(set(mimics)), "show_mimic_count": show_mimic_count},
            "solution": solution, "rules": RULES, "instructions": INSTRUCTIONS}
    return room, []


def _fitting_statements(room, speaker, prefer_type=None, n=8):
    # Statements this speaker could make that fit the room's intended answer
    # (true for a Normal chest, false for a Mimic). prefer_type options come first.
    # Never suggests a statement another chest already makes, or a giveaway.
    pos = {c["id"]: (c["row"], c["col"]) for c in room["chests"]}
    grid = grid_of(room)
    intended = {c: v == "mimic" for c, v in room["solution"].items()}
    others = [c for c in pos if c != speaker]
    taken = {statement_key(st) for st in room["statements"] if st["speaker"] != speaker}
    giveaway = SESSION.setdefault("giveaway_cache", {})
    options = []
    for t in SESSION["brief"]["allowed_types"]:
        if t == "pattern":
            candidates = [[p] for p in PATTERNS]
        elif ARITY[t] == 1:
            candidates = [[o] for o in others]
        else:
            candidates = [[x, y] for x in others for y in others if x < y or t == "if_then" and x != y]
        for tg in candidates:
            s = {"speaker": speaker, "type": t, "targets": tg}
            if statement_truth(s, intended, pos, grid) == intended[speaker]:
                continue
            key = statement_key(s)
            if key in taken:
                continue
            cache_key = (key, room["hints"]["mimic_count"], room["hints"]["show_mimic_count"],
                         tuple(sorted(pos.items())))
            if cache_key not in giveaway:
                giveaway[cache_key] = is_giveaway(s, room)
            if not giveaway[cache_key]:
                options.append({"type": t, "targets": tg, "reads": statement_text(s)})
    SESSION["rng"].shuffle(options)
    options.sort(key=lambda o: o["type"] != prefer_type)   # stable: preferred type first
    return options[:n]


def _suggest_changes(room, n=3):
    # Search one-statement changes that make the room uniquely solvable with its intended
    # answer, without breaking the brief's silent-chest count. A language model can't run
    # this search in its head; the harness can, in well under a second.
    brief = SESSION["brief"]
    want_silent = brief["silent_chests"]
    lo, hi = brief["target_difficulty"]
    speakers = [st["speaker"] for st in room["statements"]]
    silent_ids = [c["id"] for c in room["chests"] if c["id"] not in speakers]
    silent_now = len(silent_ids)
    trials = []
    if silent_now == want_silent:          # replace one existing statement
        for sp in speakers:
            for o in _fitting_statements(room, sp, n=200):
                new = {"speaker": sp, "type": o["type"], "targets": o["targets"]}
                stmts = [new if st["speaker"] == sp else st for st in room["statements"]]
                trials.append(("replace", sp, new, stmts))
    elif silent_now > want_silent:         # give a silent chest a statement
        for sp in silent_ids:
            for o in _fitting_statements(room, sp, n=200):
                new = {"speaker": sp, "type": o["type"], "targets": o["targets"]}
                trials.append(("add", sp, new, room["statements"] + [new]))
    else:                                  # too few silent chests: remove a statement
        for sp in speakers:
            stmts = [st for st in room["statements"] if st["speaker"] != sp]
            trials.append(("remove", sp, None, stmts))
    SESSION["rng"].shuffle(trials)
    now = len(solve_all(room))
    found, partial = [], []
    for action, sp, new, stmts in trials:
        trial = dict(room, statements=stmts)
        sols = solve_all(trial)
        if room["solution"] not in sols:
            continue
        if len(sols) == 1:
            d = GeneratorAgent._difficulty(trial, solve_with_trace(trial))
            found.append((0 if lo <= d <= hi else min(abs(d - lo), abs(d - hi)), action, sp, new, d))
            if len(found) >= 25:
                break
        elif len(sols) < now:
            partial.append((len(sols), action, sp, new))
    if found:
        found.sort(key=lambda f: f[0])
        picks = [(a, sp, new, {"result": "unique", "difficulty_after": d})
                 for _, a, sp, new, d in found[:n]]
    else:
        # No single change is enough: suggest the ones that remove the most answers.
        partial.sort(key=lambda f: f[0])
        picks = [(a, sp, new, {"result": f"{m} answers left (now {now})"})
                 for m, a, sp, new in partial[:n]]
    out = []
    for action, sp, new, extra in picks:
        item = {"action": action, "speaker": sp, **extra}
        if new:
            item["statement"] = {"type": new["type"], "targets": new["targets"]}
        out.append(item)
    return out


def check_room(grid_rows: int, grid_cols: int, chests: list, mimics: list,
               statements: list, show_mimic_count: bool) -> dict:
    SESSION["checks"] += 1
    room, errors = _normalize(grid_rows, grid_cols, chests, mimics, statements, show_mimic_count)
    if errors:
        SESSION["draft"], SESSION["draft_unique"] = None, False
        return {"status": "format_error", "problems": errors}

    pos = {c["id"]: (c["row"], c["col"]) for c in room["chests"]}
    intended = {c: v == "mimic" for c, v in room["solution"].items()}
    report = {"status": None, "problems": []}

    # 1. Does every statement fit the intended answer? For each one that doesn't,
    #    suggest replacements that do, so the model can swap one in instead of guessing.
    fixes = {}
    for s in room["statements"]:
        true = statement_truth(s, intended, pos, grid_of(room))
        if true == intended[s["speaker"]]:
            role = "a Mimic, so its statement must be FALSE" if intended[s["speaker"]] \
                   else "Normal, so its statement must be TRUE"
            report["problems"].append(
                f"{s['speaker']} is {role}, but \"{statement_text(s)}\" is "
                f"{'true' if true else 'false'} in your intended answer.")
            fixes[s["speaker"]] = [{"type": o["type"], "targets": o["targets"]}
                                   for o in _fitting_statements(room, s["speaker"], s["type"], n=3)]
    if fixes:
        report["suggested_fixes"] = fixes
        report["fix_advice"] = ("Keep the same mimics. Replace only the statements listed in "
                                "suggested_fixes, using one of the suggestions (each one fits your "
                                "intended answer), and check again.")

    # 1b. Duplicate statements and giveaways (statements the mimic count settles alone)
    design = design_problems(room)
    if design:
        report["design_problems"] = design
        seen = set()
        for st in room["statements"]:
            key = statement_key(st)
            if (key in seen or is_giveaway(st, room)) and st["speaker"] not in fixes:
                fixes[st["speaker"]] = [{"type": o["type"], "targets": o["targets"]}
                                        for o in _fitting_statements(room, st["speaker"], st["type"], n=3)]
            seen.add(key)
        report["suggested_fixes"] = fixes
        report["fix_advice"] = ("Keep the same mimics. Replace the statements listed in "
                                "suggested_fixes with one of the suggestions, and check again.")

    # 2. Does the draft follow the brief? (grid, chests, mimics, silent chests, count hint)
    brief = SESSION["brief"]
    rows, cols = map(int, brief["grid"].split("x"))
    silent = len(room["chests"]) - len(room["statements"])
    brief_problems = []
    if (grid_rows, grid_cols) != (rows, cols):
        brief_problems.append(f"The grid is {grid_rows}x{grid_cols}; the brief asks for {brief['grid']}.")
    if len(room["chests"]) != brief["num_chests"]:
        brief_problems.append(f"There are {len(room['chests'])} chests; the brief asks for {brief['num_chests']}.")
    if room["hints"]["mimic_count"] != brief["num_mimics"]:
        brief_problems.append(f"There are {room['hints']['mimic_count']} mimics; the brief asks for {brief['num_mimics']}.")
    if silent != brief["silent_chests"]:
        brief_problems.append(f"{silent} chest(s) say nothing; the brief asks for exactly "
                              f"{brief['silent_chests']} silent chest(s).")
    if show_mimic_count != brief["show_mimic_count"]:
        brief_problems.append(f"show_mimic_count should be {brief['show_mimic_count']}.")
    report["brief_problems"] = brief_problems

    # 3. How many answers does the room have?
    sols = solve_all(room, limit=3)
    SESSION["draft"] = room
    SESSION["draft_unique"] = len(sols) == 1
    if len(sols) == 1 and sols[0] == room["solution"]:
        report["status"] = "unique"
        report["message"] = "Exactly one answer, and it is your intended answer."
    elif len(sols) == 1:
        # Still a valid room: the solver's answer becomes the room's answer.
        # (The intended answer is only a planning aid; chasing it made the model go in circles.)
        room["solution"] = sols[0]
        room["hints"]["mimic_count"] = sum(v == "mimic" for v in sols[0].values())
        report["problems"] = []
        report["brief_problems"] = [p for p in brief_problems if "mimics;" not in p]
        if room["hints"]["mimic_count"] != brief["num_mimics"]:
            report["brief_problems"].append(
                f"The room's answer has {room['hints']['mimic_count']} mimics; "
                f"the brief asks for {brief['num_mimics']}.")
        brief_problems = report["brief_problems"]
        report["status"] = "unique"
        report["message"] = ("Exactly one answer, but not the one you intended. The room is still "
                             "valid, and its answer is the one below. Submit it as it is (if it has "
                             "no brief_problems) rather than rewriting it.")
        report["answer"] = sols[0]
    elif not sols:
        report["status"] = "contradictory"
        report["message"] = "No answer fits all the clues."
    else:
        a, b = sols[0], sols[1]
        report["status"] = "ambiguous"
        report["message"] = (f"{'3 or more' if len(sols) > 2 else 2} answers fit. Add or change "
                             "a statement that tells the differing chests apart.")
        report["competing_answers"] = [a, b]
        report["differing_chests"] = [c for c in a if a[c] != b[c]]
        if not report["problems"]:
            changes = _suggest_changes(room)
            if changes:
                report["suggested_changes"] = changes
                report["change_advice"] = ("Each suggested change was tested (see its result). "
                                           "Apply ONE and check again; if the result says answers "
                                           "are left, check_room will suggest the next step.")

    # 4. Difficulty against the brief
    if sols:
        trace = solve_with_trace(room)
        lo, hi = SESSION["brief"]["target_difficulty"]
        report["target_difficulty"] = [lo, hi]
        if SESSION["draft_unique"]:
            d = GeneratorAgent._difficulty(room, trace)
            report["difficulty"] = d
            if d < lo:
                report["advice"] = "Too easy: use more indirect clues or fewer direct ones."
            elif d > hi:
                report["advice"] = "Too hard: add a clue that gives the player a foothold."
            else:
                report["advice"] = "Difficulty fits the brief."
            if brief_problems or design:
                report["advice"] += " Fix the brief_problems and design_problems before submitting."
            elif lo <= d <= hi:
                report["advice"] += " Ready to submit."
            # Unique, but too few silent chests: find statements that can go.
            if silent < brief["silent_chests"]:
                changes = _suggest_changes(room)
                if changes:
                    report["suggested_changes"] = changes
                    report["change_advice"] = ("The room needs more silent chests. Each suggested "
                                               "removal was tested: the room stays uniquely "
                                               "solvable without that statement. Apply one and "
                                               "check again.")
    report["room_as_player_sees_it"] = [
        f"{s['speaker']}: \"{statement_text(s)}\"" for s in room["statements"]]
    return report


def statement_options(speaker: str, statement_type: str | None = None) -> dict:
    room = SESSION["draft"]
    if room is None:
        return {"error": "Check a valid draft first, so I know the layout and intended answer."}
    if speaker not in {c["id"] for c in room["chests"]}:
        return {"error": f"No chest {speaker} in the draft."}
    options = _fitting_statements(room, speaker, statement_type, n=8)
    if statement_type:
        options = [o for o in options if o["type"] == statement_type]
    role = "a Mimic (its statement must be false)" if room["solution"][speaker] == "mimic" \
           else "Normal (its statement must be true)"
    return {"speaker": speaker, "role": role, "options": options}


def submit_room(design_note: str) -> dict:
    if SESSION["draft"] is None:
        return {"accepted": False, "message": "No valid draft has been checked. Use check_room first."}
    if not SESSION["draft_unique"]:
        return {"accepted": False, "message": "The last checked draft does not have exactly one "
                "answer. Revise it and check again."}
    SESSION["final"] = dict(SESSION["draft"], design_note=design_note)
    return {"accepted": True, "message": "The room has been delivered to the player."}


from pydantic import BaseModel, Field
from langchain.tools import tool

StatementType = Literal["is_mimic", "not_mimic", "either_one", "if_then", "same_kind", "pattern"]

class Chest(BaseModel):
    id: str = Field(description="Chest id: A, B, C... in reading order")
    row: int = Field(description="Row, starting at 0 (top)")
    col: int = Field(description="Column, starting at 0 (left)")

class Statement(BaseModel):
    speaker: str = Field(description="Id of the chest making the statement")
    type: StatementType
    targets: list[str] = Field(description="Chest ids the statement is about; "
                                           "for 'pattern', one pattern name")

@tool("get_design_brief")
def design_brief_tool() -> dict:
    '''Get the design brief for this room: grid, chest count, mimic count, allowed statement
    types, silent chests, mimic-count setting, target difficulty, and twist. Call this first.'''
    return get_design_brief()

@tool("check_room")
def check_room_tool(grid_rows: int, grid_cols: int, chests: list[Chest], mimics: list[str],
                    statements: list[Statement], show_mimic_count: bool) -> dict:
    '''Check a complete draft room. mimics lists the chests you intend to be mimics; every
    other chest is Normal. Returns format problems, statements that break your intended
    answer, brief_problems (where the draft doesn't match the brief), whether the room has
    exactly one answer (and if not, the competing answers and the chests they disagree on),
    and the difficulty compared with the brief.
    Use it after every draft and every revision.'''
    return check_room(grid_rows, grid_cols, [c.model_dump() for c in chests], mimics,
                      [s.model_dump() for s in statements], show_mimic_count)

@tool("statement_options")
def statement_options_tool(speaker: str, statement_type: StatementType | None = None) -> dict:
    '''List some valid statements a chest could make under the last checked draft's intended
    answer. Use it when you are stuck on a revision. Returns a random sample.'''
    return statement_options(speaker, statement_type)

@tool("submit_room")
def submit_room_tool(design_note: str) -> dict:
    '''Deliver the last checked draft to the player. Only accepted if check_room reported
    exactly one answer. design_note: one sentence describing the key deduction.'''
    return submit_room(design_note)

TOOLS = [design_brief_tool, check_room_tool, statement_options_tool, submit_room_tool]


if USE_OFFLINE_MODEL:
    from offline_model import OfflineStandInModel
    model = OfflineStandInModel()
else:
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


pattern_help = "\n".join(f"    {name}: \"{desc}\"" for name, (_, desc) in PATTERNS.items())
pattern_rules = "\n".join(f"  - {r}" for r in PATTERN_RULES)

SYSTEM_PROMPT = f'''You are the Puzzle Generator for Mimic Logic, a dungeon logic game. You design
one room at a time, following a design brief.

Game rules:
- Chests sit on a grid. Each chest is Normal (always tells the truth) or a Mimic (always lies).
- Each chest makes at most one statement; some chests stay silent.
- A room must have exactly one answer.

Statement types (type, targets):
- is_mimic [X]: "Chest X is a mimic."         - not_mimic [X]: "Chest X is not a mimic."
- either_one [X, Y]: "Exactly one of Chest X and Chest Y is a mimic."
- if_then [X, Y]: "If Chest X is a mimic, then Chest Y is a mimic."
- same_kind [X, Y]: "Chest X and Chest Y are the same kind."
- pattern [name], where name is one of:
{pattern_help}
  What the patterns mean exactly:
{pattern_rules}
A chest never talks about itself. No two chests may make the same statement, and avoid
statements the mimic count settles on its own (e.g. a row, column or diagonal pattern when
there is only 1 mimic): they give the answer away. check_room flags both as design_problems.
Statements about a neighbouring chest are worded with
directions automatically ("the chest to my right").

How to work:
- Before each tool call, write one short sentence saying what you are about to do and why.
- First call get_design_brief. Follow the brief: grid, number of chests, number of mimics,
  allowed statement types, silent chests, mimic-count setting, and the twist.
- Decide the hidden answer and the key deduction FIRST, then write statements that lead to it.
  A Normal chest's statement must be TRUE; a Mimic's statement must be FALSE.
- Use ids A, B, C... in reading order (left to right, top to bottom). Rows and cols start at 0.
- Call check_room with the whole draft. Fix every problem it reports. If it reports a unique
  room whose answer differs from the one you intended, that room is still valid: submit it
  (if there are no brief_problems) instead of rewriting it. If it is ambiguous,
  change or add a statement about the chests the competing answers disagree on.
- While revising, keep the same mimics and change only the statements check_room lists as
  broken. Use suggested_fixes: each suggestion fits your intended answer. Do not move the
  mimics or rewrite statements that are already fine.
- If check_room returns suggested_changes, apply one of them: each was tested, and its
  result says whether it makes the room uniquely solvable or removes some of the answers.
- Use statement_options only if you are stuck.
- When check_room reports a unique room with no brief_problems and a difficulty that fits
  the brief, call submit_room.
- Be inventive: vary the layout and the kinds of claims; avoid rooms where every chest says
  the same kind of thing.
- Every reply must call a tool until submit_room is accepted. Never stop to summarise the
  brief or describe a plan in plain text: put your reasoning in the one short sentence before
  the tool call, and keep going.
- After submit_room is accepted, reply with one sentence and stop.'''


from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)

from langchain.agents.middleware import after_model, wrap_model_call
from langchain_core.messages import HumanMessage

@wrap_model_call
def one_tool_call_at_a_time(request, handler):
    # submit_room delivers "the last checked draft". If the model checks two drafts in
    # one turn, which one is "last" is an accident, so turn parallel tool calls off.
    settings = {**(request.model_settings or {}), "parallel_tool_calls": False}
    return handler(request.override(model_settings=settings))

@after_model(can_jump_to=["model"])
def keep_going_until_submitted(state, runtime):
    # The model may not end the run before a room is accepted: if it replies with
    # plain text and no tool call, send it back with a reminder.
    # Only act on a plain model reply. Other middleware (e.g. the tool-call limit) can add
    # tool messages after the model's reply; leave those turns alone.
    last = state["messages"][-1]
    if last.type != "ai":
        return None
    if not last.tool_calls and SESSION.get("final") is None:
        return {"messages": [HumanMessage(
                    "No room has been submitted yet, so you are not done. Continue: draft the "
                    "room, call check_room, and call submit_room when it has exactly one answer.")],
                "jump_to": "model"}
    return None

GUARDRAILS = [
    one_tool_call_at_a_time,
    keep_going_until_submitted,
    ModelCallLimitMiddleware(run_limit=24),
    ToolCallLimitMiddleware(run_limit=20),
    ToolRetryMiddleware(max_retries=2, initial_delay=0.5),
]

agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
    middleware=GUARDRAILS,
)


def build_trace(result):
    # Turn the agent's message history into a list of ReAct steps
    steps = []
    seen_question = False
    for message in result["messages"]:
        if message.type == "human":
            # the first human message is the request; later ones are harness reminders
            steps.append({"step": "harness" if seen_question else "question", "text": message.text})
            seen_question = True
        elif message.type == "ai" and message.tool_calls:
            if message.text:
                steps.append({"step": "thought", "text": message.text})
            for call in message.tool_calls:
                steps.append({"step": "action", "tool": call["name"], "args": call["args"]})
        elif message.type == "ai":
            steps.append({"step": "final", "text": message.text})
        elif message.type == "tool":
            steps.append({"step": "observation", "tool": message.name, "text": message.text})
    # a text-only reply that the harness sent back is a thought, not the final answer
    for s in steps[:-1]:
        if s["step"] == "final":
            s["step"] = "thought"
    return steps

def summarize_args(tool_name, args):
    if tool_name == "check_room":
        return (f"check_room(grid {args.get('grid_rows')}x{args.get('grid_cols')}, "
                f"{len(args.get('chests', []))} chests, mimics {args.get('mimics')}, "
                f"{len(args.get('statements', []))} statements)")
    return f"{tool_name}({args})"

def compact_json(text, limit=400):
    try:
        text = json.dumps(json.loads(text))
    except (ValueError, TypeError):
        pass
    return text if len(text) <= limit else text[:limit] + " ..."

def print_trace(steps):
    labels = {"question": "Question", "thought": "Thought", "action": "Action",
              "observation": "Observation", "final": "Final answer", "harness": "Harness"}
    for s in steps:
        if s["step"] == "action":
            detail = summarize_args(s["tool"], s["args"])
        elif s["step"] == "observation":
            detail = compact_json(s["text"])
        else:
            detail = s["text"]
        print(f"{labels[s['step']] + ':':<14}{detail}")


def generate_room(agent, level=2, seed=None, show_trace=True):
    # Run the agent once and return (room, steps). Falls back to the algorithmic generator.
    brief = start_session(level, seed)
    request = "Design the next room."
    error = None
    try:
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        steps = build_trace(result)
    except Exception as exc:
        steps = [{"step": "question", "text": request}]
        error = f"{type(exc).__name__}: {exc}"
    if show_trace:
        print_trace(steps)

    room = SESSION["final"]
    source = "agent"
    if room is None:
        source = "fallback"
        room = GeneratorAgent(seed=seed).generate(
            profile={"level": brief["level"]})
        if show_trace:
            print(f"\n[The agent did not deliver a room{': ' + error if error else ''}. "
                  "Using the algorithmic generator.]")

    room = dict(room, id=make_room_id(brief["level"]), rules_version=RULES_VERSION)
    trace = solve_with_trace(room)
    room["meta"] = {
        "source": source,
        "brief": brief,
        "design_note": room.pop("design_note", None),
        "skills": sorted({s["type"] for s in room["statements"]}),
        "difficulty": GeneratorAgent._difficulty(room, trace),
        "assumption_depth": trace["max_depth"],
        "tool_calls": sum(s["step"] == "action" for s in steps),
        "reminders": sum(s["step"] == "harness" for s in steps),
        "checks": SESSION["checks"],
        "error": error,
    }
    return room, steps
