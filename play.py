"""
Mimic Logic - play in the terminal.

Each round:
  1. the generator makes a room,
  2. the solver re-checks it has exactly one answer,
  3. you name the mimics,
  4. your answer is checked (and explained if it's wrong),
  5. your level goes up after 2 right answers in a row, and down after 2 wrong ones.

Usage:
  python play.py                 # ReAct agent (needs OPENROUTER_API_KEY in .env)
  python play.py --fast          # instant algorithmic generator, no API key
  python play.py --offline       # agent harness with the scripted stand-in, no API key
  python play.py --level 3       # start at level 3 instead of 1 (for testing and demos)
  python play.py --trace         # show the agent's Thought/Action/Observation trace
  python play.py --no-bank       # don't use the room bank (generate each room on demand)
  python play.py --no-log        # don't write results to game_log.jsonl

With the agent, rooms come from a bank of ready-made rooms (room_bank.py), refilled in
the background while you play. Every room played is logged to game_log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import uuid

from puzzle import check_answer, render, rules_text, solve_all, solve_with_trace
from room_bank import RoomBank

MAX_GENERATION_TRIES = 3
MIN_LEVEL, MAX_LEVEL = 1, 5
STREAK_TO_MOVE = 2          # right (or wrong) answers in a row needed to change level


# --------------------------------------------------------------------------
# Generators
# --------------------------------------------------------------------------

def make_generator(mode: str, show_trace: bool):
    """Return (name, function level -> room)."""
    if mode == "fast":
        from generator_agent import GeneratorAgent
        gen = GeneratorAgent()
        return "algorithmic generator", lambda level: gen.generate(profile={"level": level})

    if mode == "offline":
        os.environ["MIMIC_OFFLINE"] = "1"
    try:
        import mimic_agent
    except RuntimeError as e:          # no API key
        print(f"\n{e}\nStarting with the algorithmic generator instead (or use --offline).\n")
        return make_generator("fast", show_trace)
    except ImportError as e:           # packages not installed
        print(f"\nCould not load the agent ({e}).\n"
              "Run `pip install -r requirements.txt`, or use --fast.\n")
        sys.exit(1)

    name = "offline stand-in" if mode == "offline" else f"ReAct agent ({mimic_agent.MODEL})"

    def generate(level):
        room, _ = mimic_agent.generate_room(mimic_agent.agent, level=level, show_trace=show_trace)
        return room

    return name, generate


def new_room(generate, level: int) -> dict:
    """Generate a room and verify it independently before showing it."""
    for attempt in range(1, MAX_GENERATION_TRIES + 1):
        room = generate(level)
        solutions = solve_all(room, limit=2)
        if len(solutions) == 1 and solutions[0] == room["solution"]:
            return room
        print(f"  Generated room failed verification (attempt {attempt}); trying again...")
    raise RuntimeError("Could not generate a verified room.")


def next_room(bank, generate, level: int, slow: bool) -> dict:
    """Get the next room: from the bank if one is ready, otherwise build it now."""
    if bank is None:
        print(f"\nGenerating a level {level} room"
              + ("... (the agent takes about 10-30 seconds)" if slow else "..."))
        return new_room(generate, level)
    if bank.ready(level):
        print(f"\nLevel {level} room ready.")
    else:
        print(f"\nBuilding a level {level} room... (the agent takes about 10-30 seconds)")
    try:
        return bank.get(level)
    except RuntimeError as e:
        print(f"  The agent couldn't build a room ({e}).\n"
              "  Using the algorithmic generator for this one.")
        from generator_agent import GeneratorAgent
        return GeneratorAgent().generate(profile={"level": level})


# --------------------------------------------------------------------------
# Progressive difficulty
# --------------------------------------------------------------------------

class Progression:
    """Up a level after STREAK_TO_MOVE right answers in a row, down after
    STREAK_TO_MOVE wrong answers in a row. Runs reset whenever the level changes."""

    def __init__(self, level: int = 1):
        self.level = level
        self.right_run = 0
        self.wrong_run = 0
        self.highest = level

    def record(self, correct: bool) -> str | None:
        """Update after an answer; return a message if the level changed."""
        if correct:
            self.right_run, self.wrong_run = self.right_run + 1, 0
            if self.right_run >= STREAK_TO_MOVE and self.level < MAX_LEVEL:
                self.level += 1
                self.right_run = 0
                self.highest = max(self.highest, self.level)
                return f"Level up! On to level {self.level}."
        else:
            self.right_run, self.wrong_run = 0, self.wrong_run + 1
            if self.wrong_run >= STREAK_TO_MOVE and self.level > MIN_LEVEL:
                self.level -= 1
                self.wrong_run = 0
                return f"Back to level {self.level} to sharpen up."
        return None

    def progress(self) -> str:
        if self.level < MAX_LEVEL and self.right_run:
            left = STREAK_TO_MOVE - self.right_run
            return f"{left} more right answer{'s' if left > 1 else ''} in a row to reach level {self.level + 1}"
        if self.level == MAX_LEVEL:
            return "top level"
        return f"{STREAK_TO_MOVE} right answers in a row to reach level {self.level + 1}"


# --------------------------------------------------------------------------
# Results log (for the Player Tracker and playtest data)
# --------------------------------------------------------------------------

class GameLog:
    """Appends one JSON line per room to a .jsonl file."""

    def __init__(self, path: str | None, generator: str):
        self.path = path
        self.session = uuid.uuid4().hex[:8]
        self.generator = generator

    def write(self, **entry):
        if not self.path:
            return
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": self.session,
                 "generator": self.generator, **entry}
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass                        # logging must never stop the game


# --------------------------------------------------------------------------
# Input helpers
# --------------------------------------------------------------------------

def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:                   # input stream closed
        raise KeyboardInterrupt


def ask_mimics(room: dict) -> set[str] | None:
    """Ask which chests are mimics. Returns a set of ids, or None to skip the room."""
    ids = [c["id"] for c in room["chests"]]
    while True:
        raw = ask(f"\nWhich chests are mimics? (letters, e.g. {' '.join(ids[:2])}) "
                  "[r = rules, s = skip, q = quit]: ").upper()
        if raw == "Q":
            raise KeyboardInterrupt
        if raw == "S":
            return None
        if raw == "R":
            print_rules()
            continue
        letters = [ch for ch in raw.replace(",", " ").split()]
        letters = [l for part in letters for l in (list(part) if len(part) > 1 else [part])]
        unknown = [l for l in letters if l not in ids]
        if not letters:
            print("  Type at least one chest letter. Every room has at least one mimic.")
        elif unknown:
            print(f"  No chest {', '.join(unknown)} in this room. Chests: {', '.join(ids)}.")
        else:
            return set(letters)


def yes(prompt: str, default: bool = True) -> bool:
    raw = ask(prompt).lower()
    return default if raw == "" else raw.startswith("y")


def print_rules():
    print()
    print("\n".join(rules_text()))


# --------------------------------------------------------------------------
# Game loop
# --------------------------------------------------------------------------

def play(mode: str, level: int = 1, show_trace: bool = False, use_bank: bool = True,
         log_path: str | None = "game_log.jsonl"):
    name, generate = make_generator(mode, show_trace)
    slow = not name.startswith("algorithmic")
    bank = None
    if slow and use_bank and not show_trace:
        bank = RoomBank(generate, path=f"room_bank_{mode}.json")
        bank.start(first=level)
    log = GameLog(log_path, name)
    prog = Progression(level)

    print("=" * 60)
    print("  MIMIC LOGIC")
    print("=" * 60)
    print(f"Rooms are made by the {name}.")
    print(f"You start at level {prog.level}. Get {STREAK_TO_MOVE} right in a row to level up; "
          f"{STREAK_TO_MOVE} wrong in a row and you drop a level.")
    print_rules()

    rooms = played = correct = streak = best_streak = 0
    try:
        while True:
            level = prog.level
            room = next_room(bank, generate, level, slow)
            rooms += 1
            if room.get("meta", {}).get("source") == "fallback":
                print("  (The agent didn't finish this room, so the algorithmic generator made it.)")
            print("Verified: this room has exactly one answer.\n")
            print(render(room, show_rules=False, title=f"ROOM {rooms}  ·  Level {level}"))

            shown = time.time()
            guess = ask_mimics(room)
            seconds = round(time.time() - shown, 1)
            mimics = sorted(c for c, v in room["solution"].items() if v == "mimic")
            entry = dict(room_number=rooms, room_id=room.get("id"), level=level,
                         source=room.get("meta", {}).get("source", "algorithmic"),
                         difficulty=room.get("meta", {}).get("difficulty"),
                         statement_types=sorted({s["type"] for s in room["statements"]}),
                         solution=mimics, seconds=seconds)

            if guess is None:
                print(f"  Skipped. The mimics were: {', '.join(mimics)}.")
                log.write(**entry, answer=None, correct=None, skipped=True, level_after=level)
            else:
                answer = {c: ("mimic" if c in guess else "normal") for c in room["solution"]}
                ok, message = check_answer(room, answer)
                played += 1
                if ok:
                    correct += 1
                    streak += 1
                    best_streak = max(best_streak, streak)
                    print(f"\n  Correct! The mimics were {', '.join(mimics)}.")
                else:
                    streak = 0
                    print("\n  Not quite. " + textwrap.fill(message, 76, subsequent_indent="  "))
                    print(f"  The mimics were: {', '.join(mimics)}.")
                    if yes("  See the solver's reasoning? [Y/n]: "):
                        for line in solve_with_trace(room)["trace"]:
                            print("    " + line)
                change = prog.record(ok)
                log.write(**entry, answer=sorted(guess), correct=ok, skipped=False,
                          level_after=prog.level)
                if change:
                    print(f"\n  *** {change} ***")

            print(f"\nScore: {correct}/{played} correct"
                  + (f"  |  streak {streak}" if streak > 1 else "")
                  + f"  |  Level {prog.level} ({prog.progress()})")

            raw = ask("\nPlay another? [Y/n]: ").lower()
            if raw.startswith("n") or raw == "q":
                break
    except KeyboardInterrupt:
        print()

    print("\n" + "=" * 60)
    if played:
        print(f"Final score: {correct}/{played} correct ({100 * correct // played}%)"
              + (f", best streak {best_streak}" if best_streak > 1 else "")
              + f". Highest level reached: {prog.highest}.")
    if log.path and rooms:
        print(f"Results saved to {log.path}.")
    print("Thanks for playing Mimic Logic!")


def main():
    p = argparse.ArgumentParser(description="Play Mimic Logic in the terminal.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--fast", action="store_true",
                       help="use the algorithmic generator (instant, no API key)")
    group.add_argument("--offline", action="store_true",
                       help="use the agent harness with the scripted stand-in (no API key)")
    p.add_argument("--level", type=int, default=1, choices=range(MIN_LEVEL, MAX_LEVEL + 1),
                   help="starting level (default 1; for testing and demos)")
    p.add_argument("--trace", action="store_true",
                   help="show the agent's ReAct trace (turns off the room bank)")
    p.add_argument("--no-bank", action="store_true",
                   help="generate every room on demand instead of using the room bank")
    p.add_argument("--log", default="game_log.jsonl", help="results log file")
    p.add_argument("--no-log", action="store_true", help="don't write a results log")
    a = p.parse_args()
    mode = "fast" if a.fast else "offline" if a.offline else "agent"
    play(mode, a.level, a.trace, use_bank=not a.no_bank,
         log_path=None if a.no_log else a.log)


if __name__ == "__main__":
    main()
