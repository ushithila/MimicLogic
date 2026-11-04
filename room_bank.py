"""
Mimic Logic - a bank of ready-made rooms, refilled in the background.

The ReAct agent takes 10-30 seconds per room. The bank keeps one finished,
verified room per level (saved to a file, so they survive restarts). When the
player picks a level, they get that level's room instantly, and a background
worker builds its replacement while they play.

The agent keeps per-run state in a shared SESSION, so two rooms must never be
generated at once. The bank uses a single worker thread that builds one room at
a time; the level the player is waiting for always goes to the front of the line.

Saved rooms are stamped with the rules version (puzzle.RULES_VERSION). When a rule
changes what a statement means, rooms saved under the old rules are thrown away.

Usage:
    bank = RoomBank(generate)          # generate: level -> room
    bank.start(first=1)                # fills empty levels in the background
    room = bank.get(1)                 # instant if a level 1 room is ready
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque

from puzzle import RULES_VERSION, solve_all

LEVELS = (1, 2, 3, 4, 5)


def is_valid(room: dict) -> bool:
    """Re-check a room independently: made under the current rules, with exactly one
    answer, equal to the stored one."""
    if room.get("rules_version") != RULES_VERSION:
        return False
    try:
        sols = solve_all(room, limit=2)
        return len(sols) == 1 and sols[0] == room["solution"]
    except Exception:
        return False


class RoomBank:
    def __init__(self, generate, path: str = "room_bank.json", per_level: int = 1,
                 max_tries: int = 3):
        self.generate = generate
        self.path = path
        self.per_level = per_level
        self.max_tries = max_tries
        self.rooms = {lvl: [] for lvl in LEVELS}
        self.errors: dict[int, str] = {}
        self.queue: deque[int] = deque()       # levels waiting to be refilled
        self.building: int | None = None       # level the worker is building now
        self.cond = threading.Condition()
        self.worker: threading.Thread | None = None
        self._load()

    # ----------------------------------------------------------- persistence
    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                saved = json.load(f)
            for lvl in LEVELS:
                self.rooms[lvl] = [r for r in saved.get(str(lvl), []) if is_valid(r)]
        except (OSError, ValueError):
            pass                                # unreadable file: start empty

    def _save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({str(l): rs for l, rs in self.rooms.items()}, f)
            os.replace(tmp, self.path)
        except OSError:
            pass                                # saving is a convenience, not required

    # ----------------------------------------------------------- public API
    def start(self, first: int | None = None):
        """Start the worker and queue every level that needs rooms: `first` and its
        neighbours first (the player can only move one level at a time), then the rest."""
        near = [first, first + 1, first - 1] if first in LEVELS else []
        order = [l for l in near if l in LEVELS] + [l for l in LEVELS if l not in near]
        with self.cond:
            for lvl in order:
                self._request(lvl)
        if self.worker is None:
            self.worker = threading.Thread(target=self._work, daemon=True)
            self.worker.start()

    def ready(self, level: int) -> bool:
        with self.cond:
            return bool(self.rooms[level])

    def status(self) -> dict[int, int]:
        with self.cond:
            return {lvl: len(rs) for lvl, rs in self.rooms.items()}

    def get(self, level: int) -> dict:
        """Take a room for this level, waiting if none is ready, and queue a replacement."""
        with self.cond:
            if not self.rooms[level]:
                self.errors.pop(level, None)
                self._request(level, urgent=True)
                while not self.rooms[level] and level not in self.errors:
                    self.cond.wait()
                if not self.rooms[level]:
                    raise RuntimeError(self.errors.pop(level))
            room = self.rooms[level].pop(0)
            self._save()
            # The player's next room is at this level or one level up or down, so refill
            # those first, in that order (this level's replacement goes to the very front).
            for lvl in (level - 1, level + 1, level):
                if lvl in LEVELS and len(self.rooms[lvl]) < self.per_level:
                    self._request(lvl, urgent=True)
            return room

    # ----------------------------------------------------------- worker
    def _request(self, level: int, urgent: bool = False):
        """Queue a refill for this level (caller holds the lock)."""
        if level in self.queue:
            if urgent:                          # move to the front
                self.queue.remove(level)
                self.queue.appendleft(level)
        elif level != self.building:
            (self.queue.appendleft if urgent else self.queue.append)(level)
        self.cond.notify_all()

    def _work(self):
        while True:
            with self.cond:
                while not self.queue:
                    self.cond.wait()
                level = self.queue.popleft()
                if len(self.rooms[level]) >= self.per_level:
                    continue
                self.building = level
            room, error = None, None
            for _ in range(self.max_tries):     # generate outside the lock
                try:
                    candidate = self.generate(level)
                    if is_valid(candidate):
                        room = candidate
                        break
                    error = "generated room failed verification"
                except Exception as e:          # network, API, anything
                    error = f"{type(e).__name__}: {e}"
            with self.cond:
                self.building = None
                if room is not None:
                    self.rooms[level].append(room)
                    self._save()
                    if len(self.rooms[level]) < self.per_level:
                        self._request(level)
                else:
                    self.errors[level] = error or "generation failed"
                self.cond.notify_all()
