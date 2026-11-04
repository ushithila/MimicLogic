"""
Offline stand-in for the language model (NOT an LLM).

It plugs into LangChain's create_agent exactly where ChatOpenAI goes and
plays the agent's side of the ReAct loop with a fixed script: read the
design brief, draft a room with one clue held back, read the checker's report,
ask for options if the room is ambiguous, revise, and submit.

Use it to test the harness without an API key, or as a demo backup.
Its rooms come from the algorithmic generator, so they are valid but not
designed by a model.
"""

from __future__ import annotations

import json
import random
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from generator_agent import GeneratorAgent


class OfflineStandInModel(BaseChatModel):
    seed: int | None = None

    @property
    def _llm_type(self) -> str:
        return "offline-stand-in"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self

    # ------------------------------------------------------------------
    def _reply(self, thought: str, tool: str | None = None, args: dict | None = None):
        calls = ([{"name": tool, "args": args or {}, "id": f"call_{uuid.uuid4().hex[:8]}",
                   "type": "tool_call"}] if tool else [])
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=thought, tool_calls=calls))])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_msgs:
            return self._reply("I'll read the design brief first.", "get_design_brief")

        last = tool_msgs[-1]
        obs = json.loads(last.content) if last.content.startswith("{") else {}
        brief = next(json.loads(m.content) for m in tool_msgs if m.name == "get_design_brief")
        state = self._state(messages, brief)

        if last.name == "get_design_brief":
            draft = dict(state["room"])
            draft["statements"] = [s for s in draft["statements"]
                                   if s["speaker"] != state["held_back"]["speaker"]]
            return self._reply(
                f"The brief asks for {brief['num_chests']} chests on a {brief['grid']} grid. "
                f"I'll hide the mimics at {', '.join(draft['mimics'])} and check a first draft.",
                "check_room", draft)

        if last.name == "check_room":
            if obs.get("status") == "ambiguous" and not state["added_back"]:
                sp = state["held_back"]["speaker"]
                return self._reply(f"The room is ambiguous. I'll see what Chest {sp} could say "
                                   f"to tell the answers apart.",
                                   "statement_options", {"speaker": sp})
            full_checked = sum(isinstance(m, ToolMessage) and m.name == "check_room"
                               for m in messages) >= 2
            if obs.get("status") == "unique" and not full_checked:
                return self._reply("Already unique, but the brief wants more chests to speak. "
                                   "I'll add the last clue back and check again.",
                                   "check_room", state["room"])
            if obs.get("status") == "unique":
                return self._reply("Exactly one answer. Submitting the room.", "submit_room",
                                   {"design_note": "Rule out the competing answer using the "
                                                   "last clue added."})
            return self._reply("I'll check the full draft.", "check_room", state["room"])

        if last.name == "statement_options":
            return self._reply("I'll give that chest a statement and check again.",
                               "check_room", state["room"])

        if last.name == "submit_room":
            return self._reply("The room has been delivered to the player.")

        return self._reply("I can't continue.")

    def _state(self, messages, brief):
        """Rebuild the scripted plan from the conversation (deterministic per run)."""
        seed = self.seed if self.seed is not None else hash(json.dumps(brief, sort_keys=True))
        rng = random.Random(seed)
        rows, cols = map(int, brief["grid"].split("x"))
        gen = GeneratorAgent(seed=rng.randrange(10 ** 6))
        s = gen.decide_settings({"level": brief["level"]})
        s.rows, s.cols, s.num_chests = rows, cols, brief["num_chests"]
        s.min_mimics = s.max_mimics = brief["num_mimics"]
        s.allowed_types = brief["allowed_types"]
        s.show_mimic_count = brief["show_mimic_count"]
        s.silent_chests = brief["silent_chests"]
        s.minimize = False
        r = gen.generate(s)
        room = {"grid_rows": rows, "grid_cols": cols, "chests": r["chests"],
                "mimics": [c for c, v in r["solution"].items() if v == "mimic"],
                "statements": [{k: st[k] for k in ("speaker", "type", "targets")}
                               for st in r["statements"]],
                "show_mimic_count": r["hints"]["show_mimic_count"]}
        held_back = room["statements"][rng.randrange(len(room["statements"]))]
        added_back = any(isinstance(m, ToolMessage) and m.name == "statement_options"
                         for m in messages)
        return {"room": room, "held_back": held_back, "added_back": added_back}
