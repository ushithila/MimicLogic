# Mimic Logic: Puzzle Generator

Generates grid-based Mimic Logic rooms with the basic rules (Normal chests tell the truth, Mimics lie). Every room is checked by the solver and has exactly one answer.

There are two generators:

- **The ReAct agent** (`Mimic_Logic_ReAct_Generator.ipynb`): a language model (`openai/gpt-4.1-mini` through OpenRouter, built with LangChain's `create_agent`) designs each room, reads the solver's feedback, and revises. This is the main generator.
- **The algorithmic generator** (`generator_agent.py`): fast and predictable. The ReAct agent falls back to it if it fails to deliver a room.

## Files

| File | What it does |
| --- | --- |
| `Mimic_Logic_ReAct_Generator.ipynb` | The ReAct generator agent: tools, agent, trace, and evaluation, in the same structure as the course's LangChain ReAct tutorial |
| `puzzle.py` | Rules and instructions text, the statement types, the solver (brute force + step-by-step trace), the answer checker, and a text renderer. Shared by everything. |
| `generator_agent.py` | The algorithmic generator and a command-line tool |
| `offline_model.py` | A scripted stand-in for the model (not an LLM), to run the notebook without an API key or as a demo backup |
| `sample_rooms.json` | One sample room for each level, 1 to 5, from the algorithmic generator |

Keep all files in the same folder (in Colab, upload the `.py` files next to the notebook).

## ReAct agent quick start

1. Put `OPENROUTER_API_KEY` in Colab Secrets or a local `.env` file.
2. Run the notebook top to bottom. Section 5 generates rooms with a full Thought → Action → Observation trace; Section 6 evaluates the agent on 7 player profiles.
3. In the game, call `generate_room(agent, {"level": 3, "weak_type": "if_then"})`, which returns `(room, trace)`.

The agent's tools:

| Tool | What it does |
| --- | --- |
| `get_player_profile` | The player's level and weak statement type, plus a random design brief (grid, chests, mimics, allowed types, target difficulty, and a creative twist) |
| `check_room` | Validates a draft, flags statements that break the intended answer, counts answers with the solver (listing competing answers if ambiguous), and scores difficulty against the brief |
| `statement_options` | Suggests valid statements for a chest when the model is stuck |
| `submit_room` | Delivers the room, only if the solver confirmed exactly one answer |

To test without a key, set `USE_OFFLINE_MODEL = True` in Section 2. The stand-in runs the same loop with a fixed script; its rooms are valid but not designed by a model, and the LLM judge is skipped.

## Algorithmic generator

Python 3.10+, no extra packages.

## Quick start

```bash
python generator_agent.py                          # one room at level 2
python generator_agent.py --level 4 --answers      # show the answer and solver trace
python generator_agent.py --focus if_then          # lean on conditionals
python generator_agent.py --grid 3x3 --chests 6 --hide-count --silent 1
python generator_agent.py --count 10 --json rooms.json
```

## How the agent works

1. **Decide:** turns a player profile (`{"level": 1-5, "weak_type": ...}`) into settings: grid size, chest count, statement types, whether the mimic count is shown, and silent chests.
2. **Build:** picks a hidden answer first, then gives each chest a statement that is true if it's Normal and false if it's a Mimic. Statements about a neighbor use direction words ("the chest to my right").
3. **Verify:** the brute-force solver checks every assignment and confirms that exactly one fits.
4. **Revise:** if two answers fit, it rewrites the statement of a chest the answers disagree on and checks again, instead of starting over.
5. **Tag:** records the statement types used (`meta.skills`), solver metrics, and a 1–5 difficulty score.

From code:

```python
from generator_agent import GeneratorAgent
from puzzle import render, check_answer, solve_with_trace

agent = GeneratorAgent(seed=42)
room = agent.generate(profile={"level": 3, "weak_type": "if_then"})
print(render(room))
print(check_answer(room, {"A": "mimic", "B": "normal", ...}))  # (False, "You marked ...")
print("\n".join(solve_with_trace(room)["trace"]))
```

## Statement types

| type | Reads as |
| --- | --- |
| `is_mimic` | Chest X is a mimic. / The chest to my right is a mimic. |
| `not_mimic` | Chest X is not a mimic. |
| `either_one` | Exactly one of Chest X and Chest Y is a mimic. |
| `if_then` | If Chest X is a mimic, then Chest Y is a mimic. |
| `same_kind` | Chest X and Chest Y are the same kind. |
| `pattern` | All the mimics are in the same row / column, form an L, form a diagonal, or no two mimics are next to each other. |

The L pattern is 3 mimics filling 3 cells of a 2x2 block, in any rotation.

## Levels

| Level | Grid | Chests | Statement types | Mimic count | Silent chests |
| --- | --- | --- | --- | --- | --- |
| 1 | 2x2 | 3 | direct only | shown | 0 |
| 2 | 2x2 | 4 | + either-one, same kind | shown | 0 |
| 3 | 2x3 | 5 | + if-then | shown | 0 |
| 4 | 3x3 | 5 | + pattern | shown | 1 |
| 5 | 3x3 | 6 | all | hidden | 1 |

## Difficulty score

Mainly how many nested assumptions the solver needs ("assume E is a mimic... then assume B is..."), plus a little for compound statements, silent chests, a hidden mimic count, and rooms of 5+ chests. It's a starting estimate to compare against playtest results.

## Tested

900 generated rooms across all levels and focus settings: every room has exactly one answer, the step-by-step solver agrees with brute force, and the stored solution passes the answer checker. Each room takes a few milliseconds.
