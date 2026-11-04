# Mimic Logic: Puzzle Generator

Generates grid-based Mimic Logic rooms with the basic rules (Normal chests tell the truth, Mimics lie). Every room is checked by the solver and has exactly one answer.

There are two generators:

- **The ReAct agent** (`Mimic_Logic_ReAct_Generator.ipynb`): a language model (`openai/gpt-4.1-mini` through OpenRouter, built with LangChain's `create_agent`) designs each room, reads the solver's feedback, and revises. This is the main generator.
- **The algorithmic generator** (`generator_agent.py`): fast and predictable. The ReAct agent falls back to it if it fails to deliver a room.

## Files

| File | What it does |
| --- | --- |
| `play.py` | **Play the game in the terminal**: generates a room, verifies it, asks for your answer, checks it, and keeps score |
| `game_log.jsonl` | Created by `play.py`: one line per room played (answer, right or wrong, time taken, level) |
| `room_bank.py` | A bank of ready-made rooms, one per level, refilled in the background so the game rarely waits for the agent |
| `mimic_agent.py` | The notebook's agent (Sections 1–4) as an importable module, used by `play.py` and the game |
| `Mimic_Logic_ReAct_Generator.ipynb` | The ReAct generator agent: tools, agent, trace, and evaluation, in the same structure as the course's LangChain ReAct tutorial |
| `puzzle.py` | Rules and instructions text, the statement types, the solver (brute force + step-by-step trace), the answer checker, and a text renderer. Shared by everything. |
| `generator_agent.py` | The algorithmic generator and a command-line tool |
| `offline_model.py` | Optional. A scripted stand-in for the model (not an LLM), to run the notebook without an API key or as a demo backup |
| `sample_rooms.json` | One sample room for each level, 1 to 5, from the algorithmic generator |
| `requirements.txt` | The Python packages the notebook needs |

Keep all files in the same folder.

## Getting started

### What you need

- **Python 3.10 or newer** ([python.org](https://www.python.org/downloads/)). On Windows, tick **"Add Python to PATH"** during the install.
- **VS Code** with the **Python** and **Jupyter** extensions (both by Microsoft), or a Google account for Colab.
- An **OpenRouter API key** from [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys). Each room costs a few model calls; with `gpt-4.1-mini` that is well under a cent per room, and the 7-room evaluation costs a few cents.

### Option A: run locally in VS Code

**1. Get the code.**

```bash
git clone https://github.com/ushithila/MimicLogic.git
cd MimicLogic
```

Then open the folder in VS Code (**File → Open Folder**).

**2. Create a virtual environment and install the packages.** Open a terminal (**Terminal → New Terminal**) and run these **one line at a time**:

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Mac / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The prompt should now start with `(.venv)`. On Windows, if activating says scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

**3. Add your API key.** Create a file named `.env` in the project folder (not `.env.txt`) containing one line, with no quotes or spaces:

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

`.env` is listed in `.gitignore`, so it is never uploaded. **Never commit your key.** If it is ever pushed to GitHub, delete it on OpenRouter and create a new one.

**4. Pick the kernel.** Open `Mimic_Logic_ReAct_Generator.ipynb`, click **Select Kernel** (top right) → **Python Environments** → the one marked `.venv`. If VS Code offers to install the Python and Jupyter extensions instead, accept, reload, and try again.

**5. Run it.** Click **Run All**. The first `!pip install` cell is harmless if everything is already installed. You should see `OPENROUTER_API_KEY is available.` near the top.

### Option B: run in Google Colab

1. Upload `Mimic_Logic_ReAct_Generator.ipynb` to Colab.
2. Open the **Files** panel (📁) and upload `puzzle.py`, `generator_agent.py`, and `offline_model.py` next to it.
3. Open **Secrets** (🔑), add `OPENROUTER_API_KEY` with your key, and toggle notebook access on.
4. In Section 2, uncomment the Colab cell that loads the key.
5. **Runtime → Run all.**

Colab forgets uploaded files when the session ends, so re-upload the `.py` files each time.

### Without an API key

In Section 2, set `USE_OFFLINE_MODEL = True`. The notebook then runs end to end with a scripted stand-in instead of the model: useful for checking your setup, or as a backup if the internet fails during a demo. Its rooms are valid but not designed by a model, and the LLM judge is skipped.

## Playing the game

After the setup above, run from the project folder (with `.venv` active):

```bash
python play.py                 # rooms designed by the ReAct agent (needs the API key)
python play.py --fast          # instant rooms from the algorithmic generator, no key
python play.py --level 4       # start at level 4 instead of 1 (for testing and demos)
python play.py --trace         # also show the agent's Thought/Action/Observation trace
python play.py --offline       # the agent harness with the scripted stand-in, no key
python play.py --no-bank       # generate every room on demand (see "Room bank" below)
python play.py --no-log        # don't write results to game_log.jsonl
```

Each round, the generator makes a room and the solver checks it has exactly one answer before you see it. The room shows its number and level ("ROOM 3 · Level 2") and how many mimics it holds ("Mimics: 2"). At level 5 the count is hidden ("The mimics have aggressively taken over the hint system to hide their numbers.").

Type the letters of the chests you think are mimics (`B D`). A wrong answer is explained ("You marked B as a Mimic, but its statement ... would be true"), and you can see the solver's step-by-step reasoning. During a room, `r` shows the rules, `s` skips, and `q` quits. Without an API key, `play.py` starts with the algorithmic generator automatically.

### Progressive difficulty

You start at level 1. **Two right answers in a row move you up a level; two wrong answers in a row move you down** (levels 1 to 5). Anything else keeps you where you are, and a skipped room doesn't count. The score line shows how close you are to the next level. This is a basic version of the adaptive difficulty planned for the Player Tracker.

### Results log

Every room played is added to `game_log.jsonl`, one JSON line per room: the room id, level, the player's answer, whether it was right, how long it took (seconds), whether it was skipped, and the level afterwards. It is the raw data for playtesting and, later, for the Player Tracker. Add `game_log.jsonl` to `.gitignore` unless you mean to share it.

### Room bank (why most rooms appear without waiting)

The agent takes about 10–30 seconds to design a room, so the game keeps a **bank** of one finished, verified room per level (`room_bank.py`):

- When a room is ready for your level, you get it straight away, and a background worker starts building replacements while you solve: your level first, then the levels just above and below it (the only places you can go next).
- Whether the next room is ready depends on how fast you solve: a room you finish in a few seconds may beat its replacement.
- The bank is saved to `room_bank_agent.json`, so rooms are ready the next time you start the game. On the very first run, the bank fills up in the background (about 5 rooms, a few cents of API use).
- Rooms are built **one at a time**: the agent keeps its working state in one shared place (`SESSION`), so two rooms built at once would overwrite each other.
- Every room is re-verified when it is loaded from the file, and rooms saved under older rules are discarded (each room records `rules_version`). If the agent can't build a room (e.g. no internet), the game uses the algorithmic generator for that room.

`--trace` turns the bank off, so background traces don't print over the game. To start with a fresh bank, delete `room_bank_agent.json`. Add `room_bank_*.json` to `.gitignore`: each player's bank is their own.

The game screen can use the same bank:

```python
from room_bank import RoomBank
from mimic_agent import agent, generate_room

bank = RoomBank(lambda level: generate_room(agent, level=level, show_trace=False)[0])
bank.start(first=1)          # fill the levels in the background, level 1 and its neighbours first
room = bank.get(1)           # straight away when a level 1 room is ready
```

## Running the notebook

| Section | What it does | Time |
| --- | --- | --- |
| 1–2 | Imports, API key, model choice | seconds |
| 3 | The tools, tested directly (a format error, an ambiguous room, a rejected and an accepted submit) | seconds |
| 4 | Builds the agent: `@tool` wrappers, model, system prompt, guardrails | seconds |
| 5 | Generates a level 2 and a level 4 room, with the full **Thought → Action → Observation** trace and the solver's reasoning | about 30 s per room |
| 6 | Evaluates 7 seeded rooms across levels 1–5, runs the LLM judge, and prints a failing trace for error analysis | about 3–4 minutes |
| 7 | Ideas for improving the agent | — |

Results vary between runs (the model's temperature is 0.7), so run Section 6 two or three times before drawing conclusions.

To print the full trace of one evaluation case:

```python
print_trace(next(r for r in baseline_records if r["id"] == "level5_a")["steps"])
```

## Using the generator in the game

Import the agent from `mimic_agent.py` (the same code as notebook Sections 1–4):

```python
from mimic_agent import agent, generate_room
from puzzle import render, check_answer, solve_with_trace

room, trace = generate_room(agent, level=3, show_trace=False)   # levels 1 to 5

print(render(room))                              # the room as text
room["solution"]                                 # {"A": "normal", "B": "mimic", ...}
check_answer(room, player_answer)                # (False, "You marked A as a Mimic, but ...")
solve_with_trace(room)["trace"]                  # the solver's step-by-step reasoning
```

`play.py` is a complete example of this loop. `generate_room` always returns a valid room. If the agent fails to deliver one, it falls back to the algorithmic generator and records `"source": "fallback"` in `room["meta"]`.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Changed the agent in the notebook, but `play.py` behaves the old way | `mimic_agent.py` is a copy of notebook Sections 1–4. Make the same change there. |
| `play.py` crashes with "No module named 'dotenv'" when you click Run | VS Code is using the wrong Python. Run `.venv\Scripts\activate`, then `python play.py`, or pick the `.venv` interpreter (**Python: Select Interpreter**). |
| `ModuleNotFoundError: No module named 'puzzle'` | The `.py` files aren't in the same folder as the notebook, or VS Code opened a different folder. |
| `ModuleNotFoundError: No module named 'langchain'` | The notebook is using the wrong kernel. Select the `.venv` one. |
| `Missing OPENROUTER_API_KEY` | `.env` is missing, misnamed (`.env.txt`), or not in the notebook's folder. Check the variable name exactly. |
| Pasting several commands shows `>>` and nothing runs | PowerShell is waiting for more input. Press `Ctrl+C`, then run the commands one at a time. |
| Install fails with "access denied" in a OneDrive folder | OneDrive is syncing `.venv`. Pause OneDrive while installing, or keep the project outside OneDrive. |
| Every room shows `(fallback)` | Read a trace (Section 6, Step 4). It shows where the agent got stuck. |
| `git push` is rejected with "fetch first" | Someone pushed first. Run `git pull origin main --no-edit`, then push again. Never use `--force` on a shared repo. |

## How the ReAct agent works

The model designs each room; the solver keeps it honest.

| Tool | What it does |
| --- | --- |
| `get_design_brief` | A random design brief for the chosen level: grid, chests, mimics, allowed statement types, silent chests, target difficulty, and a creative twist |
| `check_room` | Validates a draft and reports: format errors; statements that break the intended answer, with **suggested fixes**; duplicate and giveaway statements (`design_problems`); mismatches with the brief; how many answers the room has (with competing answers if ambiguous, and **tested changes** that make it uniquely solvable); and difficulty against the brief |
| `statement_options` | Suggests valid statements for a chest when the model is stuck |
| `submit_room` | Delivers the room, only if the solver confirmed exactly one answer |

Guardrails (LangChain middleware):

| Guardrail | Why |
| --- | --- |
| `one_tool_call_at_a_time` | Stops the model checking two drafts at once, which made "the last checked draft" ambiguous |
| `keep_going_until_submitted` | Sends the model back if it stops with plain text before a room is accepted |
| `ModelCallLimitMiddleware` (24), `ToolCallLimitMiddleware` (20) | Budgets, so a confused agent can't loop forever |
| `ToolRetryMiddleware` | Retries a tool that raises an exception |

Each one came from a failure found in a trace during evaluation; the notebook explains each next to its code.

### Agent levels

| Level | Grid | Chests | Statement types | Mimic count | Silent chests | Target difficulty |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2x2 | 3 | direct only | shown | 0 | 1–2 |
| 2 | 2x2 or 2x3 | 4 | + either-one, same kind | shown | 0 | 2–3 |
| 3 | 2x3 or 3x2 | 4–5 | + if-then | shown | 0–1 | 2–4 |
| 4 | 3x3 or 2x3 | 5–6 | + pattern | shown | 0–1 | 3–5 |
| 5 | 3x3 | 5–7 | all | hidden | 1 | 4–5 |

## Algorithmic generator

Python 3.10+, no extra packages.

```bash
python generator_agent.py                          # one room at level 2
python generator_agent.py --level 4 --answers      # show the answer and solver trace
python generator_agent.py --grid 3x3 --chests 6 --hide-count --silent 1
python generator_agent.py --count 10 --json rooms.json
```

How it works:

1. **Decide:** turns a level (1–5) into settings: grid size, chest count, statement types, whether the mimic count is shown, and silent chests.
2. **Build:** picks a hidden answer first, then gives each chest a statement that is true if it's Normal and false if it's a Mimic. Statements about a neighbor use direction words ("the chest to my right").
3. **Verify:** the brute-force solver checks every assignment and confirms that exactly one fits.
4. **Revise:** if two answers fit, it rewrites the statement of a chest the answers disagree on and checks again, instead of starting over.
5. **Tag:** records the statement types used (`meta.skills`), solver metrics, and a 1–5 difficulty score.

From code:

```python
from generator_agent import GeneratorAgent
from puzzle import render, check_answer, solve_with_trace

gen = GeneratorAgent(seed=42)
room = gen.generate(profile={"level": 3})
print(render(room))
print("\n".join(solve_with_trace(room)["trace"]))
```

Its levels are slightly different from the agent's:

| Level | Grid | Chests | Statement types | Mimic count | Silent chests |
| --- | --- | --- | --- | --- | --- |
| 1 | 2x2 | 3 | direct only | shown | 0 |
| 2 | 2x2 | 4 | + either-one, same kind | shown | 0 |
| 3 | 2x3 | 5 | + if-then | shown | 0 |
| 4 | 3x3 | 5 | + pattern | shown | 1 |
| 5 | 3x3 | 6 | all | hidden | 1 |

## Statement types

| type | Reads as |
| --- | --- |
| `is_mimic` | Chest X is a mimic. / The chest to my right is a mimic. |
| `not_mimic` | Chest X is not a mimic. |
| `either_one` | Exactly one of Chest X and Chest Y is a mimic. |
| `if_then` | If Chest X is a mimic, then Chest Y is a mimic. |
| `same_kind` | Chest X and Chest Y are the same kind. |
| `pattern` | All the mimics are in the same row / column, form an L shape, fill a diagonal line, or no two mimics are next to each other. |

What the patterns mean exactly (these are also shown to players with the rules):

- **Same row / same column:** there are at least 2 mimics, all in one row (or column). Gaps between them are fine.
- **Diagonal:** the mimics fill one complete diagonal, corner to corner or edge to edge (at least 2 chests long), and there are no other mimics. On a 3x3 grid that allows 6 diagonals: the two corner-to-corner lines (3 chests) and four short ones (2 chests). A diagonal with an empty cell can't be filled, so it never counts.
- **L shape:** exactly 3 mimics, filling 3 of the 4 cells of a 2x2 square.
- **No two mimics next to each other:** no mimic is directly left, right, above or below another mimic.

### What every room avoids

- **Duplicate statements:** no two chests say the same thing ("A and B are the same kind" counts as the same as "B and A are the same kind").
- **Giveaways:** statements the mimic count settles on its own, such as "The mimics fill a diagonal line" when the room has 1 mimic (always false, so the speaker is obviously a Mimic).

Both generators avoid these, and the agent's `check_room` reports them as `design_problems`. The evaluation's **design** column checks them.

### Room format

Each room is a JSON object with `id` (unique, e.g. `L2-20261104-130712-a3f9c1`: level, time built, random code), `grid`, `chests`, `statements`, `hints` (`mimic_count`, `show_mimic_count`), `solution`, `rules_version`, and `meta` (generator details). `rules_version` (`puzzle.RULES_VERSION`) goes up whenever a rule changes what a statement means, so old saved rooms can be recognised and discarded.

## Difficulty score

Mainly how many nested assumptions the solver needs ("assume E is a mimic... then assume B is..."), plus a little for compound statements, silent chests, a hidden mimic count, and rooms of 5+ chests. It's a starting estimate to compare against playtest results.
