# AGENTS.md — MTGA Advisor

## Project overview

An AI advisor for Magic: The Gathering Arena. You describe your current battlefield
state in plain text, and the agent reasons through the situation using a structured
loop before returning a final play recommendation.

This file documents **Step 1**: build the initial LangGraph agent loop with a single
input/output flow — no UI, no live game integration yet.

---

## Step 1 goal

Wire up a LangGraph agent loop that:

1. Accepts a free-text description of the battlefield (hand, board, opponent board, mana available, phase)
2. Runs through the full observe → think → act → check cycle, logging each step
3. Returns the complete reasoning trace + a final play recommendation

Nothing more. No RAG integration yet. No real card database. Just the loop working end to end.

---

## Project structure

```
mtga-advisor/
├── AGENTS.md               ← this file
├── requirements.txt
├── .env                    ← OPENAI_API_KEY or ANTHROPIC_API_KEY
├── agent/
│   ├── __init__.py
│   ├── graph.py            ← LangGraph graph definition
│   ├── nodes.py            ← one function per loop node
│   ├── state.py            ← AgentState TypedDict
│   └── prompts.py          ← system + node-level prompt templates
├── tools/
│   └── __init__.py         ← placeholder, tools go here in later steps
└── main.py                 ← entrypoint: read battlefield, run graph, print trace
```

---

## Dependencies

```txt
# requirements.txt
langgraph>=0.2
langchain-core>=0.2
langchain-openai>=0.1     # swap for langchain-anthropic if using Claude
python-dotenv>=1.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## State definition — `agent/state.py`

The state is the single object passed between every node in the graph.
Keep it flat and explicit — no nested dicts.

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Input
    battlefield: str            # raw battlefield description from the user

    # Loop internals
    messages: Annotated[list, add_messages]  # full conversation with the LLM
    observations: list[str]     # what the agent noticed each iteration
    thoughts: list[str]         # reasoning produced each iteration
    actions_taken: list[str]    # what the agent decided to do each iteration
    iteration: int              # loop counter (safety limit)

    # Output
    final_recommendation: str   # populated when the loop exits
    done: bool                  # flag that tells LangGraph to exit the loop
```

---

## Node definitions — `agent/nodes.py`

Each node is a plain Python function that receives `AgentState` and returns
a dict of fields to update. LangGraph merges the return value back into state.

### Node: `observe`

Reads the battlefield and extracts structured facts the LLM can reason over.

```python
from agent.state import AgentState
from agent.prompts import OBSERVE_PROMPT
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def observe(state: AgentState) -> dict:
    """Extract structured observations from the battlefield description."""
    prompt = OBSERVE_PROMPT.format(battlefield=state["battlefield"])
    response = llm.invoke([{"role": "user", "content": prompt}])
    observation = response.content
    print(f"\n[OBSERVE — iteration {state['iteration']}]\n{observation}")
    return {
        "observations": state["observations"] + [observation],
        "messages": [{"role": "assistant", "content": observation}],
    }
```

### Node: `think`

Reasons over the observations and decides on a course of action.

```python
def think(state: AgentState) -> dict:
    """Reason over the observations and produce a plan."""
    latest_obs = state["observations"][-1]
    prompt = THINK_PROMPT.format(
        battlefield=state["battlefield"],
        observation=latest_obs,
        iteration=state["iteration"],
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    thought = response.content
    print(f"\n[THINK — iteration {state['iteration']}]\n{thought}")
    return {
        "thoughts": state["thoughts"] + [thought],
        "messages": [{"role": "assistant", "content": thought}],
    }
```

### Node: `act`

Converts the thought into a concrete action. In Step 1 this is text only.
In Step 2+ this node will call tools (RAG lookup, Scryfall API, etc.).

```python
def act(state: AgentState) -> dict:
    """Decide on a concrete action based on current reasoning."""
    latest_thought = state["thoughts"][-1]
    prompt = ACT_PROMPT.format(thought=latest_thought, iteration=state["iteration"])
    response = llm.invoke([{"role": "user", "content": prompt}])
    action = response.content
    print(f"\n[ACT — iteration {state['iteration']}]\n{action}")
    return {
        "actions_taken": state["actions_taken"] + [action],
        "messages": [{"role": "assistant", "content": action}],
    }
```

### Node: `check`

Decides whether the agent has enough information to give a final answer,
or whether it needs another loop iteration.

```python
def check(state: AgentState) -> dict:
    """Decide: are we done, or do we need another pass?"""
    latest_action = state["actions_taken"][-1]
    MAX_ITERATIONS = 3

    if state["iteration"] >= MAX_ITERATIONS:
        # Force exit — safety limit
        done = True
        recommendation = latest_action
        print(f"\n[CHECK] Max iterations reached. Forcing exit.")
    else:
        prompt = CHECK_PROMPT.format(
            action=latest_action,
            iteration=state["iteration"],
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content.strip().lower()
        done = content.startswith("done")
        recommendation = latest_action if done else ""
        print(f"\n[CHECK — iteration {state['iteration']}] done={done}")

    return {
        "done": done,
        "final_recommendation": recommendation if done else state.get("final_recommendation", ""),
        "iteration": state["iteration"] + 1,
    }
```

---

## Prompt templates — `agent/prompts.py`

Keep prompts in one place so they are easy to tune without touching logic.

```python
OBSERVE_PROMPT = """
You are a Magic: The Gathering expert advisor.

Battlefield state:
{battlefield}

Your task: extract the key facts from this battlefield state.
List them clearly:
- Your hand (cards available to cast)
- Your battlefield (permanents you control, their power/toughness, any relevant abilities)
- Opponent's battlefield (same)
- Mana available
- Current phase and whose turn it is
- Life totals (yours and opponent's)
- Any relevant stack or graveyard state

Be precise. Do not give advice yet — only extract facts.
"""

THINK_PROMPT = """
You are a Magic: The Gathering expert advisor. You are on iteration {iteration} of your analysis.

Battlefield state:
{battlefield}

Observations so far:
{observation}

Your task: reason deeply about the strategic situation.
Think through:
- What are your win conditions right now?
- What does your opponent likely have or want to do?
- What plays are available to you this turn?
- What are the risks and rewards of each option?
- Are there any interactions, tricks, or combat math to consider?

Do not give a final recommendation yet — just think out loud.
"""

ACT_PROMPT = """
You are a Magic: The Gathering expert advisor. You are on iteration {iteration}.

Based on your reasoning:
{thought}

Propose a concrete sequence of actions for this turn.
Format it as a numbered step-by-step play:
1. [action]
2. [action]
...

Include brief reasoning for each step (one sentence max per step).
End with: "Confidence: HIGH / MEDIUM / LOW" and a one-line summary.
"""

CHECK_PROMPT = """
You are evaluating whether a Magic: The Gathering play recommendation is complete and confident enough to deliver.

Proposed play (iteration {iteration}):
{action}

Reply with exactly one of:
- "DONE" — if the recommendation is clear, complete, and actionable
- "CONTINUE" — if the reasoning needs another pass (e.g. confidence is LOW, or key interactions were not addressed)

Reply with only DONE or CONTINUE and nothing else.
"""
```

---

## Graph definition — `agent/graph.py`

Wire the nodes together with LangGraph's `StateGraph`.
The conditional edge after `check` is what creates the loop.

```python
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import observe, think, act, check

def build_graph():
    g = StateGraph(AgentState)

    # Register nodes
    g.add_node("observe", observe)
    g.add_node("think",   think)
    g.add_node("act",     act)
    g.add_node("check",   check)

    # Linear flow within one iteration
    g.add_edge("observe", "think")
    g.add_edge("think",   "act")
    g.add_edge("act",     "check")

    # Conditional: loop back or exit
    g.add_conditional_edges(
        "check",
        lambda state: END if state["done"] else "observe",
    )

    g.set_entry_point("observe")
    return g.compile()
```

---

## Entrypoint — `main.py`

```python
import os
from dotenv import load_dotenv
from agent.graph import build_graph
from agent.state import AgentState

load_dotenv()

EXAMPLE_BATTLEFIELD = """
Turn 5, my main phase. My turn.

My hand: Lightning Bolt, Counterspell, Island (land)
My battlefield: 2x Island (tapped), 1x Mountain (untapped), Goblin Guide (2/2 haste), Snapcaster Mage (2/1)
Opponent's battlefield: Tarmogoyf (4/5), 1x Forest (untapped)
My life total: 12
Opponent's life total: 6
Graveyard (mine): 2 instants, 1 sorcery
"""

def main():
    graph = build_graph()

    initial_state: AgentState = {
        "battlefield": EXAMPLE_BATTLEFIELD,
        "messages": [],
        "observations": [],
        "thoughts": [],
        "actions_taken": [],
        "iteration": 1,
        "final_recommendation": "",
        "done": False,
    }

    print("=" * 60)
    print("MTGA ADVISOR — AGENT LOOP")
    print("=" * 60)

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    print(final_state["final_recommendation"])
    print(f"\nCompleted in {final_state['iteration'] - 1} iteration(s).")

if __name__ == "__main__":
    main()
```

---

## Running it

```bash
# 1. Clone / create your project folder and install deps
pip install -r requirements.txt

# 2. Set your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 3. Run
python main.py
```

Expected output shape:

```
============================================================
MTGA ADVISOR — AGENT LOOP
============================================================

[OBSERVE — iteration 1]
- Hand: Lightning Bolt, Counterspell, Island
- Your board: Goblin Guide, Snapcaster Mage, 3 lands (2 tapped)
...

[THINK — iteration 1]
Opponent is at 6 life. I have Lightning Bolt for 3 damage.
Snapcaster can flashback a spell from graveyard...

[ACT — iteration 1]
1. Cast Lightning Bolt targeting opponent (6 → 3 life). Confidence: HIGH

[CHECK — iteration 1] done=True

============================================================
FINAL RECOMMENDATION
============================================================
1. Cast Lightning Bolt targeting opponent (6 → 3 life).
2. Attack with Goblin Guide (2 damage) for lethal.
Confidence: HIGH — opponent goes to 0 life this turn.

Completed in 1 iteration(s).
```

---

## What to validate before moving to Step 2

- [ ] Graph runs end to end with the example battlefield
- [ ] Loop iterates at least once when `check` returns `CONTINUE`
- [ ] Safety limit (3 iterations) fires correctly when set to 1
- [ ] Each node prints its output so the full trace is visible
- [ ] Swapping the LLM (OpenAI ↔ Anthropic) requires only changing the import in `nodes.py`

---

## Next steps (Step 2 preview)

Once the loop is working:

- Add a `tools/rag.py` module that queries your existing card/rules vector store
- Give the `act` node access to the RAG tool so it can look up card text before committing to a line
- Add a `tools/scryfall.py` wrapper for live card lookup by name
- Introduce LangGraph's `ToolNode` to handle tool calls cleanly inside the loop