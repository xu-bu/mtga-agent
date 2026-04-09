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
