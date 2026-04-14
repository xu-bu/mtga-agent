from agent.state import AgentState


def observe_prompt(state: AgentState) -> str:
    return """
You are a Magic: The Gathering expert advisor.

Structured state details:
- Your hand: {your_hand}
- Opponent hand: {opponent_hand}
- Your battlefield: {your_battlefield}
- Opponent battlefield: {opponent_battlefield}
- Your graveyard: {your_graveyard}
- Opponent graveyard: {opponent_graveyard}
- Your exile: {your_exile}
- Opponent exile: {opponent_exile}
- Your mana pool: {your_mana_pool}
- Opponent mana pool: {opponent_mana_pool}
- Phase: {phase}
- Turn player: {turn_player}
- Stack: {stack}
- Life total: {life_total}
- Opponent life total: {opponent_life_total}

Retrieved Card Data & Rules:
{card_context}

Your task: extract the key facts from this battlefield state.
List them clearly with separate sections for each zone:
- Your hand
- Opponent's hand
- Your battlefield (permanents you control, their power/toughness, any relevant abilities)
- Opponent's battlefield (same)
- Your graveyard
- Opponent's graveyard
- Your exile
- Opponent's exile
- Mana available
- Current phase and whose turn it is
- Life totals (yours and opponent's)
- Any relevant stack state

If a zone is empty or not mentioned, say "none" for that section.
Be precise. Do not give advice yet — only extract facts.
""".format(
        your_hand=state["your_hand"],
        opponent_hand=state["opponent_hand"],
        your_battlefield=state["your_battlefield"],
        opponent_battlefield=state["opponent_battlefield"],
        your_graveyard=state["your_graveyard"],
        opponent_graveyard=state["opponent_graveyard"],
        your_exile=state["your_exile"],
        opponent_exile=state["opponent_exile"],
        your_mana_pool=state["your_mana_pool"],
        opponent_mana_pool=state["opponent_mana_pool"],
        phase=state["phase"],
        turn_player=state["turn_player"],
        stack=state["stack"],
        life_total=state["life_total"],
        opponent_life_total=state["opponent_life_total"],
        card_context=state.get("card_context", "No card data retrieved."),
    )


def think_prompt(state: AgentState, observation: str) -> str:
    return """
You are a Magic: The Gathering expert advisor. You are on iteration {iteration} of your analysis.

Observations so far:
{observation}

Retrieved Card Data & Rules:
{card_context}

Your task: reason deeply about the strategic situation.
Think through:
- What are your win conditions right now?
- What does your opponent likely have or want to do?
- What plays are available to you this turn?
- What are the risks and rewards of each option?
- Are there any interactions, tricks, or combat math to consider?

Do not give a final recommendation yet — just think out loud.
""".format(
        observation=observation,
        card_context=state.get("card_context", "No card data retrieved."),
        iteration=state["iteration"],
    )


def act_prompt(state: AgentState, thought: str) -> str:
    return """
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
""".format(
        thought=thought,
        iteration=state["iteration"],
    )


def check_prompt(state: AgentState, action: str) -> str:
    return """
You are evaluating whether a Magic: The Gathering play recommendation is complete and confident enough to deliver.

Proposed play (iteration {iteration}):
{action}

Reply with exactly one of:
- "DONE" — if the recommendation is clear, complete, and actionable
- "CONTINUE" — if the reasoning needs another pass (e.g. confidence is LOW, or key interactions were not addressed)

Reply with only DONE or CONTINUE and nothing else.
""".format(
        action=action,
        iteration=state["iteration"],
    )
