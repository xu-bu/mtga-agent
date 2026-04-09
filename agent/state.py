from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Input
    your_hand: str  # your hand contents
    opponent_hand: str  # opponent hand information (count or known cards)
    your_battlefield: str  # permanents you control
    opponent_battlefield: str  # permanents opponent controls
    your_graveyard: str  # your graveyard contents
    opponent_graveyard: str  # opponent graveyard contents
    your_exile: str  # cards you have exiled
    opponent_exile: str  # cards opponent has exiled
    your_mana_pool: str  # your available mana this turn
    opponent_mana_pool: str  # opponent's available mana this turn
    phase: str  # current phase of the turn
    turn_player: str  # whose turn it is
    stack: str  # any relevant stack state
    life_total: str  # your life total
    opponent_life_total: str  # opponent life total

    # Loop internals
    messages: Annotated[list, add_messages]  # full conversation with the LLM
    observations: list[str]  # what the agent noticed each iteration
    thoughts: list[str]  # reasoning produced each iteration
    actions_taken: list[str]  # what the agent decided to do each iteration
    iteration: int  # loop counter (safety limit)

    # Output
    final_recommendation: str  # populated when the loop exits
    card_context: str  # retrieved card data and rules
    done: bool  # flag that tells LangGraph to exit the loop
