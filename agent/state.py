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
    card_context: str           # retrieved card data and rules
    done: bool                  # flag that tells LangGraph to exit the loop
