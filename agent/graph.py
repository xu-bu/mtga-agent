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
