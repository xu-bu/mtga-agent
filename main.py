import os
from typing import cast
from dotenv import load_dotenv

load_dotenv()
MODEL_API_KEY = os.getenv("MODEL_API_KEY")

from agent.graph import build_graph
from agent.state import AgentState


EXAMPLE_STATE = {
    "your_hand": ["Lightning Bolt", "Counterspell", "Island"],
    "opponent_hand": 3,
    "your_battlefield": ["Island", "Island", "Mountain", "Nissa, Worldsoul Speaker", "Mine Security"],
    "opponent_battlefield": ["Tarmogoyf", "Forest"],
    "your_graveyard": [],
    "opponent_graveyard": "creature",
    "your_exile": "none",
    "opponent_exile": "none",
    "your_mana_pool": "none",
    "opponent_mana_pool": "none",
    "phase": "main phase",
    "turn_player": "me",
    "stack": "none",
    "life_total": "12",
    "opponent_life_total": "6",
}


def main():
    graph = build_graph()

    initial_state: AgentState = cast(
        AgentState,
        {
            **EXAMPLE_STATE,
            "messages": [],
            "observations": [],
            "thoughts": [],
            "actions_taken": [],
            "iteration": 1,
            "final_recommendation": "",
            "card_context": "",
            "done": False,
        },
    )

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
