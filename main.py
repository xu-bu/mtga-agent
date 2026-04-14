import os
from dotenv import load_dotenv

load_dotenv()
MODEL_API_KEY = os.getenv("MODEL_API_KEY")

from agent.graph import build_graph
from agent.state import AgentState
from tools.rag import retrieve, format_card_context


def build_initial_state(base_state: dict) -> AgentState:
    """Build initial state with pre-fetched card context."""
    # Collect all known cards from all zones
    all_known_cards = (
        base_state["your_hand"]
        + base_state["your_battlefield"]
        + base_state["opponent_battlefield"]
        + base_state["your_graveyard"]
        + base_state["opponent_graveyard"]
        + base_state["your_exile"]
        + base_state["opponent_exile"]
    )
    
    # Pre-fetch and format card data
    cards = retrieve(all_known_cards)
    card_context = format_card_context(cards)
    print(f"[PREPROCESS] Retrieved {len(cards) if isinstance(cards, list) else 0} cards.")
    
    # Build complete initial state
    return {
        **base_state,
        "messages": [],
        "observations": [],
        "thoughts": [],
        "actions_taken": [],
        "iteration": 1,
        "final_recommendation": "",
        "card_context": card_context,
        "done": False,
    }


EXAMPLE_STATE: AgentState = {
    "your_hand": ["Snarlfang Vermin", "Walking Sponge", "Island"],
    "opponent_hand": "3 cards (unknown)",
    "your_battlefield": ["Island", "Island", "Mountain", "Nissa, Worldsoul Speaker", "Mine Security"],
    "opponent_battlefield": ["Storm Crow", "Forest"],
    "your_graveyard": [],
    "opponent_graveyard": [],
    "your_exile": [],
    "opponent_exile": [],
    "your_mana_pool": "none",
    "opponent_mana_pool": "none",
    "phase": "main phase",
    "turn_player": "me",
    "stack": "none",
    "life_total": 12,
    "opponent_life_total": 6,
}


def main():
    graph = build_graph()

    initial_state = build_initial_state(EXAMPLE_STATE)
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
