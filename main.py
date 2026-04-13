import os
from typing import cast
from dotenv import load_dotenv

load_dotenv()
MODEL_API_KEY = os.getenv("MODEL_API_KEY")

from agent.graph import build_graph
from agent.state import AgentState


EXAMPLE_STATE = {
    "your_hand": "Lightning Bolt|Counterspell|Island",
    "opponent_hand": "unknown card",
    "your_battlefield": "Island|Island|Mountain|Goblin Guide|Snapcaster Mage",
    "opponent_battlefield": "Tarmogoyf|Forest",
    "your_graveyard": "instant|instant|sorcery",
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


def validate_card_zone(zone_name: str, value: str) -> None:
    if not value or value.strip().lower() == "none":
        return

    items = [item.strip() for item in value.split("|") if item.strip()]
    if not items:
        return

    for item in items:
        if not item or item.lower() in ["unknown card", "instant", "sorcery", "creature", "land"]:
            continue
        # Basic check: no numbers or 'x' in card names
        if any(char.isdigit() or char in "x×" for char in item):
            raise ValueError(
                f"Invalid format for {zone_name}: '{item}'. Use pipe-separated card names, repeating for multiples (e.g., 'Island|Island|Mountain')."
            )


def enforce_example_state(state: dict) -> None:
    for field in [
        "your_hand",
        "opponent_hand",
        "your_battlefield",
        "opponent_battlefield",
        "your_graveyard",
        "opponent_graveyard",
        "your_exile",
        "opponent_exile",
    ]:
        validate_card_zone(field, state[field])


enforce_example_state(EXAMPLE_STATE)


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
