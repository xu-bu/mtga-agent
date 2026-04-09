import os
from dotenv import load_dotenv

load_dotenv()
MODEL_API_KEY = os.getenv("MODEL_API_KEY")

from agent.graph import build_graph
from agent.state import AgentState

EXAMPLE_BATTLEFIELD = """
Turn 5, my main phase. My turn.

My hand: Lightning Bolt, Counterspell, Island (land)
Opponent's hand: 1
My battlefield: 2x Island, 1x Mountain (untapped), Goblin Guide (2/2 haste), Snapcaster Mage (2/1)
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
