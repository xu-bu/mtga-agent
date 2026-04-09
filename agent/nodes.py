import os
from agent.state import AgentState
from agent.prompts import OBSERVE_PROMPT, THINK_PROMPT, ACT_PROMPT, CHECK_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from tools.rag import get_card_data

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME"),
    google_api_key=os.getenv("MODEL_API_KEY"),
    temperature=0
)

def observe(state: AgentState) -> dict:
    """Extract structured observations from the battlefield description."""
    prompt = OBSERVE_PROMPT.format(battlefield=state["battlefield"])
    response = llm.invoke([{"role": "user", "content": prompt}])
    observation = response.content
    print(f"\n[OBSERVE — iteration {state['iteration']}]\n{observation}")
    
    # Fetch ground-truth card data based on battlefield description
    card_context = get_card_data(state["battlefield"])
    print(f"\n[RAG — iteration {state['iteration']}]\nRetrieved data for {len(card_context.split('---'))} cards/sections.")

    return {
        "observations": state["observations"] + [observation],
        "messages": [{"role": "assistant", "content": observation}],
        "card_context": card_context
    }

def think(state: AgentState) -> dict:
    """Reason over the observations and produce a plan."""
    latest_obs = state["observations"][-1]
    prompt = THINK_PROMPT.format(
        battlefield=state["battlefield"],
        observation=latest_obs,
        card_context=state.get("card_context", "No card data retrieved."),
        iteration=state["iteration"],
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    thought = response.content
    print(f"\n[THINK — iteration {state['iteration']}]\n{thought}")
    return {
        "thoughts": state["thoughts"] + [thought],
        "messages": [{"role": "assistant", "content": thought}],
    }

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

def check(state: AgentState) -> dict:
    """Decide: are we done, or do we need another pass?"""
    latest_action = state["actions_taken"][-1]
    MAX_ITERATIONS = 1

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
