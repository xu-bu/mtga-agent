import os
from agent.state import AgentState
from agent.prompts import observe_prompt, think_prompt, act_prompt, check_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from constants import MAX_ITERATIONS

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_NAME"),
    google_api_key=os.getenv("MODEL_API_KEY"),
    temperature=0,
)


def observe(state: AgentState) -> dict:
    """Extract structured observations from the battlefield description."""
    prompt = observe_prompt(state)
    print(f"\n[OBSERVE — iteration {state['iteration']}]\n")

    observation = ""
    for chunk in llm.stream([{"role": "user", "content": prompt}]):
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            observation += content
    print()  # New line after streaming

    return {
        "observations": state["observations"] + [observation],
        "messages": [{"role": "assistant", "content": observation}],
    }


def think(state: AgentState) -> dict:
    """Reason over the observations and produce a plan."""
    latest_obs = state["observations"][-1]
    prompt = think_prompt(state, latest_obs)

    thought = ""
    for chunk in llm.stream([{"role": "user", "content": prompt}]):
        content = chunk.content
        if content:
            thought += content

    return {
        "thoughts": state["thoughts"] + [thought],
        "messages": [{"role": "assistant", "content": thought}],
    }


def act(state: AgentState) -> dict:
    """Decide on a concrete action based on current reasoning."""
    latest_thought = state["thoughts"][-1]
    prompt = act_prompt(state, latest_thought)

    action = ""
    for chunk in llm.stream([{"role": "user", "content": prompt}]):
        content = chunk.content
        if content:
            action += content

    return {
        "actions_taken": state["actions_taken"] + [action],
        "messages": [{"role": "assistant", "content": action}],
    }


def check(state: AgentState) -> dict:
    """Decide: are we done, or do we need another pass?"""
    latest_action = state["actions_taken"][-1]

    if state["iteration"] >= MAX_ITERATIONS:
        # Force exit — safety limit
        done = True
        recommendation = latest_action
    else:
        prompt = check_prompt(state, latest_action)
        response = llm.invoke([{"role": "user", "content": prompt}])
        raw_content = response.content
        if isinstance(raw_content, str):
            content_text = raw_content
        else:
            content_text = " ".join(
                (
                    item["content"]
                    if isinstance(item, dict) and "content" in item
                    else str(item)
                )
                for item in raw_content
            )
        done = content_text.strip().lower().startswith("done")
        recommendation = latest_action if done else ""

    return {
        "done": done,
        "final_recommendation": (
            recommendation if done else state.get("final_recommendation", "")
        ),
        "iteration": state["iteration"] + 1,
    }
