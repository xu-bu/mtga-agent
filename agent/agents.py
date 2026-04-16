import os
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.state import AgentState
from agent.prompts import (
    observe_prompt,
    strategist_think_prompt,
    strategist_act_prompt,
    rulemaster_check_prompt,
)


class ObserverAgent:
    """Specialized agent for fact extraction with high precision."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME"),
            google_api_key=os.getenv("MODEL_API_KEY"),
            temperature=0,  # Maximum precision for fact extraction
        )

    def observe(self, state: AgentState) -> dict:
        """Extract structured observations from the battlefield description."""
        prompt = observe_prompt(state)
        print(f"\n[OBSERVER AGENT — iteration {state['iteration']}]\n")

        observation = ""
        for chunk in self.llm.stream([{"role": "user", "content": prompt}]):
            content = chunk.content
            if content:
                print(content, end="", flush=True)
                observation += content
        print()

        return {
            "observations": state["observations"] + [observation],
            "messages": [{"role": "assistant", "content": observation}],
        }


class StrategistAgent:
    """Specialized agent for strategic reasoning and tactical planning."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME"),
            google_api_key=os.getenv("MODEL_API_KEY"),
            temperature=0.3,  # Slightly higher creativity for strategic thinking
        )

    def think(self, state: AgentState) -> dict:
        """Reason over the observations and produce a strategic plan."""
        latest_obs = state["observations"][-1]
        prompt = strategist_think_prompt(state, latest_obs)

        thought = ""
        for chunk in self.llm.stream([{"role": "user", "content": prompt}]):
            content = chunk.content
            if content:
                thought += content

        return {
            "thoughts": state["thoughts"] + [thought],
            "messages": [{"role": "assistant", "content": thought}],
        }

    def act(self, state: AgentState) -> dict:
        """Decide on concrete tactical actions based on strategic reasoning."""
        latest_thought = state["thoughts"][-1]
        prompt = strategist_act_prompt(state, latest_thought)

        action = ""
        for chunk in self.llm.stream([{"role": "user", "content": prompt}]):
            content = chunk.content
            if content:
                action += content

        return {
            "actions_taken": state["actions_taken"] + [action],
            "messages": [{"role": "assistant", "content": action}],
        }


class RuleMasterAgent:
    """Specialized agent for rules verification and quality control."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL_NAME"),
            google_api_key=os.getenv("MODEL_API_KEY"),
            temperature=0,  # Maximum precision for rules checking
        )

    def check(self, state: AgentState) -> dict:
        """Evaluate if the recommendation is complete and rules-compliant."""
        latest_action = state["actions_taken"][-1]

        from constants import MAX_ITERATIONS

        if state["iteration"] >= MAX_ITERATIONS:
            done = True
            recommendation = latest_action
            print(f"\n[RULEMASTER AGENT] Max iterations reached. Forcing exit.")
        else:
            prompt = rulemaster_check_prompt(state, latest_action)
            response = self.llm.invoke([{"role": "user", "content": prompt}])
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
            print(f"\n[RULEMASTER AGENT — iteration {state['iteration']}] done={done}")

        return {
            "done": done,
            "final_recommendation": (
                recommendation if done else state.get("final_recommendation", "")
            ),
            "iteration": state["iteration"] + 1,
        }
