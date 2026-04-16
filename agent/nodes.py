from agent.state import AgentState
from agent.agents import ObserverAgent, StrategistAgent, RuleMasterAgent

# Initialize agent instances
observer_agent = ObserverAgent()
strategist_agent = StrategistAgent()
rulemaster_agent = RuleMasterAgent()


def observe(state: AgentState) -> dict:
    """Extract structured observations using the Observer agent."""
    return observer_agent.observe(state)


def think(state: AgentState) -> dict:
    """Reason over observations using the Strategist agent."""
    return strategist_agent.think(state)


def act(state: AgentState) -> dict:
    """Decide on concrete actions using the Strategist agent."""
    return strategist_agent.act(state)


def check(state: AgentState) -> dict:
    """Evaluate completion using the RuleMaster agent."""
    return rulemaster_agent.check(state)
