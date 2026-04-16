# MTGA Advisor

A Magic: The Gathering Arena advisor using a LangGraph multi-agent system with exact card data retrieval via Qdrant.

The system uses three specialized agents (Observer, Strategist, RuleMaster) that work in sequence to analyze battlefield state, retrieve exact card data, reason through possible plays, and return a final recommendation.

## Screenshots

![1776220883719](image/README/1776220883719.png)

## Project structure

- `main.py` — entrypoint that pre-fetches card data and runs the agent loop
- `agent/state.py` — typed agent state definition with MTG zones
- `agent/agents.py` — specialized agent classes (Observer, Strategist, RuleMaster)
- `agent/nodes.py` — observe / think / act / check node implementations that delegate to agents
- `agent/graph.py` — graph definition and loop wiring
- `agent/prompts.py` — prompt templates for each agent with distinct personas
- `tools/rag.py` — exact card data retrieval using Qdrant
- `constants.py` — configuration constants (e.g., MAX_ITERATIONS)
- `AGENTS.md` — detailed project and design documentation

## Architecture

**Multi-agent flow:**

```
Observer Agent → Strategist Agent (think + act) → RuleMaster Agent (check)
                                                    ↓ (loop if not done)
                                              Strategist Agent
                                                    ↓
                                              RuleMaster Agent
```

**Specialized agents:**

- **Observer Agent** (temperature=0): Precisely extracts structured facts from battlefield state with maximum accuracy. Runs once at the start.

- **Strategist Agent** (temperature=0.3): Handles both strategic reasoning (think) and tactical action planning (act). Uses slightly higher temperature for creative strategic thinking while maintaining practicality.

- **RuleMaster Agent** (temperature=0): Evaluates recommendations for completeness, rules compliance, and confidence. Ensures quality control before final output.

**Agent responsibilities:**

- **observe**: Extracts facts from battlefield state and card data (Observer Agent)
- **think**: Reasons over observations and card data (Strategist Agent)
- **act**: Proposes concrete tactical actions (Strategist Agent)
- **check**: Decides if recommendation is confident, complete, and rules-compliant (RuleMaster Agent)

**Card retrieval:**

- Uses Qdrant vector database with exact name matching (not semantic search)
- Cards are pre-fetched before the agent loop runs
- Single query retrieves all cards at once using "should" filter

**Streaming:**

- LLM responses stream token-by-token for immediate feedback
- Only observe output and final recommendation are printed

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your API keys to a `.env` file:

```bash
MODEL_API_KEY=your_model_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
MODEL_NAME=your_model_name
```

## Run

```bash
python main.py
```

## Configuration

Edit `constants.py` to adjust:

- `MAX_ITERATIONS`: Maximum number of think-act-check loops (default: 3)

## Notes

- State uses lists for card zones (hand, battlefield, graveyard, exile)
- Card data is retrieved via exact name matching, not semantic search
- The system uses a Sequential Specialists multi-agent architecture with three specialized agents
- Each agent has distinct temperature settings and personas optimized for their specific role
- The system is optimized for structured MTG game states with known card names
