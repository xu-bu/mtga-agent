# MTGA Advisor

A simple Magic: The Gathering Arena advisor using a LangGraph agent loop.

The agent accepts a battlefield description, extracts key game state facts, reasons through possible plays, and returns a final recommendation.

## Project structure

- `main.py` — entrypoint that builds the agent graph and runs the loop
- `agent/state.py` — typed agent state definition
- `agent/nodes.py` — observe / think / act / check node implementations
- `agent/graph.py` — graph definition and loop wiring
- `agent/prompts.py` — prompt templates for each node
- `tools/rag.py` — placeholder for card data retrieval
- `AGENTS.md` — project and design documentation

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your model API key to a `.env` file:

```bash
echo "MODEL_API_KEY=your_api_key_here" > .env
```

## Run

```bash
python main.py
```

## Notes

- The agent currently uses a raw battlefield description plus structured state fields.
- `agent/state.py` models detailed MTG zones such as hand, battlefield, graveyard, exile, and mana pools.
- The code is intended as a starting point for later integration with RAG tools and live card lookup.
