# MTGA Advisor

A simple Magic: The Gathering Arena advisor using a LangGraph agent loop.

The agent accepts a battlefield description, extracts key game state facts, reasons through possible plays, and returns a final recommendation.

## Project structure

- `main.py` — entrypoint that builds the agent graph and runs the loop
- `agent/state.py` — typed agent state definition
- `agent/nodes.py` — observe / think / act / check node implementations
- `agent/graph.py` — graph definition and loop wiring
- `agent/prompts.py` — prompt templates for each node
- `tools/rag.py` — helper for retrieving MTG card data from a Qdrant vector store
- `AGENTS.md` — project and design documentation

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your model API key and Qdrant settings to a `.env` file:

```bash
echo "MODEL_API_KEY=your_api_key_here" > .env
echo "QDRANT_URL=https://your-qdrant-host" >> .env
echo "QDRANT_API_KEY=your_qdrant_api_key" >> .env
echo "QDRANT_COLLECTION=mtg_cards" >> .env
```

## Run

```bash
python main.py
```

## Notes

- The agent currently uses a raw battlefield description plus structured state fields.
- `agent/state.py` models detailed MTG zones such as hand, battlefield, graveyard, exile, and mana pools.
- `tools/rag.py` can retrieve MTG card data from a Qdrant store and format it for LLM prompts.
