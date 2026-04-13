import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Config from env
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mtg_cards")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
TOP_K = int(os.getenv("RAG_TOP_K", "100"))

# Clients
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """Embed query and return top_k matching card payloads."""
    card_count = len([item.strip() for item in query.split("|") if item.strip()])
    if top_k is None:
        top_k = min(card_count, TOP_K)

    # LangChain embeddings return a list of floats
    vector = embeddings.embed_query(query)
    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points

    print(f"[RAG] estimated {card_count} cards, using top_k={top_k}")
    return [h.payload for h in hits if h.payload is not None]


def format_card_context(cards: list[dict]) -> str:
    """Format retrieval results into a string for the prompt."""
    if not cards:
        return "No specific card data found."
    sections = []
    for c in cards:
        # Handle cases where payload might be missing fields
        name = c.get("name", "Unknown Card")
        mana_cost = c.get("mana_cost", "")
        type_line = c.get("type_line", "")
        oracle_text = c.get("oracle_text", "(none)")
        lines = [
            f"### {name} {mana_cost}",
            f"Type: {type_line}",
            f"Rules text: {oracle_text}",
        ]
        rulings = c.get("rulings", [])
        if rulings:
            lines.append("Rulings:")
            for r in rulings:
                # Some qdrant payloads might store rulings as dicts or strings
                if isinstance(r, dict):
                    text = r.get("comment", r.get("text", str(r)))
                    lines.append(f"  • {text}")
                else:
                    lines.append(f"  • {r}")

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


def get_card_data(query: str, top_k: int | None = None) -> str:
    """Embed query, retrieve matching card payloads, and format them."""
    card_count = len([item.strip() for item in query.split("|") if item.strip()])
    if top_k is None:
        top_k = min(card_count, TOP_K)

    vector = embeddings.embed_query(query)
    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
    ).points

    print(f"[RAG] get_card_data estimated {card_count} cards, using top_k={top_k}")
    cards = [h.payload for h in hits if h.payload is not None]
    return format_card_context(cards)
