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

# Clients
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def retrieve_by_embed_query(query: str, top_k: int | None = None) -> list[dict]:
    """Embed query and return top_k matching card payloads."""
    card_count = len([item.strip() for item in query.split("|") if item.strip()])
    # LangChain embeddings return a list of floats
    vector = embeddings.embed_query(query)
    hits = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=vector,
        limit=card_count,
        with_payload=True,
    ).points

    return [h.payload for h in hits if h.payload is not None]

def retrieve(cards: list[str]) -> str:
    """Get exact card data by name using single Qdrant query."""
    if not cards:
        return "No cards specified."
    
    try:
        # Use scroll API with multiple "should" conditions
        scroll_result = qdrant.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter={
                "should": [
                    {
                        "key": "name",
                        "match": {"value": card_name}
                    }
                    for card_name in cards
                ]
            },
            limit=len(cards),
            with_payload=True
        )
        
        found_cards = [point.payload for point in scroll_result[0] if point.payload]
        
        if not found_cards:
            return "No cards found."
            
        return found_cards
        
    except Exception as e:
        return f"Error retrieving cards: {e}"
    