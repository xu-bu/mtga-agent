import os
from typing import TypedDict
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


class MtgCard(TypedDict):
    """MTG card data structure from Qdrant."""

    name: str
    mana_cost: str
    cmc: float
    type_line: str
    oracle_text: str
    colors: list[str]
    color_identity: list[str]
    set_name: str
    set: str
    released_at: str
    legalities: dict[str, str]
    keywords: list[str]
    rulings: list
    scryfall_uri: str
    image_uri: str
    rarity: str
    power: str | None
    toughness: str | None


# Config from env
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "mtg_cards")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")

# Clients
# embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# def retrieve_by_embed_query(query: str, top_k: int | None = None) -> list[dict]:
#     """Embed query and return top_k matching card payloads."""
#     card_count = len([item.strip() for item in query.split("|") if item.strip()])
#     # LangChain embeddings return a list of floats
#     vector = embeddings.embed_query(query)
#     hits = qdrant.query_points(
#         collection_name=QDRANT_COLLECTION,
#         query=vector,
#         limit=card_count,
#         with_payload=True,
#     ).points

#     return [h.payload for h in hits if h.payload is not None]


def retrieve(cards: list[str]) -> list[MtgCard] | str:
    """Get exact card data by name using single Qdrant query."""
    if not cards:
        return "No cards specified."

    # Use scroll API with multiple "should" conditions
    points, _ = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter={
            "should": [
                {"key": "name", "match": {"value": card_name}}
                for card_name in cards
            ]
        },
        limit=len(cards),
        with_payload=True,
    )

    found_cards: list[MtgCard] = [
        point.payload for point in points if point.payload
    ]

    if not found_cards:
        return "No cards found."

    return found_cards


def format_card_context(cards: list[MtgCard] | str) -> str:
    """Format card data into a readable string for LLM prompts."""
    if isinstance(cards, str):
        return cards

    if not cards:
        return "No card data retrieved."

    lines = []
    for card in cards:
        card_info = f"**{card['name']}**"
        if card.get("mana_cost"):
            card_info += f" | {card['mana_cost']}"
        if card.get("type_line"):
            card_info += f" | {card['type_line']}"
        if card.get("power") and card.get("toughness"):
            card_info += f" | {card['power']}/{card['toughness']}"
        lines.append(card_info)

        if card.get("oracle_text"):
            lines.append(f"  {card['oracle_text']}")
        lines.append("")

    return "\n".join(lines)
