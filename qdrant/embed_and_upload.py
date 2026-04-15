"""
Step 2: Parse oracle_cards.json, build text chunks, embed with Qwen,
and upsert into Qdrant Cloud.

Chunk strategy  : one vector per card  (oracle text + rulings joined)
Payload stored  : name, mana_cost, type_line, oracle_text, rulings,
                  set_name, legalities, colors, cmc, scryfall_uri
"""

import json
import os
import time
import uuid
from typing import Any

import requests
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

load_dotenv()

Card = dict[str, Any]
Ruling = dict[str, Any]

# ── Config ──────────────────────────────────────────────────────────────────
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION = os.getenv("QDRANT_COLLECTION", "mtg_cards")
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "oracle_cards.json")
EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")
BATCH_SIZE = 1
MAX_CARDS = 0  # Set e.g. 500 to do a quick test run, 0 means all

# ── Clients / model cache ───────────────────────────────────────────────────
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
_tokenizer: Any | None = None
_model: Any | None = None
vector_size = 0


# ── Helpers ─────────────────────────────────────────────────────────────────


def build_chunk(card: Card) -> str:
    """
    Combine the fields that matter most for semantic search into one string.
    Rulings are appended so 'does X interact with Y' queries can surface them.
    """
    parts = [
        f"Card: {card.get('name', '')}",
        f"Mana cost: {card.get('mana_cost', 'none')}",
        f"Type: {card.get('type_line', '')}",
    ]

    oracle = str(card.get("oracle_text", "")).strip()
    if oracle:
        parts.append(f"Rules text: {oracle}")

    rulings = card.get("rulings", [])
    if rulings:
        ruling_lines = "\n".join(f"- {r['comment']}" for r in rulings)
        parts.append(f"Rulings:\n{ruling_lines}")

    keywords = card.get("keywords", [])
    if keywords:
        parts.append(f"Keywords: {', '.join(str(keyword) for keyword in keywords)}")

    return "\n".join(parts)


def build_payload(card: Card, rulings: list[Ruling]) -> Card:
    """Metadata stored alongside the vector — filterable in Qdrant."""
    return {
        "name": card.get("name", ""),
        "mana_cost": card.get("mana_cost", ""),
        "cmc": card.get("cmc", 0),
        "type_line": card.get("type_line", ""),
        "oracle_text": card.get("oracle_text", ""),
        "colors": card.get("colors", []),
        "color_identity": card.get("color_identity", []),
        "set_name": card.get("set_name", ""),
        "set": card.get("set", ""),
        "released_at": card.get("released_at", ""),
        "legalities": card.get("legalities", {}),
        "keywords": card.get("keywords", []),
        "rulings": [r["comment"] for r in rulings],
        "scryfall_uri": card.get("scryfall_uri", ""),
        "image_uri": (card.get("image_uris") or {}).get("normal", ""),
        "rarity": card.get("rarity", ""),
        "power": card.get("power", ""),
        "toughness": card.get("toughness", ""),
    }


def _coerce_embeddings(raw_embeddings: Any) -> list[list[float]]:
    if isinstance(raw_embeddings, torch.Tensor):
        rows = raw_embeddings.cpu().tolist()
    elif hasattr(raw_embeddings, "tolist"):
        rows = raw_embeddings.tolist()
    else:
        rows = raw_embeddings

    return [
        [float(value) for value in row]
        for row in rows
    ]


def _detect_vector_size(tokenizer: Any, model: Any) -> int:
    encode = getattr(model, "encode", None)
    if callable(encode):
        return len(_coerce_embeddings(encode(["dimension probe"]))[0])

    inputs = tokenizer(
        "dimension probe",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        return int(outputs.last_hidden_state.shape[-1])


def ensure_qwen_loaded() -> tuple[Any, Any]:
    """Lazy-load tokenizer/model and detect vector dimension once."""
    global _model, _tokenizer, vector_size

    if _tokenizer is None or _model is None:
        print(f"Loading Qwen model from {EMBED_MODEL}...")
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            EMBED_MODEL, trust_remote_code=True
        )
        loaded_model = AutoModel.from_pretrained(EMBED_MODEL, trust_remote_code=True)
        loaded_model.eval()

        _tokenizer = loaded_tokenizer
        _model = loaded_model
        vector_size = _detect_vector_size(_tokenizer, _model)
        print(f"Qwen model loaded. Vector dimension: {vector_size}")


    return _tokenizer, _model


def embed_by_qwen(texts: list[str]) -> list[list[float]]:
    """Embed texts using the local Hugging Face Qwen embedding model."""
    tokenizer, model = ensure_qwen_loaded()

    encode = getattr(model, "encode", None)
    if callable(encode):
        return _coerce_embeddings(encode(texts))

    embeddings: list[list[float]] = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().float().cpu().numpy()
        embeddings.append([float(value) for value in embedding.tolist()])

    return embeddings


def ensure_collection() -> None:
    ensure_qwen_loaded()
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in existing:
        print(f"Creating collection '{COLLECTION}' ...")
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        return

    info = qdrant.get_collection(collection_name=COLLECTION)
    vectors_config = info.config.params.vectors
    existing_size = getattr(vectors_config, "size", None)
    if existing_size is None and isinstance(vectors_config, dict):
        first_vector = next(iter(vectors_config.values()), None)
        existing_size = getattr(first_vector, "size", None)

    if existing_size is not None and existing_size != vector_size:
        raise ValueError(
            f"Collection '{COLLECTION}' uses vector size {existing_size}, "
            f"but {EMBED_MODEL} returns {vector_size}. Recreate the collection or "
            "point QDRANT_COLLECTION to a fresh collection."
        )

    print(f"Collection '{COLLECTION}' already exists — will upsert.")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Loading {DATA_FILE} ...")
    with open(DATA_FILE, encoding="utf-8") as f:
        all_cards: list[Card] = json.load(f)

    # Filter: skip tokens, art cards, and cards with no oracle text.
    cards = [
        card
        for card in all_cards
        if card.get("oracle_text") is not None
        and card.get("layout")
        not in ("token", "emblem", "art_series", "double_faced_token")
    ]
    if MAX_CARDS > 0:
        cards = cards[:MAX_CARDS]

    print(f"Cards to index: {len(cards):,}")
    ensure_collection()

    # Fetch existing IDs to avoid duplicates
    print(f"Fetching existing IDs from '{COLLECTION}'...")
    existing_ids: set[str] = set()
    next_page = None
    while True:
        points, next_page = qdrant.scroll(
            collection_name=COLLECTION,
            limit=1000,
            with_payload=False,
            with_vectors=False,
            offset=next_page,
        )
        for point in points:
            existing_ids.add(str(point.id))
        if next_page is None:
            break
    print(f"Found {len(existing_ids):,} existing points in Qdrant.")

    # ── Fetch rulings from Scryfall (one bulk file, cached) ──────────────────
    rulings_map: dict[str, list[Ruling]] = {}
    rulings_file = os.path.join(os.path.dirname(DATA_FILE), "rulings.json")
    if os.path.exists(rulings_file):
        print("Loading cached rulings ...")
        with open(rulings_file, encoding="utf-8") as f:
            raw_rulings: list[Ruling] = json.load(f)
        for ruling in raw_rulings:
            rulings_map.setdefault(str(ruling["oracle_id"]), []).append(ruling)
    else:
        print("Downloading rulings bulk file ...")
        resp = requests.get("https://api.scryfall.com/bulk-data", timeout=30)
        resp.raise_for_status()
        for entry in resp.json()["data"]:
            if entry["type"] == "rulings":
                dl = requests.get(entry["download_uri"], timeout=120)
                dl.raise_for_status()
                raw_rulings = dl.json()
                with open(rulings_file, "w", encoding="utf-8") as f:
                    json.dump(raw_rulings, f)
                for ruling in raw_rulings:
                    rulings_map.setdefault(str(ruling["oracle_id"]), []).append(ruling)
                break

    # ── Batch embed + upsert ─────────────────────────────────────────────────
    points_buffer: list[PointStruct] = []
    total_upserted = 0

    for index in tqdm(range(0, len(cards), BATCH_SIZE), desc="Embedding & uploading"):
        batch = cards[index : index + BATCH_SIZE]
        rulings_batch = [rulings_map.get(str(card.get("oracle_id", "")), []) for card in batch]

        # Filter to new cards only
        batch_ids = [str(uuid.UUID(str(card["oracle_id"]))) for card in batch]
        filtered_work = [
            (card, rulings)
            for card, rulings, batch_id in zip(batch, rulings_batch, batch_ids)
            if batch_id not in existing_ids
        ]

        if not filtered_work:
            continue

        batch, rulings_batch = map(list, zip(*filtered_work))
        texts = [build_chunk({**card, "rulings": rulings}) for card, rulings in zip(batch, rulings_batch)]

        vectors: list[list[float]] = []
        for attempt in range(3):
            try:
                vectors = embed_by_qwen(texts)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 10 * (attempt + 1)
                print(f"\nEmbedding error ({exc}), waiting {wait}s ...")
                time.sleep(wait)

        for card, rulings, vector in zip(batch, rulings_batch, vectors):
            points_buffer.append(
                PointStruct(
                    id=str(uuid.UUID(str(card["oracle_id"]))),
                    vector=vector,
                    payload=build_payload(card, rulings),
                )
            )

        # Upsert every 200 points
        if len(points_buffer) >= 200:
            qdrant.upsert(collection_name=COLLECTION, points=points_buffer)
            total_upserted += len(points_buffer)
            points_buffer = []

    # Final flush
    if points_buffer:
        qdrant.upsert(collection_name=COLLECTION, points=points_buffer)
        total_upserted += len(points_buffer)

    print(f"\nDone! Upserted {total_upserted:,} cards into '{COLLECTION}'.")


if __name__ == "__main__":
    main()
