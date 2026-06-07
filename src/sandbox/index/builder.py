"""Build a vector index over a category's scraped products.

Encodes each product as: title + top bullets + short description tail, using
BAAI/bge-large-en-v1.5. Persists embeddings as .npy plus a parallel
products.json metadata file (ASIN order matches row order in the embedding).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Cap per field to keep within model context (~512 tokens).
MAX_BULLET_CHARS = 1200
MAX_DESC_CHARS = 600


def _product_text(product: dict[str, Any]) -> str:
    title = product.get("title") or ""
    bullets = product.get("bullets") or []
    bullets_joined = " | ".join(bullets)[:MAX_BULLET_CHARS]
    desc = (product.get("description") or "")[:MAX_DESC_CHARS]
    parts = [title]
    if bullets_joined:
        parts.append(bullets_joined)
    if desc:
        parts.append(desc)
    return "\n".join(parts)


def load_products(catalog_path: Path) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    with catalog_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                products.append(json.loads(line))
    return products


def build_index(
    catalog_path: Path,
    index_dir: Path,
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
) -> None:
    """Build and persist the embedding index for a category."""
    index_dir.mkdir(parents=True, exist_ok=True)

    products = load_products(catalog_path)
    if not products:
        raise ValueError(f"No products in {catalog_path}")

    print(f"[index] loading model {model_name}...")
    model = SentenceTransformer(model_name)

    texts = [_product_text(p) for p in products]
    print(f"[index] encoding {len(texts)} products...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # cosine via dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "products.json"
    np.save(emb_path, embeddings.astype(np.float32))
    with meta_path.open("w") as f:
        json.dump(products, f)

    print(f"[index] wrote {emb_path} (shape={embeddings.shape})")
    print(f"[index] wrote {meta_path} ({len(products)} products)")
