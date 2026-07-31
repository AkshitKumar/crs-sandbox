"""Build a validated embedding matrix in catalog order."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer

from sandbox.product_text import PRODUCT_TEXT_VERSION, serialize_product


MODEL_NAME = "BAAI/bge-large-en-v1.5"
MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
INDEX_SCHEMA_VERSION = "dense-index-v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ordered_asin_sha256(products: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(product.get("asin") or "") for product in products)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
    model_revision: str = MODEL_REVISION,
    batch_size: int = 32,
) -> None:
    """Build and persist the embedding index for a category."""
    index_dir.mkdir(parents=True, exist_ok=True)

    products = load_products(catalog_path)
    if not products:
        raise ValueError(f"No products in {catalog_path}")

    print(f"[index] loading model {model_name}...")
    model = SentenceTransformer(model_name, revision=model_revision)

    texts = [serialize_product(product) for product in products]
    print(f"[index] encoding {len(texts)} products...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,  # cosine via dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    emb_path = index_dir / "embeddings.npy"
    manifest_path = index_dir / "index_manifest.json"
    np.save(emb_path, embeddings.astype(np.float32))
    manifest = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "model_id": model_name,
        "model_revision": model_revision,
        "product_text_version": PRODUCT_TEXT_VERSION,
        "product_count": len(products),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": "float32",
        "catalog_sha256": _sha256(catalog_path),
        "ordered_asin_sha256": _ordered_asin_sha256(products),
        "embeddings_sha256": _sha256(emb_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[index] wrote {emb_path} (shape={embeddings.shape})")
    print(f"[index] wrote {manifest_path}")
