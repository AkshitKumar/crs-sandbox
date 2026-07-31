"""Auto-generate config + product/raw folders for all 40 categories.

Idempotent: does not overwrite an existing config (so manual edits stick).

Seed queries per category are drawn from a hand-curated map (variant-rich,
covers common shopping intents). Add more variants here if discovery isn't
finding enough products.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# Each category maps to (canonical_slug, display_name, seed_queries).
# slug is what shows up in path names and config keys.
CATEGORY_SPEC: dict[str, tuple[str, str, list[str]]] = {
    "laptop": ("laptop", "Laptop", [
        "laptop", "gaming laptop", "laptop for work", "laptop for students",
        "ultrabook", "2 in 1 laptop",
    ]),
    "air purifier": ("air_purifier", "Air Purifier", [
        "air purifier", "hepa air purifier", "air purifier for large rooms",
        "air purifier for bedroom", "air purifier for allergies", "air purifier for pets",
    ]),
    "headphones": ("headphones", "Headphones", [
        "headphones", "wireless headphones", "noise cancelling headphones",
        "over ear headphones", "headphones for gaming", "studio headphones",
    ]),
    "earbuds": ("earbuds", "Earbuds", [
        "earbuds", "wireless earbuds", "noise cancelling earbuds",
        "earbuds for running", "earbuds for gaming", "true wireless earbuds",
    ]),
    "speaker": ("speaker", "Speaker", [
        "bluetooth speaker", "portable speaker", "outdoor speaker",
        "smart speaker", "bookshelf speaker", "party speaker",
    ]),
    "knife": ("knife", "Knife", [
        "kitchen knife", "chef knife", "pocket knife", "hunting knife",
        "knife set", "japanese knife",
    ]),
    "blender": ("blender", "Blender", [
        "blender", "high speed blender", "personal blender",
        "smoothie blender", "blender for ice", "professional blender",
    ]),
    "kettle": ("kettle", "Kettle", [
        "electric kettle", "stovetop kettle", "tea kettle",
        "gooseneck kettle", "fast boil kettle", "stainless steel kettle",
    ]),
    "vacuum": ("vacuum", "Vacuum", [
        "vacuum cleaner", "robot vacuum", "cordless vacuum",
        "vacuum for pet hair", "stick vacuum", "shop vacuum",
    ]),
    "printer": ("printer", "Printer", [
        "printer", "all in one printer", "laser printer",
        "color printer", "wireless printer", "photo printer",
    ]),
    "fan": ("fan", "Fan", [
        "tower fan", "desk fan", "ceiling fan",
        "bladeless fan", "quiet fan for bedroom", "floor fan",
    ]),
    "humidifier": ("humidifier", "Humidifier", [
        "humidifier", "cool mist humidifier", "humidifier for bedroom",
        "ultrasonic humidifier", "humidifier for baby", "large room humidifier",
    ]),
    "alarm clock": ("alarm_clock", "Alarm Clock", [
        "alarm clock", "digital alarm clock", "sunrise alarm clock",
        "loud alarm clock for heavy sleepers", "alarm clock for bedroom", "smart alarm clock",
    ]),
    "mp3 player": ("mp3_player", "MP3 Player", [
        "mp3 player", "portable mp3 player", "bluetooth mp3 player",
        "mp3 player for running", "high resolution audio player", "kids mp3 player",
    ]),
    "trimmer": ("trimmer", "Trimmer", [
        "beard trimmer", "hair trimmer", "body trimmer",
        "cordless trimmer", "trimmer for men", "trimmer kit",
    ]),
    "razor": ("razor", "Razor", [
        "electric razor", "razor for men", "safety razor",
        "razor blades", "manual razor", "rotary razor",
    ]),
    "bag": ("bag", "Bag", [
        "bag", "messenger bag", "tote bag",
        "shoulder bag", "travel bag", "work bag",
    ]),
    "backpack": ("backpack", "Backpack", [
        "backpack", "laptop backpack", "travel backpack",
        "hiking backpack", "school backpack", "everyday backpack",
    ]),
    "briefcase": ("briefcase", "Briefcase", [
        "briefcase", "leather briefcase", "laptop briefcase",
        "business briefcase", "messenger briefcase", "rolling briefcase",
    ]),
    "wallet": ("wallet", "Wallet", [
        "wallet", "leather wallet", "minimalist wallet",
        "rfid wallet", "bifold wallet", "front pocket wallet",
    ]),
    "boot": ("boot", "Boots", [
        "boots", "work boots", "hiking boots",
        "winter boots", "leather boots", "rain boots",
    ]),
    "shoe": ("shoe", "Shoes", [
        "shoes", "running shoes", "dress shoes",
        "walking shoes", "casual shoes", "athletic shoes",
    ]),
    "sock": ("sock", "Socks", [
        "socks", "athletic socks", "wool socks",
        "compression socks", "no show socks", "dress socks",
    ]),
    "belt": ("belt", "Belt", [
        "belt", "leather belt", "men's belt",
        "reversible belt", "dress belt", "casual belt",
    ]),
    "glove": ("glove", "Gloves", [
        "gloves", "work gloves", "winter gloves",
        "gardening gloves", "leather gloves", "touchscreen gloves",
    ]),
    "shirt": ("shirt", "Shirt", [
        "shirt", "dress shirt", "t-shirt",
        "casual shirt", "button down shirt", "polo shirt",
    ]),
    "pant": ("pant", "Pants", [
        "pants", "dress pants", "jeans",
        "casual pants", "work pants", "khaki pants",
    ]),
    "jacket": ("jacket", "Jacket", [
        "jacket", "winter jacket", "leather jacket",
        "rain jacket", "fleece jacket", "casual jacket",
    ]),
    "sunglasse": ("sunglasses", "Sunglasses", [
        "sunglasses", "polarized sunglasses", "men's sunglasses",
        "aviator sunglasses", "sports sunglasses", "designer sunglasses",
    ]),
    "umbrella": ("umbrella", "Umbrella", [
        "umbrella", "travel umbrella", "windproof umbrella",
        "automatic umbrella", "large umbrella", "compact umbrella",
    ]),
    "chair": ("chair", "Chair", [
        "office chair", "ergonomic chair", "gaming chair",
        "desk chair", "executive chair", "task chair",
    ]),
    "desk": ("desk", "Desk", [
        "desk", "standing desk", "computer desk",
        "writing desk", "l shaped desk", "small desk",
    ]),
    "bed": ("bed", "Bed", [
        "bed frame", "platform bed", "queen bed frame",
        "king bed frame", "storage bed", "metal bed frame",
    ]),
    "mattress": ("mattress", "Mattress", [
        "mattress", "memory foam mattress", "queen mattress",
        "king mattress", "firm mattress", "hybrid mattress",
    ]),
    "pillow": ("pillow", "Pillow", [
        "pillow", "memory foam pillow", "cooling pillow",
        "side sleeper pillow", "pillows for sleeping", "down pillow",
    ]),
    "blanket": ("blanket", "Blanket", [
        "blanket", "throw blanket", "weighted blanket",
        "fleece blanket", "wool blanket", "knit blanket",
    ]),
    "comforter": ("comforter", "Comforter", [
        "comforter", "down comforter", "queen comforter",
        "king comforter", "all season comforter", "lightweight comforter",
    ]),
    "air mattress": ("air_mattress", "Air Mattress", [
        "air mattress", "queen air mattress", "twin air mattress",
        "air mattress with built in pump", "camping air mattress", "raised air mattress",
    ]),
    "water bottle": ("water_bottle", "Water Bottle", [
        "water bottle", "insulated water bottle", "stainless steel water bottle",
        "water bottle for gym", "kids water bottle", "leak proof water bottle",
    ]),
    "opener": ("opener", "Can Opener", [
        "can opener", "electric can opener", "manual can opener",
        "smooth edge can opener", "heavy duty can opener", "ergonomic can opener",
    ]),
}


CONFIG_TEMPLATE = """category: {slug}
display_name: {display_name}

catalog_path: data/categories/{slug}/products.jsonl
questions_path: data/categories/{slug}/questions.yaml
index_dir: data/categories/{slug}/index

# Curate exactly three ASINs after scraping. Added to every recommendation candidate pool.
default_slate: []

scrape:
  seed_queries:
{seed_queries_yaml}
  target_products: {target}
  reviews_per_product: 8

"""


def main() -> int:
    configs_dir = REPO_ROOT / "configs"
    data_dir = REPO_ROOT / "data" / "categories"
    configs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    target = 100
    written, skipped = [], []
    for _, (slug, display, queries) in CATEGORY_SPEC.items():
        config_path = configs_dir / f"{slug}.yaml"
        if config_path.exists():
            skipped.append(slug)
            continue
        seed_yaml = "\n".join(f"    - {q!r}" for q in queries)
        content = CONFIG_TEMPLATE.format(
            slug=slug, display_name=display,
            seed_queries_yaml=seed_yaml, target=target,
        )
        config_path.write_text(content)
        (data_dir / slug).mkdir(exist_ok=True)
        written.append(slug)

    print(f"Wrote {len(written)} new configs: {written}")
    print(f"Skipped {len(skipped)} (already existed): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
