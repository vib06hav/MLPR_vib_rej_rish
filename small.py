import json
from pathlib import Path
from collections import defaultdict

processed = Path(r"C:\Users\vibha\Downloads\archive\processed_dataset")
counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

for meta_file in processed.rglob("metadata.json"):
    parts = meta_file.parts
    domain = [p for p in parts if p in ["day", "night"]][0]
    split = [p for p in parts if p in ["train", "val", "test"]][0]
    category = meta_file.parent.name
    with open(meta_file) as f:
        records = json.load(f)
    counts[domain][split][category] += len(records)

for domain in ["day", "night"]:
    for split in ["train", "val", "test"]:
        print(f"{domain}/{split}: ", dict(counts[domain][split]))