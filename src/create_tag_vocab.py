"""Create a tag-to-class-id mapping from Danbooru Parquet files.

All unique tags are used as classes, sorted by descending frequency
(ties broken alphabetically so runs are deterministic).

Example:
    uv run python -m src.create_tag_vocab \
        data/danbooru2025_train.parquet \
        data/danbooru2025_val.parquet \
        --output data/tag_to_id.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def create_tag_vocab(parquet_paths, output_path):
    counts = Counter()

    for parquet_path in parquet_paths:
        frame = pd.read_parquet(parquet_path, columns=["tags"])
        if "tags" not in frame:
            raise ValueError(f"{parquet_path} does not contain a 'tags' column")

        for tags in frame["tags"]:
            if tags is not None:
                counts.update(tags)

    vocabulary = sorted(counts, key=lambda tag: (-counts[tag], tag))
    tag_to_id = {tag: index for index, tag in enumerate(vocabulary)}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as file:
        json.dump(tag_to_id, file, indent=2)
        file.write("\n")

    print(f"Wrote {len(tag_to_id)} tags to {output_path}")
    for tag, index in tag_to_id.items():
        print(f"{index:>2}  {tag:<24} {counts[tag]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/tag_to_id.json"))
    args = parser.parse_args()
    create_tag_vocab(args.parquet, args.output)


if __name__ == "__main__":
    main()
