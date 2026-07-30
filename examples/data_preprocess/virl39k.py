# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert TIGER-Lab/ViRL39K into the multimodal parquet format used by verl."""

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import datasets
import pandas as pd

DATA_SOURCE = "TIGER-Lab/ViRL39K"
DEFAULT_REVISION = "812ec617dea4bc8a4e751663b88e4ebb7de4d00e"
INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
REQUIRED_COLUMNS = {
    "question",
    "answer",
    "PassRate_32BTrained",
    "PassRate_7BBase",
    "category",
    "source",
    "qid",
    "image",
}


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _question_group(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip()


def normalize_placeholders(question: str, image_count: int) -> tuple[str, bool]:
    """Ensure each image has exactly one ``<image>`` placeholder."""
    placeholder_count = question.count("<image>")
    if placeholder_count == image_count:
        return question, False
    if placeholder_count == 0 and image_count > 0:
        return "\n".join(["<image>"] * image_count + [question]), True
    raise ValueError(f"image/placeholder mismatch: images={image_count}, placeholders={placeholder_count}")


def _allocate_validation_groups(frame: pd.DataFrame, val_size: int, seed: int) -> set[str]:
    """Select validation question groups, stratified by category without leakage."""
    if not 0 < val_size < len(frame):
        raise ValueError(f"val_size must be between 1 and {len(frame) - 1}, got {val_size}")

    grouped: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for group_key, group in frame.groupby("_question_group", sort=False):
        category = str(group.iloc[0]["category"])
        grouped[category].append((group_key, group.index.tolist()))

    category_counts = frame["category"].value_counts().to_dict()
    raw_targets = {category: val_size * count / len(frame) for category, count in category_counts.items()}
    targets = {category: int(target) for category, target in raw_targets.items()}
    remaining = val_size - sum(targets.values())
    ordered_remainders = sorted(targets, key=lambda key: raw_targets[key] - targets[key], reverse=True)
    for category in ordered_remainders[:remaining]:
        targets[category] += 1

    selected: set[str] = set()
    for category, groups in grouped.items():
        target = targets[category]
        current = 0
        ordered = sorted(groups, key=lambda item: _stable_hash(f"{category}\0{item[0]}", seed))
        for group_key, indices in ordered:
            group_size = len(indices)
            if current >= target:
                break
            if current + group_size <= target or target - current >= group_size / 2:
                selected.add(group_key)
                current += group_size
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert_row(row: pd.Series, split: str, image_root: Path) -> tuple[dict, bool]:
    relative_images = [str(path) for path in row["image"]]
    question, repaired = normalize_placeholders(str(row["question"]), len(relative_images))
    image_paths = [str((image_root / path).resolve()) for path in relative_images]
    data = {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": question + INSTRUCTION}],
        "images": image_paths,
        "ability": str(row["category"]),
        "reward_model": {"style": "rule", "ground_truth": str(row["answer"])},
        "extra_info": {
            "split": split,
            "index": int(row["_index"]),
            "qid": str(row["qid"]),
            "answer": str(row["answer"]),
            "question": str(row["question"]),
            "category": str(row["category"]),
            "source": str(row["source"]),
            "passrate_7b_base": float(row["PassRate_7BBase"]),
            "passrate_32b_trained": float(row["PassRate_32BTrained"]),
            "image_count": len(image_paths),
            "placeholder_repaired": repaired,
        },
    }
    return data, repaired


def _write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(rows).to_parquet(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-parquet", type=Path, required=True)
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Directory against which parquet image paths such as images/foo.jpg are resolved.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-size", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--skip-image-validation", action="store_true")
    args = parser.parse_args()

    frame = pd.read_parquet(args.raw_parquet)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"ViRL39K parquet is missing columns: {sorted(missing_columns)}")
    if frame["qid"].duplicated().any():
        raise ValueError("ViRL39K qid values must be unique")
    if not frame["answer"].str.contains(r"\\boxed", regex=True).all():
        raise ValueError("Every ViRL39K answer must contain a boxed final answer")

    frame = frame.copy()
    frame["_index"] = range(len(frame))
    frame["_question_group"] = frame["question"].map(_question_group)
    validation_groups = _allocate_validation_groups(frame, args.val_size, args.seed)
    frame["_split"] = frame["_question_group"].map(lambda value: "val" if value in validation_groups else "train")

    all_relative_images = [str(path) for images in frame["image"] for path in images]
    if not args.skip_image_validation:
        missing_images = [path for path in all_relative_images if not (args.image_root / path).is_file()]
        if missing_images:
            raise FileNotFoundError(f"{len(missing_images)} image references are missing; first={missing_images[0]}")

    converted: dict[str, list[dict]] = {"train": [], "val": []}
    repaired_count = 0
    for _, row in frame.iterrows():
        split = row["_split"]
        item, repaired = _convert_row(row, split, args.image_root)
        converted[split].append(item)
        repaired_count += int(repaired)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for split, rows in converted.items():
        output_path = args.output_dir / f"{split}.parquet"
        _write_parquet(rows, output_path)
        output_paths[split] = output_path
        single_image_path = args.output_dir / f"{split}_single_image.parquet"
        _write_parquet([row for row in rows if len(row["images"]) == 1], single_image_path)
        output_paths[f"{split}_single_image"] = single_image_path

    audit = {
        "dataset": DATA_SOURCE,
        "revision": args.revision,
        "raw_parquet": str(args.raw_parquet.resolve()),
        "raw_parquet_sha256": _sha256(args.raw_parquet),
        "image_root": str(args.image_root.resolve()),
        "rows": len(frame),
        "split_counts": dict(Counter(frame["_split"])),
        "category_counts": dict(Counter(frame["category"])),
        "source_counts": dict(Counter(frame["source"])),
        "image_count_distribution": dict(sorted(Counter(map(len, frame["image"])).items())),
        "image_references": len(all_relative_images),
        "unique_image_references": len(set(all_relative_images)),
        "placeholder_repairs": repaired_count,
        "question_duplicates": int(frame["question"].duplicated().sum()),
        "question_answer_duplicates": int(frame.duplicated(["question", "answer"]).sum()),
        "question_group_overlap": len(
            set(frame.loc[frame["_split"] == "train", "_question_group"])
            & set(frame.loc[frame["_split"] == "val", "_question_group"])
        ),
        "passrate_7b_base": {
            "untested": int((frame["PassRate_7BBase"] == -1).sum()),
            "zero": int((frame["PassRate_7BBase"] == 0).sum()),
            "one": int((frame["PassRate_7BBase"] == 1).sum()),
            "middle": int(((frame["PassRate_7BBase"] > 0) & (frame["PassRate_7BBase"] < 1)).sum()),
        },
        "outputs": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": os.path.getsize(path)}
            for name, path in output_paths.items()
        },
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
