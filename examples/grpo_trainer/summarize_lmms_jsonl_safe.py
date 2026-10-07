#!/usr/bin/env python3
"""Merge and validate eight lmms-eval TP1 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def request_failed(row: dict) -> bool:
    return "LMMS_EVAL_REQUEST_FAILED" in str(row.get("filtered_resps", "")) or (
        bool(row.get("token_counts"))
        and all(item.get("output_tokens", 0) == 0 for item in row["token_counts"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--total", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repairs-root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows: list[dict] = []
    for card in range(8):
        files = list((root / f"shard{card}").rglob(f"*samples_{args.task}.jsonl"))
        if len(files) != 1:
            raise RuntimeError(f"Expected one samples file for shard{card}: {files}")
        rows.extend(json.loads(line) for line in files[0].read_text().split("\n") if line.strip())

    # Optional targeted retries replace every metric row for the retried doc.
    # Some tasks (GPQA) emit multiple rows with no explicit filter field, so
    # doc_id is the only reliable replacement key. Sorting makes later retry
    # attempts win deterministically.
    if args.repairs_root:
        repair_rows_by_doc: dict[int, list[dict]] = {}
        repair_files = sorted(
            Path(args.repairs_root).rglob(f"*samples_{args.task}.jsonl")
        )
        for repair_file in repair_files:
            file_rows = [
                json.loads(line) for line in repair_file.read_text().split("\n") if line.strip()
            ]
            for doc_id in {int(row["doc_id"]) for row in file_rows}:
                repair_rows_by_doc[doc_id] = [
                    row for row in file_rows if int(row["doc_id"]) == doc_id
                ]
        rows = [row for row in rows if int(row["doc_id"]) not in repair_rows_by_doc]
        for repair_rows in repair_rows_by_doc.values():
            rows.extend(repair_rows)

    doc_ids = {int(row["doc_id"]) for row in rows}
    expected = set(range(args.total))
    if doc_ids != expected:
        raise RuntimeError(
            f"Missing or unexpected doc ids: missing={sorted(expected-doc_ids)[:20]} "
            f"extra={sorted(doc_ids-expected)[:20]}"
        )

    failed_ids = sorted({int(row["doc_id"]) for row in rows if request_failed(row)})
    extraction_failure_rate = None
    if args.task == "mmmu_pangu_official":
        from lmms_eval.tasks.mmmu.utils import evaluate_mmmu

        _, metric = evaluate_mmmu([row["mmmu_acc"] for row in rows])
        accuracy = float(metric["acc"])
    elif args.task == "ocrbench":
        accuracy = sum(row["ocrbench_accuracy"]["score"] for row in rows) / args.total
    elif args.task == "geometry3k_freeform":
        accuracy = sum(row["geometry3k_freeform_accuracy"] for row in rows) / args.total
    elif args.task == "mathvista_testmini_rules":
        accuracy = sum(row["mathvista_rules_accuracy"] for row in rows) / args.total
        extraction_failure_rate = (
            sum(row["mathvista_rules_extraction_failure"] for row in rows) / args.total
        )
    elif args.task == "gpqa_diamond_cot_zeroshot":
        grouped = {
            doc_id: [row for row in rows if int(row["doc_id"]) == doc_id]
            for doc_id in range(args.total)
        }
        if not all(len(group) == 2 for group in grouped.values()):
            raise RuntimeError("GPQA must emit strict and flexible rows for every document")
        accuracy = sum(group[-1]["exact_match"] for group in grouped.values()) / args.total
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    combined = root / "combined_samples.jsonl"
    with combined.open("w") as handle:
        for row in sorted(rows, key=lambda item: (int(item["doc_id"]), str(item.get("filter", "")))):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "task": args.task,
        "samples": args.total,
        "accuracy": accuracy,
        "extraction_failure_rate": extraction_failure_rate,
        "failed_doc_ids": failed_ids,
        "status": "needs_repair" if failed_ids else "complete",
        "backend": "vLLM BF16 TP1x8",
        "temperature": 0,
        "seed": 42,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
