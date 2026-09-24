"""Compute source-page retrieval recall on held-out QA questions."""
import argparse
from pathlib import Path

from check_qa import read_jsonl
from retrieve import search


def evaluate(test, corpus):
    rows = list(read_jsonl(Path(test)))
    if not rows:
        raise SystemExit("测试集为空")
    hits = {1: 0, 3: 0, 5: 0}
    for row in rows:
        results = search(row["question"], corpus, 5)
        found = [item["path"] for _, item in results]
        for k in hits:
            hits[k] += row["source_path"] in found[:k]
    for k, count in hits.items():
        print(f"recall@{k}: {count}/{len(rows)} = {count / len(rows):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/qa/test.jsonl")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    args = parser.parse_args()
    evaluate(args.test, args.corpus)
