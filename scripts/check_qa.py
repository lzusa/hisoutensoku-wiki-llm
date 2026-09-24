"""Validate manually reviewed QA records and keep page-level splits disjoint."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc


def check(paths):
    ids = set()
    pages = {}
    counts = Counter()
    for path in paths:
        for row in read_jsonl(path):
            for field in ("id", "question", "answer", "source_path", "source_revision"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{path}: 缺少非空字段 {field}")
            if row["id"] in ids:
                raise ValueError(f"重复 id: {row['id']}")
            ids.add(row["id"])
            page = row["source_path"]
            split = path.stem
            if page in pages and pages[page] != split:
                raise ValueError(f"页面跨训练/评测集: {page}")
            pages[page] = split
            counts[split] += 1
    print(dict(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    check(parser.parse_args().files)

