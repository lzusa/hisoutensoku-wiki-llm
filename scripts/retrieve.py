"""Dependency-free lexical retrieval baseline with traceable sources."""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def tokens(text):
    # Chinese bigrams plus Latin words and numbers; sufficient for a transparent baseline.
    compact = re.findall(r"[\u3400-\u9fff]|[a-zA-Z0-9_]+", text.lower())
    return [compact[i] + compact[i + 1] for i in range(len(compact) - 1)
            if re.fullmatch(r"[\u3400-\u9fff]", compact[i]) and re.fullmatch(r"[\u3400-\u9fff]", compact[i + 1])] + compact


def search(query, corpus, k=5):
    rows = [json.loads(line) for line in Path(corpus).read_text(encoding="utf-8").splitlines() if line.strip()]
    docs = [Counter(tokens(row["heading"] + " " + row["text"])) for row in rows]
    df = Counter(term for doc in docs for term in doc)
    q = Counter(tokens(query))
    scored = []
    for row, doc in zip(rows, docs):
        score = sum(min(q[t], doc[t]) * (math.log((len(rows) + 1) / (df[t] + 1)) + 1)
                    for t in q if doc[t])
        if score:
            scored.append((score, row))
    return sorted(scored, key=lambda x: x[0], reverse=True)[:k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    for score, row in search(args.question, args.corpus, args.top_k):
        print(f"{score:.2f} {row['heading']} {row['source_url']}\n{row['text'][:500]}\n")

