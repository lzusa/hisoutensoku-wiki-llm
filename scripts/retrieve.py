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
    docs = [Counter(tokens(row["text"])) for row in rows]
    headings = [Counter(tokens(row["heading"])) for row in rows]
    df = Counter(term for doc, head in zip(docs, headings) for term in doc.keys() | head.keys())
    q = Counter(tokens(query))
    named_terms = [term for term in q if re.fullmatch(r"[a-z_][a-z0-9_]*", term)]
    avg_length = sum(sum(doc.values()) for doc in docs) / max(len(docs), 1)
    scored = []
    for row, doc, head in zip(rows, docs, headings):
        length_norm = 0.25 + 0.75 * sum(doc.values()) / max(avg_length, 1)
        score = 0.0
        for term in q:
            frequency = doc[term] + 3 * head[term]
            if frequency:
                idf = math.log(1 + (len(rows) - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * length_norm)
        if named_terms:
            matched = sum(bool(doc[term] or head[term]) for term in named_terms)
            score *= 0.3 + 0.7 * matched / len(named_terms)
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
