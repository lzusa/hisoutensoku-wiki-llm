"""Extract provenance-preserving text records from the upstream VuePress wiki."""
import argparse
import json
import re
import subprocess
from pathlib import Path


REPO = "https://github.com/soku-cn/soku-cn.github.io"


def clean_markdown(raw: str) -> str:
    raw = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", raw, flags=re.S)
    raw = re.sub(r"^:::\s*\w*\s*$", "", raw, flags=re.M)
    raw = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"[图片：\1]", raw)
    raw = re.sub(r"\[([^]]+)\]\(([^)]*)\)", r"\1", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def sections(text: str):
    title = ""
    body = []
    for line in text.splitlines():
        if line.startswith("#") and re.match(r"^#{1,4}\s+", line):
            if body and "\n".join(body).strip():
                yield title, "\n".join(body).strip()
            title = line.lstrip("# ").strip()
            body = []
        else:
            body.append(line)
    if body and "\n".join(body).strip():
        yield title, "\n".join(body).strip()


def build(source: Path, output: Path, revision: str):
    root = source / "src"
    if not root.is_dir():
        raise SystemExit(f"找不到 {root}，请先克隆上游仓库")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as dest:
        for path in sorted(root.rglob("*.md")):
            relative = path.relative_to(source).as_posix()
            text = clean_markdown(path.read_text(encoding="utf-8"))
            for index, (heading, content) in enumerate(sections(text)):
                if len(content) < 30:
                    continue
                record = {
                    "id": f"{relative}#{index}",
                    "path": relative,
                    "heading": heading,
                    "text": content,
                    "source_url": f"{REPO}/blob/{revision}/{relative}",
                    "revision": revision,
                }
                dest.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"写入 {count} 段：{output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("data/upstream"))
    parser.add_argument("--output", type=Path, default=Path("data/corpus.jsonl"))
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    build(args.source, args.output, revision)

