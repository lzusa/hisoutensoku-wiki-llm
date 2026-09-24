"""Generate source-grounded QA from the local corpus using a local OpenAI API."""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .check_qa import read_jsonl
except ImportError:
    from check_qa import read_jsonl


SYSTEM = """你负责把游戏 Wiki 资料整理为中文训练问答。
严格只根据输入资料出题和回答，不使用常识补全、不猜测、不添加资料没有的结论。
优先覆盖操作步骤、数值、条件、角色/招式机制、限制、例外和术语定义；避免只问页面标题。
每条答案应简洁完整，足以独立回答问题。问题自然、多样，不能把答案直接写进问题。
为每条问答附上能直接支持答案的资料原文短句，必须逐字复制输入内容，不能改写。
同一资料单元内不重复问题。若资料不足以形成有效问答，可以少于要求数量；不要凑数。
输出一个 JSON 对象，格式为 {"qas":[{"chunk_id":"输入中的片段ID","question":"...","answer":"...","evidence":"原文逐字摘录"}]}。"""


def read_corpus(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_page = defaultdict(list)
    for row in rows:
        by_page[row["path"]].append(row)
    units = []
    for page, chunks in sorted(by_page.items()):
        unit, size = [], 0
        for row in chunks:
            block = f"[{row['id']}] {row['heading']}\n{row['text']}"
            if unit and size + len(block) > 7000:
                units.append(unit)
                unit, size = [], 0
            unit.append((row, block))
            size += len(block)
        if unit:
            units.append(unit)
    return rows, units


def call_model(unit, *, url, model, timeout):
    blocks = "\n\n".join(block for _, block in unit)
    prompt = ("请为以下资料单元生成 3 到 8 条高质量、互不重复的问答。每个问题必须能从资料中直接回答；"
              "覆盖资料里的具体信息。chunk_id 必须使用对应方括号里的 ID。\n\n" + blocks)
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.9,
        # This is a fact extraction task; avoid spending most of the response on chain-of-thought.
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 4096,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    request = Request(url.rstrip("/") + "/v1/chat/completions",
                      data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    content = result["choices"][0]["message"]["content"]
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    decoded = json.loads(content)
    if isinstance(decoded, list):
        return decoded
    if isinstance(decoded, dict) and isinstance(decoded.get("qas"), list):
        return decoded["qas"]
    raise ValueError(f"模型输出缺少 qas 数组，返回类型={type(decoded).__name__}，内容={content[:600]!r}")


def has_body(unit):
    for row, _ in unit:
        text = re.sub(r"\A---\s*\r?\n.*?\r?\n---\s*", "", row["text"].strip(), count=1, flags=re.S).strip()
        if text:
            return True
    return False


def normalized(text):
    return "".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/qa/dev.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/qa/test.jsonl"))
    parser.add_argument("--manual-train", type=Path, default=Path("data/qa/train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/qa/generated"))
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--model", default=r"D:\AI\Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf")
    parser.add_argument("--limit", type=int, help="只处理前 N 个资料单元，便于试跑")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows, units = read_corpus(args.corpus)
    if args.limit:
        units = units[:args.limit]
    source_by_id = {row["id"]: row for row in source_rows}
    heldout = {row["source_path"] for file in (args.dev, args.test)
               for row in read_jsonl(file)}
    all_path = args.output_dir / "all.jsonl"
    train_path = args.output_dir / "train.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    checkpoint_path = args.output_dir / "completed_units.json"
    completed = set(json.loads(checkpoint_path.read_text(encoding="utf-8"))) if checkpoint_path.exists() else set()
    seen_questions = {normalized(row["question"]).lower() for row in read_jsonl(args.manual_train)}
    existing = list(read_jsonl(all_path)) if all_path.exists() else []
    for row in existing:
        seen_questions.add(normalized(row["question"]).lower())
    errors = errors_path.open("a", encoding="utf-8")
    try:
        for index, unit in enumerate(units):
            unit_id = str(index)
            if unit_id in completed:
                continue
            if not has_body(unit):
                completed.add(unit_id)
                checkpoint_path.write_text(json.dumps(sorted(completed), ensure_ascii=False), encoding="utf-8")
                print(f"unit {index + 1}/{len(units)}: skipped metadata-only source", flush=True)
                continue
            result = None
            last_error = None
            for attempt in range(args.retries + 1):
                try:
                    result = call_model(unit, url=args.url, model=args.model, timeout=args.timeout)
                    break
                except (HTTPError, URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    last_error = f"unit={index} attempt={attempt + 1}: {type(exc).__name__}: {exc}"
                    if attempt < args.retries:
                        time.sleep(2 + attempt * 3)
            if result is None:
                errors.write(json.dumps({"unit": index, "error": last_error}, ensure_ascii=False) + "\n")
                errors.flush()
                print(f"failed unit {index}: {last_error}", flush=True)
                continue
            accepted, unit_ids = [], {row["id"] for row, _ in unit}
            for item in result:
                if not isinstance(item, dict):
                    continue
                chunk_id = item.get("chunk_id", "")
                question, answer, evidence = (item.get(key, "") for key in ("question", "answer", "evidence"))
                if not all(isinstance(value, str) and value.strip() for value in (chunk_id, question, answer, evidence)):
                    continue
                source = source_by_id.get(chunk_id)
                if not source or chunk_id not in unit_ids:
                    continue
                if len(normalized(evidence)) < 8 or normalized(evidence) not in normalized(source["text"]):
                    continue
                key = normalized(question).lower()
                if key in seen_questions:
                    continue
                seen_questions.add(key)
                accepted.append({"id": f"syn-{len(existing) + len(accepted) + 1:06d}",
                                 "question": question.strip(), "answer": answer.strip(),
                                 "source_path": source["path"], "source_revision": source["revision"],
                                 "source_chunk_id": chunk_id, "evidence": evidence.strip()})
            with all_path.open("a", encoding="utf-8") as dest:
                for row in accepted:
                    dest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    existing.append(row)
            completed.add(unit_id)
            checkpoint_path.write_text(json.dumps(sorted(completed), ensure_ascii=False), encoding="utf-8")
            print(f"unit {index + 1}/{len(units)}: accepted {len(accepted)}, total {len(existing)}", flush=True)
    finally:
        errors.close()
    manual_rows = list(read_jsonl(args.manual_train))
    train_rows = manual_rows + [row for row in existing if row["source_path"] not in heldout]
    train_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train_rows), encoding="utf-8")
    print(f"complete: {len(existing)} total, {len(train_rows)} train, {len(heldout)} held-out pages, {len(units)} units")


if __name__ == "__main__":
    main()
