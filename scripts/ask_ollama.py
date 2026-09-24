"""Answer wiki questions with local retrieval and an Ollama model."""
import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .retrieve import search
except ImportError:
    from retrieve import search


SYSTEM = (
    "你是东方非想天则 Wiki 助手。仅根据用户消息中的 Wiki 资料回答。"
    "资料不支持时说‘在检索到的 Wiki 内容中没有找到答案’，不要猜测。"
    "回答简短、具体，不要编造路径、数字或出处。"
)


def build_prompt(question, found, max_chars=1200):
    if not found:
        return f"问题：{question}\n\n没有检索到相关 Wiki 资料。"
    excerpts = []
    for index, (_, row) in enumerate(found, 1):
        excerpts.append(f"[{index}] {row['heading']}\n{row['text'][:max_chars]}")
    return "Wiki 资料：\n" + "\n\n".join(excerpts) + f"\n\n问题：{question}"


def answer(question, *, model, corpus, top_k, url, num_predict=160):
    found = search(question, corpus, top_k)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_prompt(question, found)},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    request = Request(
        url.rstrip("/") + "/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except (HTTPError, URLError) as error:
        raise SystemExit(f"Ollama 请求失败：{error}") from error
    sources = [{"heading": row["heading"], "path": row["path"],
                "url": row["source_url"], "score": round(score, 3)}
               for score, row in found]
    return {"question": question, "answer": result["message"]["content"].strip(),
            "sources": sources}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--model", default="soku-wiki-full")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--test", type=Path, help="JSONL 问答集；逐题生成结果")
    parser.add_argument("--output", type=Path, help="将评测结果保存为 JSONL")
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k 不能为负数")
    if args.test and not args.output:
        parser.error("使用 --test 时须指定 --output")
    if not args.test and not args.question:
        parser.error("提供问题或 --test")
    if args.test:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as dest:
            for line in args.test.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                result = answer(row["question"], model=args.model, corpus=args.corpus,
                                top_k=args.top_k, url=args.url)
                result.update(id=row["id"], reference_answer=row["answer"],
                              source_path=row["source_path"])
                dest.write(json.dumps(result, ensure_ascii=False) + "\n")
                print(f"generated {row['id']}")
    else:
        result = answer(args.question, model=args.model, corpus=args.corpus,
                        top_k=args.top_k, url=args.url)
        print(result["answer"])
        for source in result["sources"]:
            print(f"检索资料：{source['url']}")


if __name__ == "__main__":
    main()
