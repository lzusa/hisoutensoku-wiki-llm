"""Generate comparable, locally stored answers for manual factual review."""
import argparse
import json
from pathlib import Path

from check_qa import read_jsonl
from retrieve import search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=Path("data/qa/test.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="auto", dtype=torch.bfloat16,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as dest:
        for row in read_jsonl(args.test):
            found = search(row["question"], args.corpus, args.top_k) if args.top_k else []
            context = "\n\n".join(f"[{item['source_url']}] {item['heading']}\n{item['text'][:1200]}"
                                  for _, item in found)
            system = "你是东方非想天则 Wiki 助手。只回答有把握的内容；不确定时明确说明。"
            if found:
                system += ("只根据下面的 Wiki 片段回答。最多两句话，只给出直接答案；"
                           "不要抄写链接，也不要补充片段未写明的原因、建议或版本信息。"
                           "片段不支持时说明不知道。\n" + context)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   tokenize=True, return_dict=True,
                                                   return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            dest.write(json.dumps({"id": row["id"], "question": row["question"],
                                   "reference_answer": row["answer"], "prediction": answer,
                                   "source_path": row["source_path"],
                                   "retrieved": [item["source_url"] for _, item in found]},
                                  ensure_ascii=False) + "\n")
            print(f"generated {row['id']}")


if __name__ == "__main__":
    main()
