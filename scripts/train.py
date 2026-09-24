"""4-bit QLoRA SFT of reviewed wiki questions on a single 16 GB GPU."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/qa/train.jsonl"))
    parser.add_argument("--output", default="outputs/qwen3-4b-wiki-lora")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=2)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA GPU；请检查 Windows Conda 环境中的 PyTorch CUDA 安装")
    rows = [json.loads(line) for line in args.train.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("训练集为空。先人工审核并填写 data/qa/train.jsonl")
    dataset = Dataset.from_list([{
        "prompt": [{"role": "system", "content": "你是东方非想天则 Wiki 助手。只回答有把握的内容；不确定时明确说明。"},
                   {"role": "user", "content": row["question"]}],
        "completion": [{"role": "assistant", "content": row["answer"]}],
    } for row in rows])
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="auto", dtype=torch.bfloat16,
    )
    config = SFTConfig(
        output_dir=args.output, max_length=args.max_length, num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        gradient_checkpointing=True, learning_rate=1e-4, logging_steps=10,
        save_strategy="epoch", report_to="none", bf16=True,
        completion_only_loss=True,
    )
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset,
                         processing_class=tokenizer, peft_config=lora)
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"peak_reserved_mib={torch.cuda.max_memory_reserved() / 1024**2:.1f}")


if __name__ == "__main__":
    main()
