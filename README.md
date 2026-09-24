# 非想天则 Wiki 模型实验

针对[非想天则指南](https://github.com/soku-cn/soku-cn.github.io)的可复现实验：先建立可追溯的检索基线，再用人工核对的中文问答对 Qwen3-4B-Instruct-2507 做 4-bit QLoRA 监督微调。

## 技术选择

| 项目 | 选择 | 理由 |
| --- | --- | --- |
| 显卡 | RTX 5070 Ti 16 GB | 本项目在此卡上规划训练；实际峰值显存尚待实测 |
| 底座 | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 中文指令能力、Apache-2.0、4B 参数量 |
| 微调 | [QLoRA](https://huggingface.co/docs/peft/developer_guides/quantization) + [TRL SFT](https://huggingface.co/docs/trl/sft_trainer) | NF4 4-bit、双重量化、LoRA；只对人工审核答案计算 loss |
| 对照组 | `scripts/retrieve.py` 的可追溯词法检索 | Wiki 经常更新，检索方便核对版本和出处 |

上游目前约 198 篇 Markdown、约 0.5 MB。这样的文本量不适合直接把整站内容继续预训练后期待准确记忆。微调主要用于术语、答题方式与稳定知识；需要最新细节、精确数字和出处时，应给模型提供检索到的原文。不要把微调模型当成唯一事实来源。

## 数据与授权

上游 GitHub 仓库当前没有声明许可证。本仓库的 **代码** 采用 MIT 许可证；这不授予复制、再发布 Wiki 文本、图片、生成的问答或模型权重的权利。本仓库不提交上述内容。数据由使用者本地从上游获取，发布数据集或训练产物前需要取得相应授权并检查模型许可证。

每个本地语料片段保留上游 commit SHA、路径、标题和 GitHub 来源 URL，方便检查更新及纠错。Wiki 中的角色名、招式、补丁和社区意见可能随版本变化，人工问答应标注来源版本。

## 快速开始

推荐在 WSL2 Ubuntu 内运行训练，使用与 RTX 50 系列兼容的 NVIDIA 驱动、CUDA PyTorch。先用 `nvidia-smi` 和 Python 确认 GPU 可见。训练需要下载底座模型，运行时间和显存峰值需在本机实测。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# 按 PyTorch 官方说明安装适合本机 CUDA/Blackwell 的 PyTorch 后，再装项目依赖
pip install -r requirements.txt
git clone --filter=blob:none --sparse https://github.com/soku-cn/soku-cn.github.io.git data/upstream
git -C data/upstream sparse-checkout set src
python scripts/build_corpus.py
python scripts/retrieve.py '相对等级如何划分？'
```

`data/` 被 Git 忽略。上游仓库含大量图片，使用 sparse checkout 减少下载；如果只运行数据整理和检索，Python 标准库即可。

## 人工问答和训练

在 `data/qa/train.jsonl` 和 `data/qa/test.jsonl` 中每行填入一个经人工核对的 JSON 对象，例如字段结构如下；示例占位符不是训练数据：

```json
{"id":"unique-id","question":"问题","answer":"经核对的答案","source_path":"src/路径.md","source_revision":"完整上游commit SHA"}
```

按**页面**划分训练/测试集，不能让同一页面的相似问答跨集合。至少覆盖常见问法、同义词、容易混淆的角色和不确定问题；不要根据页面标题机械生成答案，也不要把未核对的模型生成内容直接投入训练。

```bash
python scripts/check_qa.py data/qa/train.jsonl data/qa/test.jsonl
python scripts/eval_retrieval.py --test data/qa/test.jsonl
python scripts/train.py --train data/qa/train.jsonl
```

训练默认 `max_length=1024`、batch 1、累积 8、LoRA r=16、2 epoch。若显存不足，先降至 `--max-length 512`；若欠拟合，先改善问答质量再考虑增加轮数。训练只保存本地 adapter 至 `outputs/`，不自动上传。正式实验记录底座版本、上游 SHA、数据量、训练参数、峰值显存和测试结果。

## 评测门槛

1. 固定同一份未见过的页面级测试集，先记录原版模型表现。
2. 检索基线报告 `recall@1/3/5`；人工复核检索片段是否真正支持答案。
3. 对原版模型、检索增强模型、QLoRA 模型使用相同问题；人工按事实正确、拒答是否合理、版本/出处可核查打分。
4. 只有在微调明显改善目标能力且没有明显增加幻觉时，才考虑将 adapter 用于产品。上游变动时重新构建语料并重测。

## 路线图

- [x] 上游结构与授权检查、硬件和底座选型
- [x] 本地语料抽取、来源追踪、检索基线和训练脚本
- [ ] 人工编写并审核训练/测试问答
- [ ] 在 5070 Ti 上实测显存、吞吐与训练稳定性
- [ ] 三组模型对照评测与错误分析
- [ ] 确认授权后再决定是否发布数据或权重

## 参考

- [Qwen3-4B-Instruct-2507 模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Hugging Face PEFT 量化说明](https://huggingface.co/docs/peft/developer_guides/quantization)
- [TRL SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer)
- [Unsloth RTX 50 系列说明](https://unsloth.ai/docs/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth)

