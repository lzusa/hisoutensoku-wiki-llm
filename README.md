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

本机采用原生 Windows + Conda 的 `py310` 环境。已验证该环境的 PyTorch 2.11.0+cu128 能在 RTX 5070 Ti 上执行 CUDA 矩阵运算。该环境里的 Transformers、Datasets、Accelerate 版本较旧，运行训练前还需升级，并安装 TRL、PEFT、bitsandbytes。训练需要下载底座模型，运行时间和显存峰值需在本机实测。

在 PowerShell 中：

```powershell
conda activate py310
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 本机已经装有支持 RTX 5070 Ti 的 PyTorch 2.11.0+cu128
python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data | Out-Null
git clone --filter=blob:none --sparse https://github.com/soku-cn/soku-cn.github.io.git data/upstream
git -C data/upstream sparse-checkout set src
python scripts/build_corpus.py
python scripts/retrieve.py '相对等级如何划分？'
python scripts/download_model.py
```

`download_model.py` 从[Qwen 在 ModelScope 的同名模型](https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)下载到本地 `data/model/`，再运行 `python scripts/train.py --model data/model`。实验时记录实际权重修订版本。

本机原有 `py310` 同时装有其他软件。升级训练依赖后，`pix2tex` 对旧版 `tokenizers` 的精确版本要求与本项目冲突；使用其他软件前请先检查 `python -m pip check`。本项目的[首次实验记录](docs/pilot-2026-09-24.md)列出实测版本和结果。

`data/` 被 Git 忽略。上游仓库含大量图片，使用 sparse checkout 减少下载；如果只运行数据整理和检索，Python 标准库即可。

## 人工问答和训练

在 `data/qa/train.jsonl` 和 `data/qa/test.jsonl` 中每行填入一个经人工核对的 JSON 对象，例如字段结构如下；示例占位符不是训练数据：

```json
{"id":"unique-id","question":"问题","answer":"经核对的答案","source_path":"src/路径.md","source_revision":"完整上游commit SHA"}
```

按**页面**划分训练/测试集，不能让同一页面的相似问答跨集合。至少覆盖常见问法、同义词、容易混淆的角色和不确定问题。自动生成的问答应保留可核对的原文证据，训练前抽样人工复核，过滤错答和重复题。

```bash
python scripts/generate_synthetic_qa.py --output-dir data/qa/generated
python scripts/check_qa.py data/qa/generated/train.jsonl data/qa/dev.jsonl data/qa/test.jsonl
python scripts/eval_retrieval.py --test data/qa/test.jsonl
python scripts/train.py --model data/model --train data/qa/generated/train.jsonl
```

训练默认 `max_length=1024`、batch 1、累积 8、LoRA r=16、2 epoch。若显存不足，先降至 `--max-length 512`；若欠拟合，先改善问答质量再考虑增加轮数。训练只保存本地 adapter 至 `outputs/`，不自动上传。正式实验记录底座版本、上游 SHA、数据量、训练参数、峰值显存和测试结果。

## 评测门槛

1. 固定同一份未见过的页面级测试集，先记录原版模型表现。
2. 检索基线报告 `recall@1/3/5`；人工复核检索片段是否真正支持答案。
3. 对原版模型、检索增强模型、QLoRA 模型使用相同问题；人工按事实正确、拒答是否合理、版本/出处可核查打分。
4. 只有在微调明显改善目标能力且没有明显增加幻觉时，才考虑将 adapter 用于产品。上游变动时重新构建语料并重测。

三组答案可以用以下命令生成，输出保留在本地 `outputs/`；脚本不会自动把模型答案判为正确：

```powershell
python scripts/generate_eval.py --model data/model --output outputs/base.jsonl
python scripts/generate_eval.py --model data/model --top-k 3 --output outputs/rag.jsonl
python scripts/generate_eval.py --model data/model --adapter outputs/pilot-lora --output outputs/lora.jsonl
```

## 导出 GGUF 并在 Ollama 运行

本机的小样本试验模型 `soku-wiki-pilot` 保留作对照。全量训练模型已导出为 `outputs/full-q4_k_m.gguf`（Q4_K_M，约 2.5 GB），并以 `soku-wiki-full` 导入 Ollama。可用 `ollama run soku-wiki-full` 直接试用；Wiki 问答请走下方的本地检索入口。

需要查询 Wiki 时，使用本地检索入口；它会把匹配的 Wiki 片段交给 Ollama，并另行列出检索资料，便于核对：

```powershell
python scripts/ask_ollama.py '练习模式显示 60F，正常应该是多少？'
python scripts/ask_ollama.py --test data/qa/test.jsonl --output outputs/ollama-rag-test.jsonl
```

默认调用本机 `soku-wiki-full`，也可传 `--model` 指定其他已安装的 Ollama 模型。资料不足时应明确说不知道。当前 [8 题复测记录](docs/full-qa-training-2026-09-24.md)显示全量 adapter 单独答对 0/8，加检索后答对 8/8；这是小规模人工复核结果，不能代表整体 Wiki 准确率。

本机全量训练集由 27B 本地模型按页面生成。脚本会校验每条问答的证据句确实出现在所标注的语料片段中，并将开发/测试页面从训练集排除。原文证据匹配不能代替答案含义的人工复核。

Windows 上可用 llama.cpp 的转换脚本和工具复现。以下路径按本机目录编写，其他机器请调整：

```powershell
conda activate py310
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git data/llama.cpp
python -m pip install -e data/llama.cpp/gguf-py
python -m pip install 'sentencepiece>=0.1.98,<0.3.0' 'protobuf>=4.21.0,<5.0.0'
python data/llama.cpp/convert_hf_to_gguf.py data/model --outfile outputs/base-f16.gguf --outtype f16
python data/llama.cpp/convert_lora_to_gguf.py outputs/pilot-lora --base data/model --outfile outputs/pilot-lora.gguf --outtype f16
& D:/AI/llama.cpp/build/bin/Release/llama-export-lora.exe -m outputs/base-f16.gguf --lora outputs/pilot-lora.gguf -o outputs/pilot-merged-f16.gguf
& D:/AI/llama.cpp/build/bin/Release/llama-quantize.exe outputs/pilot-merged-f16.gguf outputs/pilot-q4_k_m.gguf Q4_K_M
Set-Content data/Modelfile.pilot 'FROM D:/workspace/wiki-agent/outputs/pilot-q4_k_m.gguf'
ollama create soku-wiki-pilot -f data/Modelfile.pilot
ollama run soku-wiki-pilot
```

导出时临时升级了 `protobuf` 和 `sentencepiece`；完成后，本机已恢复原来的 `3.20.3` 和 `0.1.97`。共用 Conda 环境的其他项目可能对这两个版本有要求。

这些 GGUF 和 Ollama 模型仅留在本机。上游 Wiki 未声明再发布许可，不要将含训练数据的权重公开上传。

## 路线图

- [x] 上游结构与授权检查、硬件和底座选型
- [x] 本地语料抽取、来源追踪、检索基线和训练脚本
- [ ] 人工编写并审核训练/测试问答
- [x] 在 5070 Ti 上完成一次 Windows QLoRA 小样本试跑并记录显存
- [x] 在首次 8 条测试题上完成三组对照与错误分析
- [ ] 扩大人工审核数据与新测试集，验证结果是否保持
- [ ] 确认授权后再决定是否发布数据或权重

## 参考

- [Qwen3-4B-Instruct-2507 模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Hugging Face PEFT 量化说明](https://huggingface.co/docs/peft/developer_guides/quantization)
- [TRL SFTTrainer 文档](https://huggingface.co/docs/trl/sft_trainer)
- [Unsloth RTX 50 系列说明](https://unsloth.ai/docs/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth)
