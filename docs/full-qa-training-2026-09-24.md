# 全量合成问答与 QLoRA 复测（2026-09-24）

## 数据生成

- 输入语料：本地 Wiki 快照 `b03bd40710774725f8053780c547514d83fd226e`，126 页、428 个片段，约 230k 字符。
- 本地生成模型：llama.cpp `b10721` 服务上的 `Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf`，经 `http://localhost:8080/v1/chat/completions` 调用；对事实整理任务关闭思考模式，以 JSON 输出问答和逐字证据句。
- 全量生成 610 条；108 个含有正文的页面有生成结果。无正文、只有 VuePress frontmatter 的片段不产问答。
- 每条保留 Wiki 路径、revision、chunk id 和证据句。自动校验 610/610 证据句都能在对应原文中匹配，问题文本无重复。
- 29 条来自保留开发/测试页面的问答只进入 `all.jsonl`，不进训练。`train.jsonl` 合并 581 条合成问答与 20 条原有人工问答，共 601 条、100 个来源页。`data/qa/dev.jsonl` 和 `data/qa/test.jsonl` 各 8 条，页面级隔离检查通过。
- 本地数据文件和 checkpoint 位于 Git 忽略的 `data/`，不上传仓库。

## 训练

在 RTX 5070 Ti 16 GB 上原生 Windows + Conda `py310` 完成：

```powershell
python scripts/train.py --model data/model --train data/qa/generated/train.jsonl --epochs 2 --max-length 1024 --output outputs/full-lora
```

152 个优化步骤，约 8 分 23 秒，训练 loss 2.321，峰值预留显存 4854 MiB。LoRA 已合并并量化为本地 `outputs/full-q4_k_m.gguf`，在 Ollama 注册为 `soku-wiki-full`。

## 未见题评测

8 道测试题沿用此前固定的 FAQ 测试集，参考答案经人工核对：

| 模式 | 关键事实正确数 | 页面 recall@3 |
| --- | ---: | ---: |
| 全量 adapter，无检索 | 0/8 | — |
| 全量 adapter + 本地 Wiki 检索 | 8/8 | 8/8 |

裸 adapter 仍会把 Wiki 故障原因替换成常见猜测，或编造设置项；训练 loss 降低没有带来独立知识问答改善。加检索的 8 条回答都包含参考答案中的核心事实，来源 URL 由程序列出。样本数较少，不代表整个 Wiki 的准确率；目前应通过检索入口使用该模型，不把裸 adapter 当作 Wiki 知识库。
