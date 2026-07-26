# 加载 DFlash 检查点训练 DSpark 草稿模型技术报告

## 1. 背景

### 1.1 推测解码与 DSpark

推测解码（Speculative Decoding）是一种无损加速大语言模型推理的技术：用一个更小的草稿模型（draft model）预测多个 token，由验证模型（verifier）在单次前向传播中验证，从而在不牺牲输出质量的前提下提升吞吐。Speculators 是 vLLM 生态下的统一训练库，支持 Eagle-3、P-EAGLE、DFlash、DSpark 等多种草稿模型算法。

DSpark 是 DFlash 的扩展算法，在 DFlash 的锚点块草稿（anchored-block drafting）基础上引入了 **Markov 序列头** 和 **置信度头（confidence head）**，通过低秩分解的 Markov 偏置和逐位置置信度预测，提升多 token 草稿的准确率。DSpark 的核心特性包括：

- **块级并行草稿**：以锚点为起点，一次前向传播生成 `block_size`（默认 7）个候选 token
- **Markov 序列建模**：通过低秩分解矩阵（`markov_rank=256`）建模块内 token 的序列依赖
- **置信度校准**：逐位置预测草稿 token 的接受概率，辅助推理时的 rejection sampling
- **滑动窗口注意力（SWA）**：默认使用 `sliding_window=2048` 的滑动窗口注意力，降低 KV cache 开销

### 1.2 Qwen3-8B 推测解码需求

Qwen3-8B 是广泛使用的开源大语言模型，在数学推理、代码生成等任务上有良好表现。在实际部署中，Qwen3-8B 的推理吞吐受限于自回归解码的逐 token 串行性。通过训练 DSpark 草稿模型并配合 vLLM 推测解码，可以显著提升推理吞吐。

## 2. 挑战

### 2.1 开源 DSpark 权重稀缺

截至 2026 年 7 月，HuggingFace 上仅有 DeepSeek 官方发布的 `deepseek-ai/dspark_qwen3_8b_block7` 一个 Qwen3-8B DSpark 检查点。该检查点使用大规模数据训练，accept_len≈6.92（GSM8K 数据集），但用户无法针对特定领域数据微调或从头训练自己的 DSpark 模型。

### 2.2 从零训练代价高

DSpark 草稿模型的 decoder body（5 层 Transformer，4096 hidden size）从随机初始化训练需要大量数据和多个 epoch 才能收敛。在我们的实验中，使用 5K ShareGPT 样本从零训练 5 个 epoch：

| Epoch | accept_rate | accept_len |
|:---:|:---:|:---:|
| 4（最佳）| 0.265 | 1.765 |

accept_len 仅 1.765，推理吞吐 277.4 tok/s（dense 的 1.25×），加速效果有限。要达到 DeepSeek 官方水平（accept_len≈6.92）需要数十倍的数据和训练量，对大多数用户不可行。

### 2.3 DFlash 与 DSpark 的架构差异

DFlash 检查点（如 `z-lab/Qwen3-8B-DFlash-b16`）可从 HuggingFace 获取，但其架构与 DSpark 存在差异：

- **Attention 类型**：DFlash ckpt 使用 `full_attention`，DSpark 默认使用 `sliding_attention`（SWA, window=2048）
- **`sample_from_anchor`**：DFlash 默认 `False`（anchor 为 bonus token，block_size-1 个预测），DSpark 默认 `True`（anchor 参与预测，block_size 个预测）
- **额外组件**：DSpark 有 Markov head 和 confidence head，DFlash 没有
- **层索引约定**：DFlash 使用 `target_layer_ids`（index 0 = embedding output），DSpark/speculators 使用 `aux_hidden_state_layer_ids`（直接索引 transformer 层 id），两者差 1

直接加载 DFlash ckpt 到 DSpark 需要正确处理这些差异。

## 3. 解决方案：加载 DFlash 检查点初始化 DSpark

### 3.1 核心思路

利用 DSpark 继承自 DFlash 的架构关系，将已训练的 DFlash backbone 权重迁移到 DSpark 模型中，作为 DSpark 训练的初始化点。Markov head 和 confidence head 从随机初始化开始训练，backbone 在 DFlash 权重基础上进行 fine-tuning。

### 3.2 `load_dflash_backbone` 实现

在 `src/speculators/models/dspark/core.py` 中实现 `load_dflash_backbone` 方法：

```python
_DFLASH_BACKBONE_SKIP_KEYS: ClassVar[frozenset[str]] = frozenset({
    "embed_tokens.weight",      # verifier 共享，由 load_verifier_weights 设置
    "lm_head.weight",           # verifier 共享，标准 next-token 预测
    "verifier_lm_head.weight",  # 冻结的 verifier 权重
    "verifier_norm.weight",     # 冻结的 verifier 权重
    "t2d",                      # 词表映射 buffer
    "d2t",                      # 词表映射 buffer
})
```

**迁移策略**：
- **迁移的 backbone 权重**（60 个 tensor）：decoder layers（q/k/v/o_proj, MLP, norms）、fc 投影、hidden_norm、norm
- **跳过的权重**：embed_tokens 和 lm_head 从 verifier 加载（标准 next-token 预测），verifier_lm_head 和 verifier_norm 冻结，t2d/d2d 词表映射保留 DSpark 自身值
- **随机初始化的 DSpark 专属组件**：Markov head（`markov_w1`: [151936, 256], `markov_w2`: [151936, 256]）、confidence head（`proj`: [1, 4352]）

### 3.3 CLI 接口

通过 `--init-from-dflash` 和 `--draft-config` CLI 参数控制：

```bash
torchrun --standalone --nproc_per_node 2 scripts/train.py \
    --speculator-type dspark \
    --init-from-dflash /path/to/dflash_ckpt \  # 加载 backbone 权重
    --draft-config /path/to/dflash_ckpt \      # 使用 DFlash 的 config 结构
    --block-size 7 --max-anchors 512 \
    --target-layer-ids 2 10 18 26 34 \
    ...
```

- `--init-from-dflash`：触发 `load_dflash_backbone()`，迁移 backbone 权重
- `--draft-config`：从 DFlash ckpt 读取 config（layer_types, hidden_size 等）
- `--target-layer-ids`：显式指定 speculators 格式的层 ID（已 +1 转换后的值）

## 4. 遇到的问题及解决方法

### 4.1 Attention 类型不匹配

**问题**：DFlash ckpt 使用 `full_attention`，DSpark 默认使用 `sliding_attention`（SWA, window=2048）。直接使用 `--draft-config` 会继承 DFlash 的 full attention config，导致推理时 SWA 的 KV cache 优势无法体现，throughput 差异显著（full: 48.8 tok/s vs SWA: 108.5 tok/s，2.2× 差距）。

**解决方法**：创建 DFlash ckpt 的 SWA 配置副本——保留原始权重（model.safetensors 通过 symlink 共享），修改 config.json 将 `layer_types` 设为 `["sliding_attention", ...]`，`sliding_window` 设为 `2048`，`use_sliding_window` 设为 `True`。权重无需修改，因为 attention 类型的差异仅在计算图层面，不影响权重 tensor 的名称和 shape。

### 4.2 `sample_from_anchor` 差异

**问题**：DFlash 默认 `sample_from_anchor=False`（anchor 位为 bonus token，slot 0 不参与训练），DSpark 默认 `True`（anchor 位参与预测，block_size 个 speculative tokens）。两者 target 对齐差 1 位。

**分析**：该差异**不影响 backbone 权重迁移**，因为：
1. decoder 权重是位置无关的（同一组权重处理所有 block 位置）
2. `lm_head` 从 verifier 加载（标准 next-token 预测），不依赖 DFlash 的 target 对齐
3. attention mask 不受 `sample_from_anchor` 影响
4. slot 0 的行为差异由 DSpark fine-tuning 补偿——decoder 在 slot 1+ 已学会的预测能力可泛化到 slot 0

### 4.3 层索引约定差异

**问题**：DFlash（z-lab）格式使用 `target_layer_ids`（index 0 = embedding output），speculators 使用 `aux_hidden_state_layer_ids`（直接索引 transformer 层 id），两者差 1。

**解决方法**：Speculators 库的 `DFlashConverter` 已内置映射逻辑：
```python
# z-lab reads hidden_states[layer_id + 1] (index 0 is the embedding
# output) while speculators uses the layer id directly.
aux_hidden_state_layer_ids = [i + 1 for i in target_layer_ids]
```

训练时通过 `--target-layer-ids 2 10 18 26 34` 显式传入 speculators 格式的值（已 +1 转换）。vLLM 端的 `launch_vllm.py` 也需传入相同的 `--target-layer-ids` 以保证隐状态提取层一致。

### 4.4 vLLM 版本兼容性

**问题**：vLLM 0.22.1 不支持 `dspark` speculator method（仅支持 `dflash`）。DSpark 推理需要 vLLM 0.25.0 + vllm-ascend 最新版。

**解决方法**：
1. 升级 vLLM 到 0.25.0（`VLLM_TARGET_DEVICE=empty` 纯 Python 安装）
2. 从 GitHub 拉取 vllm-ascend 最新代码，覆盖已安装的 vllm_ascend Python 包
3. 修补两处兼容性问题：
   - `patch_dp_device_ids.py`：用 `getattr` 守卫跳过不存在的 `get_physical_gpu_ids_for_local_dp_rank`
   - `hunyuan_vl_processor_compat.py`：用 `try/except` 跳过不存在的 `HunYuanVLProcessor` import

### 4.5 `--save-best` 脚本错误

**问题**：训练脚本的 `--save-best` 参数因 shell 换行问题被解释为独立命令，导致 `save_best=False`，epoch 0 的 checkpoint 被后续 epoch 覆盖。

**解决方法**：修正脚本换行，确保 `--save-best` 作为 `train.py` 的参数传递。

## 5. 测试结果

### 5.1 环境与版本

| 组件 | 版本 |
|---|---|
| 硬件 | 4 × Ascend 910B3 NPU (64GB HBM each) |
| OS | Linux (aarch64) |
| Python | 3.12.13 |
| vLLM | 0.25.0+empty |
| vllm-ascend | 最新 (commit 69d1d6f) |
| speculators | 0.7.0.dev67 |
| transformers | 5.5.4 |
| torch | 2.10.0+cpu (torch_npu 2.10.0) |
| 训练布局 | 2 NPU vLLM (DP=2) + 2 NPU DDP |
| 推理布局 | 1 NPU, PIECEWISE cudagraph |

### 5.2 训练配置

| 参数 | 值 |
|---|---|
| Verifier | Qwen3-8B |
| DFlash ckpt | `/home/model/dflash_qwen3_8b_block7_speculators` (z-lab 转换) |
| Attention | sliding_attention (SWA, window=2048) |
| 数据集 | ShareGPT, 5000 samples |
| Epochs | 5 |
| LR | 3e-4 |
| seq_length | 4096 |
| block_size | 7 |
| max_anchors | 512 |
| target_layer_ids | 2 10 18 26 34 |
| markov_rank | 256 |
| loss_fn | {"ce": 0.1, "tv": 0.9} |

### 5.3 训练结果

#### 5.3.1 DFlash seeding 对训练收敛的影响

| Epoch | from-dflash (SWA) | from-scratch (SWA) |
|:---:|:---:|:---:|
| | accept_rate / accept_len | accept_rate / accept_len |
| 0 | 0.362 / 1.967 | — |
| 1 | 0.423 / 2.246 | — |
| 2 | 0.470 / 2.486 | — |
| 3 | 0.499 / 2.634 | — |
| **4** | **0.524 / 2.775** | **0.265 / 1.765** |

DFlash backbone 初始化使 **epoch 0 的 accept_rate (0.362) 已超过 from-scratch 最终 epoch 4 (0.265)**，说明 DFlash 预训练权重提供了显著的起步优势。

#### 5.3.2 训练吞吐对比（DP2 vs TP2 vLLM 布局）

| 指标 | DP2 (2×TP1) | TP2 (1×TP2) |
|---|---|---|
| 训练 tokens/s | 2828 | 2808 |
| vLLM 隐状态生成吞吐 | 6380 tok/s | 6164 tok/s |
| 验证 epoch 时长 | 34.8s | 61.4s |
| 单 epoch 总时长 | 18.18 min | 18.75 min |

DP2 在验证阶段快 1.76×（独立 server 分流），训练稳态基本持平。

### 5.4 推理测试结果

#### 5.4.1 GSM8K 数据集 benchmark（50 prompts, max_tokens=512, temperature=0）

| 模型 | Throughput (tok/s) | vs Dense | accept_len | Avg accept% |
|---|---|---|---|---|
| Dense Qwen3-8B | 221.0 | 1.0× | — | — |
| DeepSeek 官方 DSpark | 1009.0 | 4.56× | 6.92 | 84.5% |
| **from-dflash SWA** | **335.8** | **1.52×** | **2.43** | 20.4% |
| from-scratch SWA | 277.4 | 1.25× | 2.05 | 13.5% |

#### 5.4.2 DFlash seeding 加速效果（公平对比，两者均 SWA）

| 指标 | from-scratch | from-dflash | 提升 |
|---|---|---|---|
| 推理 throughput | 277.4 tok/s | 335.8 tok/s | **+21.0%** |
| 推理 accept_len | 2.05 | 2.43 | **+18.5%** |
| 训练 accept_len | 1.765 | 2.775 | +57.2% |
| vs Dense 加速比 | 1.25× | 1.52× | — |

#### 5.4.3 训练 vs 推理 acceptance 对比

| 模型 | 训练 accept_len (ShareGPT) | 推理 accept_len (GSM8K) | 比值 |
|---|---|---|---|
| DeepSeek 官方 | ~6.75 | 6.92 | 102% |
| from-dflash SWA | 2.775 | 2.43 | 88% |
| from-scratch SWA | 1.765 | 2.05 | 116% |

训练与推理 acceptance 基本对齐（±20% 内）。from-dflash 训练值偏高（DFlash backbone 对话领域偏置 + 训练验证时 verifier 提供完整 hidden states），from-scratch 训练值偏低（欠拟合保守估计，GSM8K 可预测性补偿）。

#### 5.4.4 Attention 类型对推理吞吐的影响

| 模型 | Attention | Throughput (tok/s) | 差异 |
|---|---|---|---|
| from-dflash | full | 48.8 | 基线 |
| from-dflash | SWA (window=2048) | 108.5 | 2.2× 更快 |

SWA 的固定 KV cache 大幅降低 attention 计算开销，推理吞吐提升 2.2×。

### 5.5 DeepSeek 官方模型对比

| 维度 | DeepSeek 官方 | 我们 (from-dflash SWA) |
|---|---|---|
| 训练数据 | 大规模 | 5K ShareGPT |
| accept_len (GSM8K) | 6.92 | 2.43 |
| Throughput | 1009.0 tok/s | 335.8 tok/s |
| vs Dense | 4.56× | 1.52× |
| architectures | Qwen3DSparkModel (vLLM 原生) | DSparkDraftModel (speculators) |
| Attention | full_attention | sliding_attention |

差距主要来自训练数据规模。DFlash seeding 方案使 5K 样本训练达到 accept_len=2.43，虽远低于 DeepSeek 官方 (6.92)，但相比 from-scratch (2.05) 已有 +18.5% 提升，且推理吞吐超过 dense (1.52×)。

## 6. 总结

### 6.1 方案有效性

加载 DFlash 检查点初始化 DSpark 草稿模型是**有效且实用**的方案：
- **训练加速**：epoch 0 即超越 from-scratch 最终 epoch，收敛速度显著提升
- **推理加速**：GSM8K 上 1.52× dense 吞吐（from-scratch 仅 1.25×），DFlash seeding 贡献 +21% throughput
- **权重兼容**：60 个 backbone key 名称和 shape 完全一致，迁移零损耗
- **实现简洁**：`--init-from-dflash` 一行参数即可启用

### 6.2 关键经验

1. **Attention 类型必须对齐**：DFlash ckpt 的 full attention 和 DSpark 默认的 SWA 会导致 2.2× 吞吐差异，公平对比和实际部署都应使用 SWA
2. **训练与推理 acceptance 可对齐**：使用相同数据类型（如 GSM8K 数学推理）时，训练 accept_len 和推理 accept_len 在 ±20% 内一致
3. **Prompt 难度影响显著**：创意写作类 prompt 的 accept_len (1.76) 远低于数学推理 (2.43)，benchmark 必须使用标准化数据集
4. **vLLM + vllm-ascend 版本需匹配**：DSpark 推理依赖 vLLM 0.25.0 + vllm-ascend 最新版，版本不匹配会导致 speculator method 不被识别

### 6.3 后续方向

- **扩大训练数据**：从 5K 扩展到 50K+ 样本，追平 DeepSeek 官方 accept_len 水平
- **多领域数据混合**：在 ShareGPT 基础上增加数学（GSM8K）、代码（HumanEval）数据，提升跨领域 acceptance
- **speculators → vLLM-native 格式转换**：编写转换脚本，使 speculators 格式 checkpoint 可转为 DeepSeek/vLLM-native 格式（`Qwen3DSparkModel`），无需依赖 speculators 库即可部署
- **NPU 推测解码优化**：当前 spec pipeline per-step 开销偏高（verifier 的 ~9×），需 vllm-ascend 优化 proposer/rejection 的 kernel 融合和 cudagraph 覆盖
