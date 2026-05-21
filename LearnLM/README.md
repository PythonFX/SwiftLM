<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-21 | Updated: 2026-05-21 -->

# LearnLM

## Purpose
从零开始学习 LLM 推理的学习项目。逐步实现一个最小化的推理引擎，最终目标是在 16GB M1 Pro 上高效运行 Qwen3.6-35B-A3B。

## 学习路线图

### Step 01: Hello MLX ✅
**目标**: 理解 MLX 张量、GPU 计算、矩阵乘法
- MLXArray 创建 (标量/向量/矩阵/张量)
- matmul — LLM 最核心的计算
- 惰性求值 (lazy evaluation) 与 eval()
- Apple Silicon 统一内存优势

### Step 02: Tokenizer (下一步)
**目标**: 把文字变成数字，理解 tokenization
- 加载 HuggingFace tokenizer
- encode: "你好世界" → [123, 4567, 890, ...]
- decode: [123, 4567, ...] → "你好世界"
- BPE 算法原理

### Step 03: Load Weights
**目标**: 理解模型文件结构和权重加载
- 读取 config.json — 理解模型架构参数
- 读取 safetensors — 二进制权重文件格式
- 将权重映射到 MLXArray

### Step 04: Embedding
**目标**: token ID → 语义向量
- Embedding lookup — token ID 到向量
- 理解 hidden_size, vocab_size 的含义

### Step 05: Attention
**目标**: 实现自注意力机制 (Self-Attention)
- Q/K/V 投影
- RoPE 旋转位置编码
- Scaled dot-product attention
- Multi-head attention

### Step 06: Transformer
**目标**: 组装完整的 Transformer 前向传播
- 堆叠所有层 (embedding → N × attention + FFN → output)
- Layer Normalization / RMSNorm
- FFN (Feed-Forward Network)
- Residual connections

### Step 07: Sampling
**目标**: 从概率分布中采样下一个 token
- Softmax → 概率分布
- Temperature (温度)
- Top-k / Top-p 采样
- Greedy decoding

### Step 08: KV Cache
**目标**: 加速自回归生成
- 理解 KV Cache 为什么能加速
- 实现 KV Cache 的存取
- 对比有/无 KV Cache 的性能差异

### Step 09: Generate
**目标**: 完整的文字生成
- 自回归生成循环: token → model → next token
- 停止条件 (EOS token, max_tokens)
- 实现一个可交互的命令行聊天

## 运行方式

```bash
cd LearnLM

# 构建 metallib (首次运行需要 cmake)
export PATH="$PATH:/opt/homebrew/bin"
MLX_SRC=".build/checkouts/mlx-swift/Source/Cmlx/mlx"
mkdir -p .build/metallib_build && cd .build/metallib_build
cmake "../../$MLX_SRC" -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_METAL_JIT=OFF -DCMAKE_BUILD_TYPE=Release
make mlx-metallib -j$(sysctl -n hw.ncpu)
cp lib/mlx.metallib ../debug/default.metallib
cp lib/mlx.metallib ../release/default.metallib
cd ../..

# 运行
swift run Step01_HelloMLX
```

## For AI Agents

### Working In This Directory
- 这是一个独立于 SwiftLM 主项目的 SPM 包
- 必须在 LearnLM/ 目录内运行 `swift build` / `swift run`
- 依赖官方 Apple mlx-swift (非 SharpAI fork)
- 每个 Step 是独立的可执行目标，互不依赖

### Common Patterns
- 每个 Step 的 main.swift 都是自包含的，可以直接运行
- 注释用中文 + 英文术语，方便理解
- 新 Step 需要在 Package.swift 中添加对应的 product 和 target

<!-- MANUAL: Custom project notes can be added below -->
