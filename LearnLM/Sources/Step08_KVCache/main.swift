// Step08: KV Cache — 避免重复计算的关键优化
//
// 学习目标:
//   1. 朴素推理的问题: 每个 token 都重算所有历史
//   2. KV Cache 的原理: 缓存已计算的 K 和 V
//   3. Pre-fill vs Decode 两个阶段
//   4. 内存开销分析
//   5. 速度对比: 有/无 KV Cache 的计算量
//
// 运行: cd LearnLM && swift run Step08_KVCache

import Foundation
import MLX

print("=== Step08: KV Cache ===")
print()

// ═══════════════════════════════════════════════════════
// 1. 朴素推理的问题 — 巨大的浪费
// ═══════════════════════════════════════════════════════
//
// 生成 "猫追狗在院子里" 的过程:
//
//   Step 1: 输入 "猫" → 预测 "追"
//     Q₁, K₁, V₁ = Attention("猫")           ← 计算 1 组 K,V
//
//   Step 2: 输入 "猫追" → 预测 "狗"
//     Q₂, K₂, V₂ = Attention("猫", "追")      ← 重新计算 "猫" 的 K,V!
//                                             ↑ "猫" 的 K,V 和 Step 1 完全相同!
//
//   Step 3: 输入 "猫追狗" → 预测 "在"
//     Q₃, K₃, V₃ = Attention("猫", "追", "狗") ← 又重新计算 "猫" 和 "追" 的 K,V!
//
//   Step N: 输入 "猫追狗...第N个" → 预测下一个
//     重新计算前面所有 N 个 token 的 K,V
//
// 计算量:
//   第 1 个 token: 1 次 K,V 计算
//   第 2 个 token: 2 次 K,V 计算 (重复 1 次)
//   第 N 个 token: N 次 K,V 计算
//   生成 T 个 token: 总共 T×(T+1)/2 次 K,V 计算 → O(T²)
//
// 对于 1000 token 的输出: 500,500 次 K,V 计算
//   其中 999,000/1,000 = 99.8% 是重复计算!

print("--- 1. 朴素推理: 大量重复计算 ---")
print()
print("生成 T 个 token:")
print("  朴素: 每步重算所有历史 → O(T²) 计算量")
print("  例: 1000 个 token → 500,500 次 KV 计算，99.8% 是重复的")
print()

// ═══════════════════════════════════════════════════════
// 2. KV Cache 的原理 — 只算新的
// ═══════════════════════════════════════════════════════
//
// 关键观察: Attention 中 K 和 V 只取决于 token 自身
//   K_i = embed(token_i) × W_k   — 只和 token_i 有关，和上下文无关
//   V_i = embed(token_i) × W_v   — 同上
//   (注意: 是 Attention 投影前的 K,V，不是经过 attention 的结果)
//
// 所以: 一旦算过 K_i 和 V_i，就存起来，以后不需要重算
//
//   Step 1: 输入 "猫"
//     计算 K₁, V₁ → 存入 cache → 用 K₁,V₁ 算 Attention → 预测 "追"
//
//   Step 2: 输入 "追"
//     计算 K₂, V₂ → 追加到 cache (现在有 K₁,V₁,K₂,V₂)
//     只算 Q₂ (新 token 的 Query)
//     用 Q₂ 和 [K₁,K₂] 算 Attention → 预测 "狗"
//     ↑ "猫" 的 K₁,V₁ 直接从 cache 读取，不重算!
//
//   Step N: 输入第 N 个 token
//     计算 K_N, V_N → 追加到 cache
//     用 Q_N 和 [K₁,...,K_N] 算 Attention
//     只多了 1 次 K,V 计算!
//
// 计算量:
//   每步只算 1 组新的 K,V → 总共 T 次 → O(T)
//   从 O(T²) 降到 O(T) — 巨大提升!

print("--- 2. KV Cache: O(T²) → O(T) ---")
print()
print("每次只算新 token 的 K,V，历史的从缓存读取")
print("计算量从 O(T²) 降到 O(T)")
print()

// ═══════════════════════════════════════════════════════
// 3. 两个阶段: Pre-fill 和 Decode
// ═══════════════════════════════════════════════════════
//
// ┌────────────────────────────────────────────────────────┐
// │ Pre-fill 阶段 (处理用户输入)                            │
// ├────────────────────────────────────────────────────────┤
// │                                                        │
// │ 用户输入: "请帮我写一首关于春天的诗" (12 tokens)        │
// │                                                        │
// │ 一次性处理全部 12 个 token:                              │
// │   12 个 token → 12 组 Q, K, V → 并行计算 Attention     │
// │   → 得到第 1 个输出 token                               │
// │                                                        │
// │ 特点: 并行处理 (矩阵运算，GPU 高效)                     │
// │       这一步不算重复，所以不需要 cache 的优化             │
// │       但结果 K, V 全部存入 cache                        │
// │                                                        │
// │ 速度: 处理速度 ≈ prefill 吞吐 (tokens/sec)              │
// │       受限于矩阵乘法带宽 (compute-bound)                │
// │                                                        │
// ├────────────────────────────────────────────────────────┤
// │ Decode 阶段 (逐个生成)                                  │
// ├────────────────────────────────────────────────────────┤
// │                                                        │
// │ 每步只有 1 个新 token:                                  │
// │   新 token → 1 组 Q, K, V                              │
// │   Q 和 cache 中所有历史 K, V 算 Attention               │
// │   → 得到下一个 token                                    │
// │                                                        │
// │ 特点: 串行 (每次只处理 1 个 token)                      │
// │       这是 KV Cache 发挥作用的阶段                       │
// │       没有缓存的话，每步要重算所有历史的 K, V            │
// │                                                        │
// │ 速度: 每个生成的 token 耗时 ≈ decode 延迟               │
// │       受限于内存带宽 (memory-bound)                     │
// │       因为每步要读取 cache 中所有历史的 K, V            │
// │                                                        │
// └────────────────────────────────────────────────────────┘
//
// 为什么 Decode 是 memory-bound:
//   假设已生成 1000 token，cache 有 1000 组 K, V
//   每步要读取 1000 × hidden_dim × 2 (K+V) × sizeof(float) 的数据
//   但只做 1 次 K,V 计算和 1 次 attention
//   计算/访存比很低 → 瓶颈在内存带宽
//
// 这就是为什么 Apple Silicon 的统一内存带宽很重要:
//   M1 Pro: 200 GB/s → 能快速读取 KV Cache
//   也是为什么长上下文 (100K+) 需要大量内存: KV Cache 太大了

print("--- 3. 两个阶段 ---")
print()
print("Pre-fill: 处理用户输入 (并行，compute-bound)")
print("  \"请帮我写一首关于春天的诗\" → 一次性处理 12 tokens → 缓存全部 K,V")
print()
print("Decode: 逐个生成 (串行，memory-bound)")
print("  新 token → 1 组 K,V → 和缓存的历史 K,V 算 Attention")
print("  瓶颈: 读取 KV Cache → 内存带宽决定速度")
print()

// ═══════════════════════════════════════════════════════
// 4. Mini 演示: 有/无 KV Cache 的计算量对比
// ═══════════════════════════════════════════════════════

print("--- 4. 计算量对比 ---")
print()

// 模拟: 隐藏维度 4，2 个 token 序列
let Wk = MLXArray([Float]([0.4, -0.2, 0.3, 0.1, -0.1, 0.6, -0.3, 0.4]), [2, 4])
let Wv = MLXArray([Float]([0.3, 0.2, -0.4, 0.5, -0.2, 0.4, 0.1, -0.3]), [2, 4])

// Token embeddings (简化)
let emb0 = MLXArray([Float]([1.0, 0.5, -0.3, 0.8]), [1, 4])  // "猫"
let emb1 = MLXArray([Float]([0.2, -0.1, 0.9, 0.4]), [1, 4])  // "追"

// === 朴素方法: 每次重算所有 ===
print("【朴素方法】生成第 3 个 token 时:")
let allEmbs = concatenated([emb0, emb1], axis: 0)  // [2, 4]
let allK_naive = matmul(allEmbs, Wk.T)  // 重算 2 个 token 的 K
let allV_naive = matmul(allEmbs, Wv.T)  // 重算 2 个 token 的 V
eval(allK_naive); eval(allV_naive)
print("  重算 K (2 tokens): \(allK_naive.shape)")
print("  重算 V (2 tokens): \(allV_naive.shape)")
print("  → 浪费! \"猫\" 的 K,V 和之前算的一样")
print()

// === KV Cache 方法: 只算新的 ===
print("【KV Cache 方法】Pre-fill 阶段 (处理 \"猫\"):")
let K0 = matmul(emb0, Wk.T)  // "猫" 的 K
let V0 = matmul(emb0, Wv.T)  // "猫" 的 V
eval(K0); eval(V0)
print("  计算 K₀: \(K0.shape) → 存入 cache")
print("  计算 V₀: \(V0.shape) → 存入 cache")
print()

print("【KV Cache 方法】Decode 第 1 个 token (\"追\"):")
let K1 = matmul(emb1, Wk.T)  // 只算 "追" 的 K
let V1 = matmul(emb1, Wv.T)  // 只算 "追" 的 V
eval(K1); eval(V1)
print("  计算 K₁: \(K1.shape) → 新增 1 个 (不重算 K₀!)")
print("  计算 V₁: \(V1.shape) → 新增 1 个 (不重算 V₀!)")

// 拼接 cache
let cachedK = concatenated([K0, K1], axis: 0)  // [2, 4]
let cachedV = concatenated([V0, V1], axis: 0)  // [2, 4]
print("  Cache: K=\(cachedK.shape), V=\(cachedV.shape)")
print("  → 结果相同，但只算了 1 个新 token 的 K,V")
print()

// ═══════════════════════════════════════════════════════
// 5. KV Cache 内存开销
// ═══════════════════════════════════════════════════════
//
// 每层的 KV Cache 大小:
//   2 (K+V) × seq_len × hidden_dim × sizeof(dtype)
//
// Qwen2.5-0.5B (24 层, FP16):
//   每层: 2 × seq_len × 896 × 2 bytes = 3,584 × seq_len bytes
//   24 层: 86,016 × seq_len bytes ≈ 86 KB × seq_len
//
//   1K tokens:   ~86 MB
//   8K tokens:   ~688 MB
//   32K tokens:  ~2.75 GB
//   128K tokens: ~11 GB  ← 16GB 机器上这都快满了!
//
// 优化方向:
//   1. GQA (Grouped Query Attention): 多个 Q 头共享 1 组 KV
//      Qwen2.5-0.5B: num_heads=14, kv_heads=2 → KV 只有单头的 1/7
//      128K tokens: 11 GB → ~1.6 GB
//
//   2. KV Cache 量化: FP16 → 8-bit → 4-bit
//      再省 2-4 倍
//
//   3. Paged Attention: 不预分配最大长度，按需分配
//      vLLM 的核心技术
//
//   4. Sliding Window: 只保留最近 W 个 token 的 cache
//      Mistral 的方法

print("--- 5. KV Cache 内存 ---")
print()
print("Qwen2.5-0.5B (24层, GQA: 14 Q头/2 KV头, FP16):")
let kbPerToken = 2 * 2 * 896 * 2  // 2(K+V) × 2(kv_heads) × hidden × 2(FP16 bytes)
let mbPerToken = Double(kbPerToken) / 1024.0 / 1024.0
let totalLayers = 24

for seqLen in [100, 1000, 8192, 32768] {
    let totalMB = mbPerToken * Double(seqLen) * Double(totalLayers)
    let label = seqLen >= 1000 ? "\(seqLen/1000)K" : "\(seqLen)"
    print("  \(String(format: "%6s", label)) tokens: ~\(String(format: "%6.1f", totalMB)) MB")
}
print()
print("优化: GQA 省头数、量化省精度、Paged Attention 省空间")
print()

// ═══════════════════════════════════════════════════════
// 6. 速度对比: 理论分析
// ═══════════════════════════════════════════════════════
//
// 生成 100 个 token，prompt 长度 P:
//
// 朴素方法:
//   Pre-fill: P 个 token → 1 次 forward
//   Decode:   每步处理 (P+i) 个 token → P+1, P+2, ..., P+100
//   总 token 处理: P + Σ(P+i) = P + 100P + 5050
//   例: P=50 → 10050 个 token 的 forward 计算
//
// KV Cache 方法:
//   Pre-fill: P 个 token → 1 次 forward (存 cache)
//   Decode:   每步只处理 1 个新 token → 100 次 forward
//   总 token 处理: P + 100 (每个只算 1 个新 token 的 K,V)
//   例: P=50 → 150 个 token 的 forward 计算
//
// 加速比: 10050 / 150 = 67x!

print("--- 6. 速度对比 ---")
print()
print("生成 100 token, prompt 50 token:")
print("  朴素: ~10050 次 token 处理")
print("  KV Cache: ~150 次 token 处理")
print("  加速: ~67x")
print()
print("生成越长，加速越大: 1000 token → ~500x")
print()

print("=== Step08 完成! ===")
print("下一步: Step09_Generate — 把所有步骤串起来，实现完整生成")
