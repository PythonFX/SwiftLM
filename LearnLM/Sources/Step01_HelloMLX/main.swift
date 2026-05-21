// Step01: Hello MLX — 理解张量、GPU计算和Metal的基础
//
// 学习目标:
//   1. MLXArray 是什么 — LLM中一切计算的基石
//   2. GPU vs CPU — 为什么LLM推理需要GPU
//   3. 矩阵乘法 — LLM最核心的计算操作
//   4. 内存管理 — unified memory下GPU如何分配和释放内存
//
// 重要: Apple GPU 不支持 Float64 (Double)，所有张量必须用 Float32
//
// 运行: cd LearnLM && swift run Step01_HelloMLX

import CoreFoundation
import MLX

// ═══════════════════════════════════════════════════════
// 1. MLXArray — LLM中所有数据的容器
// ═══════════════════════════════════════════════════════
//
// 在LLM中:
//   - 模型权重 (weights) → MLXArray (高维浮点数组)
//   - 输入文本 → tokenize → 整数 MLXArray
//   - 中间激活值 (activations) → MLXArray
//   - 最终输出概率 → MLXArray
//
// MLXArray 在 Apple Silicon 上自动使用 GPU (Metal)

print("=== 1. 创建 MLXArray ===")

// 标量 (0维) — 单个数字
let scalar = MLXArray(Float(42.0))
print("标量:     \(scalar.shape) → \(scalar)")

// 向量 (1维) — 一行数字 (比如一个token的embedding)
let vector = MLXArray([Float]([1, 2, 3, 4]))
print("向量:     \(vector.shape) → \(vector)")

// 矩阵 (2维) — 用一维数组 + shape 参数创建
// 比如一个token序列的embedding: 3个token, 每个用4维向量表示
let matrix = MLXArray([Float](  [1, 2, 3, 4,
                       5, 6, 7, 8,
                       9, 10, 11, 12]), [3, 4])
print("矩阵:     \(matrix.shape)")  // shape = [3, 4]  → 3行4列

// 3维张量 — shape从右往左读: 基础向量长度 → 多少个向量 → 多少组
// [2, 4, 8] = 2组 × 4个向量 × 8维
//
// 在 LLM 推理中对应 (batch, seq_len, hidden_dim):
//   dim=2 (batch):     同时处理的输入条数 (服务器场景下多个并发请求)
//   dim=1 (seq_len):   每条输入的 token 数量
//   dim=0 (hidden):    每个 token 的向量维度
//
// 实际推理中, 单用户生成阶段几乎总是 batch=1: [1, seq_len, hidden_dim]
// batch>1 主要用于 inference server 的 continuous batching (如 SwiftLM)
let tensor3d = MLXArray.zeros([2, 4, 8])  // batch=2, seq=4, dim=8
print("3维张量:  \(tensor3d.shape)")

// ═══════════════════════════════════════════════════════
// 2. GPU计算 — 为什么要用GPU
// ═══════════════════════════════════════════════════════
//
// LLM推理的核心操作是矩阵乘法 (matmul):
//   output = input × weight
//
// Qwen3.5-4B 的一个attention层:
//   input:  [seq_len, 2560]    (序列长度 × 隐藏维度)
//   weight: [2560, 2560]       (隐藏维度 × 隐藏维度)
//   output: [seq_len, 2560]    (结果)
//
// 这需要 2560 × 2560 = 6.5M 次乘加运算 / 每个token
// 一个4B模型大约有36层 → 每个token需要 ~240M 次运算
// CPU串行做: 很慢. GPU并行做: 快很多.

print("\n=== 2. 矩阵乘法 (LLM的核心计算) ===")

let a = MLXArray([Float]([1, 2, 3, 4]), [2, 2])
let b = MLXArray([Float]([5, 6, 7, 8]), [2, 2])

// matmul — 这就是GPU每天都在为LLM做的计算
let result = matmul(a, b)
print("matmul结果 shape=\(result.shape):")
print("  \(result)")
// [[19, 22],    ← [1*5+2*7, 1*6+2*8]
//  [43, 50]]    ← [3*5+4*7, 3*6+4*8]

// ═══════════════════════════════════════════════════════
// 3. 实际LLM规模的矩阵乘法 — 感受GPU的速度
// ═══════════════════════════════════════════════════════
//
// 模拟 Qwen3.5-4B 一层的权重 (2560 × 2560)

print("\n=== 3. LLM规模的矩阵乘法 ===")

let hiddenDim = 2560  // Qwen3.5-4B的隐藏维度
let seqLen = 1        // 生成时每次只有1个token

// 模拟一个层的权重矩阵 (随机初始化) — MLXRandom.normal 默认生成 Float32
let weight = MLXRandom.normal([hiddenDim, hiddenDim])
// 模拟输入 (1个token的embedding)
let input = MLXRandom.normal([seqLen, hiddenDim])

print("权重 shape: \(weight.shape)  (~\(hiddenDim * hiddenDim / 1024 / 1024)M 参数)")
print("输入 shape: \(input.shape)")

// 计时
let start = CFAbsoluteTimeGetCurrent()
let output = matmul(input, weight)
// MLX是惰性求值的 — 需要显式求值(Eval)才能真正执行GPU计算
eval(output)
let elapsed = CFAbsoluteTimeGetCurrent() - start

print("输出 shape: \(output.shape)")
print("耗时: \(String(format: "%.3f", elapsed * 1000))ms")
print("(MLX会自动将这个计算发送到Apple GPU上执行)")

// ═══════════════════════════════════════════════════════
// 4. 惰性求值 (Lazy Evaluation) — MLX的重要概念
// ═══════════════════════════════════════════════════════
//
// MLX不会在你写 a + b 的时候立刻计算，而是记录操作图
// 等到你调用 eval() 或者需要读取结果时才真正执行
// 这样可以把多个操作融合(fuse)成一个GPU kernel，提高效率

print("\n=== 4. 惰性求值 ===")

let x = MLXArray([Float]([1, 2, 3]))
let y = x * 2        // 还没计算
let z = y + 1        // 还没计算
let w = z * z        // 还没计算
print("操作已定义，但GPU还没执行")
eval(w)              // 现在才真正执行: x*2+1 然后平方
print("eval()后结果: \(w)")  // [9, 25, 49]

// ═══════════════════════════════════════════════════════
// 5. 内存管理 — Apple Silicon的统一内存
// ═══════════════════════════════════════════════════════
//
// Apple Silicon的关键优势: CPU和GPU共享同一块物理内存
// 这意味着:
//   - 不需要把数据从CPU内存复制到GPU内存 (像NVIDIA那样)
//   - 模型权重只需要存一份, CPU和GPU都能直接访问
//   - 这就是为什么16GB Mac能跑LLM — 统一内存被GPU直接使用

print("\n=== 5. 内存大小计算 ===")

// Qwen3.5-4B 在不同量化下的内存占用
let paramsB: Float = 4.0  // 4B参数
let bytesPerParam_FP16: Float = 2.0
let bytesPerParam_4bit: Float = 0.5
let bytesPerParam_3bit: Float = 0.375

let fp16GB = paramsB * bytesPerParam_FP16
let q4GB = paramsB * bytesPerParam_4bit
let q3GB = paramsB * bytesPerParam_3bit

print("Qwen3.5-4B 参数量: ~\(Int(paramsB))B")
print("  FP16:   \(fp16GB) GB")
print("  4-bit:  \(q4GB) GB")
print("  3-bit:  \(String(format: "%.2f", q3GB)) GB")
print("  在16GB Mac上, 4-bit只占\(q4GB)GB → 完全放得进内存!")

print("\n=== Step01 完成! ===")
print("下一步: Step02_Tokenizer — 学习如何把文字变成数字(Tokenizer)")
