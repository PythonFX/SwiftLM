// Step03: Load Weights — 理解模型文件结构和权重加载
//
// 学习目标:
//   1. config.json — 理解模型架构参数 (hidden_size, num_layers, etc.)
//   2. safetensors — 二进制权重文件格式
//   3. 权重名 (weight names) 如何映射到模型的各个层
//   4. 将权重数据加载为 MLXArray
//
// 运行: cd LearnLM && swift run Step03_LoadWeights

import Foundation
import MLX
import Hub

print("=== Step03: Load Weights ===")
print()

// ═══════════════════════════════════════════════════════
// 1. 模型文件结构概览
// ═══════════════════════════════════════════════════════
//
// 一个 HuggingFace 模型仓库通常包含:
//
//   config.json                               — 模型架构参数 (几KB)
//   tokenizer.json                            — Tokenizer 词汇表 (~20MB)
//   tokenizer_config.json                     — Tokenizer 配置
//   generation_config.json                    — 生成参数 (temperature等)
//   model.safetensors                         — 模型权重 (小模型，单个文件)
//   model-00001-of-00008.safetensors          — 模型权重 (大模型，分片存储)
//   model.safetensors.index.json              — 分片索引 (权重名 → 文件映射)
//
// 推理前必须:
//   1. 读 config.json → 知道模型结构 (多少层、多大)
//   2. 读 safetensors → 加载权重数据 (矩阵/张量)
//   3. 两者结合 → 初始化模型的每一层 → 开始推理

// ═══════════════════════════════════════════════════════
// 2. 读取 config.json — 模型的"设计图纸"
// ═══════════════════════════════════════════════════════
//
// config.json 就像盖楼前的设计图纸:
//   - hidden_size → 每层楼多宽 (每个token的向量维度)
//   - num_hidden_layers → 盖多少层 (Transformer层数)
//   - vocab_size → 有多少种建材 (词汇表大小)

print("--- 2. 读取 config.json ---")
let modelName = "Qwen/Qwen2.5-0.5B"
print("正在下载 config.json: \(modelName)...")
let hubApi = HubApi.shared
let repo = Hub.Repo(id: modelName)
let modelFolder = try await hubApi.snapshot(from: repo, matching: ["config.json"])

let configURL = modelFolder.appendingPathComponent("config.json")
let configData = try Data(contentsOf: configURL)
let config = try JSONSerialization.jsonObject(with: configData) as! [String: Any]

let hiddenSize = config["hidden_size"] as! Int
let numLayers = config["num_hidden_layers"] as! Int
let numHeads = config["num_attention_heads"] as! Int
let headDim = hiddenSize / numHeads
let vocabSize = config["vocab_size"] as! Int
let intermediateSize = config["intermediate_size"] as! Int

print()
print("config.json 关键参数:")
print("  hidden_size:         \(hiddenSize)       ← 每个token的向量维度 (Step01讲过的dim)")
print("  num_hidden_layers:   \(numLayers)        ← Transformer 层数 (模型有多'深')")
print("  num_attention_heads: \(numHeads)         ← 注意力头数 (Step05会讲)")
print("  head_dim:            \(headDim)        ← 每个头的维度 (hidden_size / num_heads)")
print("  vocab_size:          \(vocabSize)     ← 词汇表大小 (Step02讲过的151K)")
print("  intermediate_size:   \(intermediateSize)      ← FFN 中间层维度 (Step06会讲)")
print()

// 参数量估算 — 这些数字意味着多少参数
let embeddingParams = vocabSize * hiddenSize
let attnParamsPerLayer = 4 * hiddenSize * hiddenSize  // Q, K, V, O 四个投影矩阵
let ffnParamsPerLayer = 3 * hiddenSize * intermediateSize  // gate, up, down
let normParamsPerLayer = 2 * hiddenSize  // 两个 RMSNorm
let perLayer = attnParamsPerLayer + ffnParamsPerLayer + normParamsPerLayer
let totalParams = embeddingParams + numLayers * perLayer + embeddingParams  // +lm_head

print("参数量估算:")
let embM = Double(embeddingParams) / 1_000_000
let attnM = Double(attnParamsPerLayer) / 1_000_000
let ffnM = Double(ffnParamsPerLayer) / 1_000_000
let totalM = Double(totalParams) / 1_000_000
let totalB = Double(totalParams) / 1_000_000_000
print("  Embedding:    \(String(format: "%6.1fM", embM))  (vocab_size × hidden_size)")
print("  每层 Attention: \(String(format: "%5.1fM", attnM))  (4 × hidden²)")
print("  每层 FFN:     \(String(format: "%6.1fM", ffnM))  (3 × hidden × intermediate)")
print("  总计: ~\(String(format: "%.0f", totalM))M 参数 (~\(String(format: "%.2f", totalB))B)")
print()

// ═══════════════════════════════════════════════════════
// 3. Safetensors 格式 — 模型权重的存储格式
// ═══════════════════════════════════════════════════════
//
// safetensors 是 HuggingFace 推荐的权重格式:
//   - 比 PyTorch 的 .bin 格式更快更安全
//   - 不执行任意代码 (安全的二进制格式)
//   - 支持 mmap 零拷贝加载
//
// 二进制结构 (非常简单):
//
//   ┌───────────────────────────────────────┐
//   │  8 bytes: header_size (UInt64 LE)     │  ← 告诉你 header 有多大
//   ├───────────────────────────────────────┤
//   │  N bytes: JSON header                 │  ← 描述每个权重张量
//   │  {                                    │
//   │    "layers.0.q_proj.weight": {        │
//   │      "dtype": "F16",                  │
//   │      "shape": [896, 896],             │
//   │      "data_offsets": [0, 1605632]     │  ← 字节范围 [start, end)
//   │    },                                 │
//   │    "layers.0.k_proj.weight": { ... }  │
//   │  }                                    │
//   ├───────────────────────────────────────┤
//   │  剩余: 所有张量的原始二进制数据       │  ← 连续排列，不压缩
//   └───────────────────────────────────────┘

print("--- 3. Safetensors 格式 ---")
print("结构: [8字节 header大小] [JSON header] [原始张量数据]")
print("优点: 加载快 (可 mmap), 安全 (无代码执行), 格式简单")
print()

// ═══════════════════════════════════════════════════════
// 4. 实战: 创建并解析一个 mini safetensors
// ═══════════════════════════════════════════════════════
//
// 为了真正理解格式，我们创建一个最小的 safetensors 文件
// 包含一个 2×3 的 float32 矩阵 (模拟一个权重)

print("--- 4. Mini safetensors 演示 ---")
print()

// 4a. 准备权重数据: 2×3 矩阵
let demoShape: [Int] = [2, 3]
let demoFloats: [Float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
var demoBytes = Data()
for f in demoFloats {
    var value = f
    withUnsafeBytes(of: &value) { demoBytes.append(contentsOf: $0) }
}
print("权重数据: \(demoFloats) → \(demoBytes.count) bytes (6个float32)")

// 4b. 构建 safetensors 文件
let header: [String: Any] = [
    "demo_weight": [
        "dtype": "F32",
        "shape": demoShape,
        "data_offsets": [0, demoBytes.count]
    ] as [String: Any]
]
let headerJSON = try JSONSerialization.data(withJSONObject: header)

var safetensorFile = Data()
// 写入 header size (8字节, UInt64 little-endian)
let headerSize = UInt64(headerJSON.count)
var hsLE = headerSize.littleEndian
withUnsafeBytes(of: &hsLE) { safetensorFile.append(contentsOf: $0) }
safetensorFile.append(headerJSON)
safetensorFile.append(demoBytes)

print("构建 safetensors: \(safetensorFile.count) bytes")
print("  [0..8):     header size = \(headerJSON.count)")
print("  [8..\(8 + headerJSON.count)): JSON header")
print("  [\(8 + headerJSON.count)..\(safetensorFile.count)): tensor data (\(demoBytes.count) bytes)")
print()

// 4c. 解析: 读回 safetensors (模拟加载真实模型权重)
print("解析 safetensors:")

// Step 1: 读 header size (前8字节, UInt64 little-endian)
var readHeaderSize: UInt64 = 0
for i in 0..<8 {
    readHeaderSize |= UInt64(safetensorFile[i]) << (i * 8)
}

// Step 2: 读 JSON header
let hStart = 8
let hEnd = 8 + Int(readHeaderSize)
let parsedHeader = try JSONSerialization.jsonObject(
    with: safetensorFile.subdata(in: hStart..<hEnd)
) as! [String: Any]

// Step 3: 遍历每个权重
let dataStart = hEnd
for (name, info) in parsedHeader {
    guard name != "__metadata__",
          let info = info as? [String: Any],
          let dtype = info["dtype"] as? String,
          let shape = info["shape"] as? [Int],
          let offsets = info["data_offsets"] as? [Int]
    else { continue }

    let tensorData = safetensorFile.subdata(in: (dataStart + offsets[0])..<(dataStart + offsets[1]))

    // Step 4: 字节 → MLXArray (模拟权重的实际加载过程)
    let floatCount = tensorData.count / MemoryLayout<Float>.size
    var floats = [Float](repeating: 0, count: floatCount)
    _ = floats.withUnsafeMutableBytes { dest in
        tensorData.copyBytes(to: dest)
    }
    let weight = MLXArray(floats, shape)

    print("  权重: \"\(name)\"")
    print("    dtype: \(dtype), shape: \(shape)")
    print("    字节范围: [\(offsets[0]), \(offsets[1]))")
    print("    MLXArray: \(weight)")
}
print()

// ═══════════════════════════════════════════════════════
// 5. 真实模型的权重名映射
// ═══════════════════════════════════════════════════════
//
// safetensors 里的每个权重都有一个名字，对应模型中的一层
// Qwen2.5 的命名规则:
//
//   model.embed_tokens.weight                          ← Embedding (Step04)
//   model.layers.{N}.self_attn.q_proj.weight           ← 第N层 Query 投影 (Step05)
//   model.layers.{N}.self_attn.k_proj.weight           ← 第N层 Key 投影
//   model.layers.{N}.self_attn.v_proj.weight           ← 第N层 Value 投影
//   model.layers.{N}.self_attn.o_proj.weight           ← 第N层 Output 投影
//   model.layers.{N}.mlp.gate_proj.weight              ← 第N层 FFN gate (Step06)
//   model.layers.{N}.mlp.up_proj.weight                ← 第N层 FFN up
//   model.layers.{N}.mlp.down_proj.weight              ← 第N层 FFN down
//   model.layers.{N}.input_layernorm.weight            ← 第N层 RMSNorm
//   model.layers.{N}.post_attention_layernorm.weight    ← 第N层 RMSNorm
//   model.norm.weight                                  ← 最终 RMSNorm
//   lm_head.weight                                     ← 输出投影
//
// 数据流:
//   token IDs → embed_tokens → layers[0..N] → norm → lm_head → logits
//   (Step04)    (Step04)      (Step05+06)    (Step06)  (输出)
//
// Qwen2.5-0.5B: 24层 × 每层10个权重 + 3个顶层 ≈ 243个权重张量
// Qwen2.5-72B:  80层 × 每层10个权重 + 3个顶层 ≈ 803个权重张量
// 大模型分片: model-00001-of-00030.safetensors (30个文件)

print("--- 5. 真实模型的权重名映射 ---")
print()
print("Qwen2.5-0.5B 的权重结构:")
print()
print("  model.embed_tokens.weight              [\(vocabSize), \(hiddenSize)]   ← Embedding")
for i in 0..<min(2, numLayers) {
    print("  model.layers.\(i).self_attn.q_proj.weight  [\(hiddenSize), \(hiddenSize)]  ← 第\(i)层 Q投影")
    print("  model.layers.\(i).self_attn.k_proj.weight  [\(hiddenSize), \(hiddenSize)]  ← 第\(i)层 K投影")
    print("  model.layers.\(i).self_attn.v_proj.weight  [\(hiddenSize), \(hiddenSize)]  ← 第\(i)层 V投影")
    print("  model.layers.\(i).self_attn.o_proj.weight  [\(hiddenSize), \(hiddenSize)]  ← 第\(i)层 O投影")
    print("  model.layers.\(i).mlp.gate_proj.weight     [\(intermediateSize), \(hiddenSize)]  ← 第\(i)层 FFN gate")
    print("  model.layers.\(i).mlp.up_proj.weight       [\(intermediateSize), \(hiddenSize)]  ← 第\(i)层 FFN up")
    print("  model.layers.\(i).mlp.down_proj.weight     [\(hiddenSize), \(intermediateSize)]  ← 第\(i)层 FFN down")
    print("  model.layers.\(i).input_layernorm.weight    [\(hiddenSize)]                ← 第\(i)层 RMSNorm")
    print("  model.layers.\(i).post_attn_layernorm.weight [\(hiddenSize)]                ← 第\(i)层 RMSNorm")
    if i == 0 { print("  ... (省略其余 \(numLayers - 2) 层，结构相同)") }
}
print("  model.norm.weight                      [\(hiddenSize)]                ← 最终 RMSNorm")
print("  lm_head.weight                         [\(vocabSize), \(hiddenSize)]   ← 输出投影")
print()

// ═══════════════════════════════════════════════════════
// 6. 内存占用 — 这些权重要吃多少内存
// ═══════════════════════════════════════════════════════

print("--- 6. 内存占用 ---")
let bodyParams = Double(numLayers * perLayer)
let vocabParams = Double(2 * embeddingParams)  // Embedding + lm_head
let bodyM = bodyParams / 1_000_000
let vocabM = vocabParams / 1_000_000

print("  参数分布:")
print("    模型主体 (Attention+FFN+Norm): \(String(format: "%6.1fM", bodyM))")
print("    词汇表 (Embedding+lm_head):   \(String(format: "%6.1fM", vocabM))")
print()
print("  不同精度下的内存:")
let fp16GB = Double(totalParams) * 2 / 1_000_000_000
let mixedGB = (bodyParams * 0.5 + vocabParams * 2) / 1_000_000_000
print("    FP16:  \(String(format: "%5.2f", fp16GB)) GB")
print("    4-bit: \(String(format: "%5.2f", mixedGB)) GB (主体4bit + 词汇表FP16)")
print()
print("  (回顾 Step02 的结论: 模型越小，词汇表的内存'税'越重)")
print()

print("=== Step03 完成! ===")
print("下一步: Step04_Embedding — 学习如何把 token ID 变成语义向量")
