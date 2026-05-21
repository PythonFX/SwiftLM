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
//
// ┌─────────────────────────────────────────────────────────┐
// │ Embedding: vocab_size × hidden_size = 136.1M 参数       │
// ├─────────────────────────────────────────────────────────┤
// │                                                         │
// │ 什么是 Embedding:                                        │
// │   一个二维查找表 (lookup table)，shape = [151936, 896]    │
// │   每行是一个 token 的 896 维向量                          │
// │                                                         │
// │   为什么要 Embedding:                                    │
// │     Token ID 是任意编号 (如 "猫"=1234, "狗"=5678)        │
// │     这些整数没有语义 — 1234 和 5678 的距离不代表含义差异    │
// │     但 "猫" 和 "狗" 语义上显然比 "猫" 和 "的" 更近        │
// │     Embedding 把离散整数映射到连续向量空间，解决这个问题    │
// │                                                         │
// │   怎么用:                                                │
// │     "Hello" → tokenizer → token ID = 9707               │
// │     → embedding_matrix[9707] → [896] 维向量              │
// │     就是数组索引，没有乘法运算                              │
// │                                                         │
// │   怎么学出来的:                                           │
// │     初始时全部随机数                                      │
// │     训练过程中不断做 "预测下一个词"                        │
// │     预测错了 → 反向传播 → 调整矩阵里的数字                 │
// │     训练完后: "猫"和"狗"的向量自动靠近，"猫"和"的"自动远离  │
// │     136.1M 参数 = 151936 行 × 896 列，全是训练学出来的     │
// │                                                         │
// │   在 safetensors 中的权重名:                              │
// │     model.embed_tokens.weight  [151936, 896]             │
// │                                                         │
// ├─────────────────────────────────────────────────────────┤
// │ 每层 Attention: 4 × hidden² = 3.2M 参数                 │
// ├─────────────────────────────────────────────────────────┤
// │                                                         │
// │ 4 个投影矩阵，每个 [896, 896]:                            │
// │   W_q → Q (Query):  "我在找什么？"                        │
// │   W_k → K (Key):    "我有什么特征？"                      │
// │   W_v → V (Value):  "我的实际内容是什么？"                 │
// │   W_o → O (Output): 合并多头注意力的结果                   │
// │                                                         │
// │ 重要: W_q/k/v/o 是固定的模型权重，推理时不变              │
// │   训练时学出来，保存在 safetensors 里                      │
// │   不同输入只是 X 不同，W 不变                              │
// │   就像固定的"提问模板"，不同输入套模板后提出不同的问题       │
// │                                                         │
// │ 完整推理流程 (以 "猫坐在垫子上" 为例，6个token):           │
// │                                                         │
// │   起点: Embedding 后得到 X [6, 896]                       │
// │                                                         │
// │   Step 1: 生成 Q, K, V (三个不同的矩阵提取不同侧面)        │
// │     Q = X × W_q    [6,896]×[896,896] = [6, 896]          │
// │     K = X × W_k    [6,896]×[896,896] = [6, 896]          │
// │     V = X × W_v    [6,896]×[896,896] = [6, 896]          │
// │                                                         │
// │     同一条输入 X，乘不同矩阵:                              │
// │       W_q 让 "猫" 学会表达 "我在找坐的对象"                │
// │       W_k 让 "垫子" 学会表达 "我是可以坐的物体"            │
// │       W_v 让 "垫子" 学会表达 "我的语义内容是柔软家具"       │
// │                                                         │
// │   Step 2: 切分多头                                       │
// │     Q [6,896] → [6, 14, 64]                              │
// │                  ↑  ↑  ↑                                 │
// │                  |  |  每个头的维度                        │
// │                  |  14个头 (每个头关注不同关系)              │
// │                  6个token                                 │
// │     K, V 同理切分                                        │
// │                                                         │
// │     为什么多头: 1个896维注意力只能从一个角度看              │
// │     14个头可以同时关注语法、位置、语义、指代等不同关系       │
// │                                                         │
// │   Step 3: 计算注意力分数                                  │
// │     score = Q × K^T  (每个头内: [6,64]×[64,6]=[6,6])      │
// │                                                         │
// │     结果是 6×6 的注意力矩阵 (单头, 简化数值):              │
// │            看→  "猫"  "坐"  "在" "垫子" "子"  "上"        │
// │         "猫"  [0.05 0.10 0.05 0.55 0.08 0.05]            │
// │         "坐"  [0.35 0.08 0.12 0.15 0.08 0.12]            │
// │         ...                                              │
// │                                                         │
// │     "猫" 看 "垫子" = 0.55 → 55%注意力在"垫子"上           │
// │     "坐" 看 "猫"   = 0.35 → 主语是猫                      │
// │                                                         │
// │     score / sqrt(64) → softmax → 每行加起来=1             │
// │                                                         │
// │   Step 4: 用 V 加权汇总                                  │
// │     context = softmax_result × V                         │
// │                                                         │
// │     "猫"的新向量 = 0.05×V["猫"] + 0.10×V["坐"]            │
// │                   + 0.05×V["在"] + 0.55×V["垫子"]         │
// │                   + 0.08×V["子"] + 0.05×V["上"]           │
// │                                                         │
// │     55%来自"垫子" → "猫"的向量融合了上下文                 │
// │     之前只知道自己是猫，现在还知道有垫子                    │
// │                                                         │
// │   Step 5: O 矩阵合并多头                                 │
// │     14个头各出 [6,64] → 拼接 [6,896]                     │
// │     output = 拼接结果 × W_o  [6,896]×[896,896]=[6,896]   │
// │                                                         │
// │     W_o 把14个头学到的不同关系融合:                        │
// │       head1: "猫和坐是主谓关系"                            │
// │       head7: "猫和垫子是语义关联"                          │
// │       → W_o 综合成一个896维向量                           │
// │                                                         │
// │   输出 [6, 896]: 每个 token 的向量现在包含了上下文信息     │
// │                                                         │
// │   一句话: Q和K算"谁跟谁相关"，V提供实际内容，O合并多头结果  │
// │                                                         │
// │   正确理解 W_q/k/v — 固定矩阵凭什么能匹配万物:            │
// │                                                         │
// │   W_q: 固定的几何变换函数，把任意输入投影到"查询空间"      │
// │     训练让它学会: 在这个空间里，相关的token距离近           │
// │     它不知道什么是"动作对象"、"主语"、"宾语"               │
// │     它只知道: 经过我变换后，需要互相关注的token会自然靠近   │
// │                                                         │
// │   W_k: 另一个固定的几何变换函数，把输入投影到"被查询空间"   │
// │     Q和K是配对训练的 — 它们的空间必须兼容                  │
// │     Q向量靠近的K向量就是要被关注的                          │
// │                                                         │
// │   W_v: 固定的提取函数，把输入变成"值得传递的内容"          │
// │     不关心匹配，只关心: 如果我被选中了，该传递什么信息      │
// │                                                         │
// │   固定矩阵能处理所有输入的原因:                            │
// │     跟 f(x)=2x 能算所有数字是同一个道理                    │
// │     不同输入在896维空间本来就落在不同位置                   │
// │     固定变换把相关的输入映射到相近的区域                    │
// │     这不是在"理解语言"，是在高维几何里做模式匹配            │
// │                                                         │
// ├─────────────────────────────────────────────────────────┤
// │ 每层 FFN: 3 × hidden × intermediate = 13.1M 参数        │
// ├─────────────────────────────────────────────────────────┤
// │                                                         │
// │ 3 个矩阵:                                                │
// │   W_gate [4864, 896]  ← 门控，决定哪些信息通过            │
// │   W_up   [4864, 896]  ← 扩展维度 (896→4864)              │
// │   W_down [896, 4864]  ← 压缩回来 (4864→896)              │
// │                                                         │
// │ 计算:                                                    │
// │   gate = 激活函数(输入 × W_gate)  ← 0~1之间的开关         │
// │   up   = 输入 × W_up              ← 扩展到高维            │
// │   中间结果 = gate × up             ← 过滤+变换            │
// │   输出 = 中间结果 × W_down         ← 压缩回原维度          │
// │                                                         │
// │ 为什么先扩展再压缩:                                       │
// │   896 维空间无法表达复杂概念                              │
// │   扩展到 4864 维 → 在更大空间做非线性变换 → 表达更复杂关系  │
// │   再压缩回 896 维 → 保持维度一致，传给下一层               │
// │                                                         │
// │ FFN 是模型参数量的大头 (13.1M vs Attention 3.2M)          │
// │ 这也是 MoE 模型选择在 FFN 层做路由的原因 (省掉FFN最划算)   │
// │ (Step06 会详细实现)                                       │
// │                                                         │
// ├─────────────────────────────────────────────────────────┤
// │ lm_head: vocab_size × hidden_size = 136.1M 参数         │
// ├─────────────────────────────────────────────────────────┤
// │                                                         │
// │ 和 Embedding 一样大，因为它们是同一个问题的两端:            │
// │   Embedding: token ID → 语义向量 (查表)                  │
// │   lm_head:   语义向量 → token 概率 (矩阵乘法)            │
// │                                                         │
// │ 计算:                                                    │
// │   logits = 最终向量 × lm_head^T                          │
// │   → [151936] 个分数，每个 token 一个                      │
// │   → softmax → 概率分布 → 选概率最高的 → 就是下一个字      │
// │                                                         │
// │ 有些模型让 Embedding 和 lm_head 共享同一个矩阵             │
// │ 叫 tied weights，直接省一半内存                           │
// │                                                         │
// └─────────────────────────────────────────────────────────┘
//  多头注意力机制： 14 个头各自分化出不同的模式，是训练自然而然找到的最优解，而不是设计出来的。

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
// Embedding 表的每一行，是该 token 在 896 维语义空间中的坐标。 有多少种不同的token就有多少行。
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
