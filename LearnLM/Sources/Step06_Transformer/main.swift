// Step06: Transformer Block — 完整的 Transformer 层
//
// 学习目标:
//   1. Transformer 层的完整结构: Norm → Attention → Norm → FFN
//   2. RMSNorm — 归一化，稳定训练
//   3. FFN (Feed-Forward Network) — 非线性变换
//   4. 残差连接 (Residual Connection) — 为什么需要它
//   5. 用真实模型权重跑完一整层
//   6. lm_head — 最终投影到词表
//
// 运行: cd LearnLM && swift run Step06_Transformer

import Foundation
import MLX
import Hub
import Tokenizers

print("=== Step06: Transformer Block ===")
print()

// ═══════════════════════════════════════════════════════
// 1. Transformer 层的完整结构
// ═══════════════════════════════════════════════════════
//
// 每个 Transformer 层做的事情 (Qwen2.5):
//
//   输入 x [seq_len, 896]
//     │
//     ├─→ RMSNorm ─→ Attention ─→ + ─→ 残差连接
//     │                              │
//     │                              ├─→ RMSNorm ─→ FFN ─→ + ─→ 残差连接
//     │                              │                        │
//     └──────────────────────────────┴────────────────────────┘
//                                                                │
//                                                          输出 [seq_len, 896]
//
// 关键设计:
//   1. Pre-Norm: 先归一化再做 Attention/FFN (比 Post-Norm 更稳定)
//   2. 残差连接: output = input + sublayer(norm(input))
//      保证梯度流通，即使子层学不到有用的东西，信息也不会丢失
//   3. 每层不改变 shape: 输入 [seq, 896] → 输出 [seq, 896]
//      可以堆叠任意多层 (Qwen2.5-0.5B 堆叠 24 层)

print("--- 1. Transformer 层结构 ---")
print("每层: x → RMSNorm → Attention → +残差 → RMSNorm → FFN → +残差 → 输出")
print("关键: 输入输出 shape 相同 → 可堆叠 N 层")
print()

// ═══════════════════════════════════════════════════════
// 2. RMSNorm — 比 LayerNorm 更高效的归一化
// ═══════════════════════════════════════════════════════
//
// LayerNorm: normalize(x) = (x - mean) / std * weight + bias
// RMSNorm:   normalize(x) = x / rms * weight
//   其中 rms = sqrt(mean(x^2) + eps)
//
// RMSNorm 去掉了均值减法和偏置，计算更快，效果接近
// Qwen2.5 全部使用 RMSNorm

print("--- 2. RMSNorm ---")
print()

func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-5) -> MLXArray {
    // x: [seqLen, dim], weight: [dim]
    let x2 = x * x
    let lastAxis = x.shape.count - 1
    let meanX2 = mean(x2, axis: lastAxis)  // [seqLen]
    let rms = sqrt(meanX2 + MLXArray(Float(eps)))   // [seqLen]
    return x / rms.expandedDimensions(axis: lastAxis) * weight
}

// Mini demo
let normInput = MLXArray([Float]([3.0, -1.0, 2.0, 0.5]), [1, 4])
let normWeight = MLXArray([Float]([1.0, 1.0, 1.0, 1.0]))
let normed = rmsNorm(normInput, weight: normWeight)
eval(normed)
print("输入:     \(normInput)")
print("RMSNorm:  \(normed)  (向量长度归一化到 ~1)")
print()

// ═══════════════════════════════════════════════════════
// 3. FFN — 前馈网络，非线性变换
// ═══════════════════════════════════════════════════════
//
// Qwen2.5 使用 SwiGLU 变体的 FFN:
//
//   gate = silu(x @ W_gate)    ← 门控信号 (0~1)
//   up   = x @ W_up            ← 扩展维度
//   output = (gate * up) @ W_down  ← 过滤 + 压缩
//
// 其中 silu(x) = x * sigmoid(x)
//
// 维度变化:
//   x:         [seq, 896]
//   gate/up:   [seq, 4864]  (896 → 4864, 扩展 ~5.4 倍)
//   down:      [seq, 896]   (4864 → 896, 压缩回来)
//
// 为什么先扩展再压缩:
//   896 维空间表达力有限
//   在 4864 维空间做非线性变换 → 表达更复杂的关系
//   再压缩回 896 维 → 保持维度一致，传给下一层
//
// FFN 是模型参数量的大头:
//   Qwen2.5-0.5B: 每层 FFN ~13M 参数 vs Attention ~3.2M
//   这也是 MoE 模型选择在 FFN 做专家路由的原因

print("--- 3. FFN (SwiGLU) ---")
print()
print("FFN: x → gate(silu) + up → 相乘 → down → 输出")
print("维度: [seq, 896] → [seq, 4864] → [seq, 896]")
print("参数量: ~13M/层 (模型参数最多的大块)")
print()

func silu(_ x: MLXArray) -> MLXArray {
    return x * sigmoid(x)
}

func ffn(_ x: MLXArray, gateW: MLXArray, upW: MLXArray, downW: MLXArray) -> MLXArray {
    let gate = silu(matmul(x, gateW.T))
    let up = matmul(x, upW.T)
    return matmul(gate * up, downW.T)
}

// ═══════════════════════════════════════════════════════
// 4. 残差连接 — 信息高速公路
// ═══════════════════════════════════════════════════════
//
// output = x + sublayer(norm(x))
//          ↑     ↑
//          │     子层输出 (Attention 或 FFN)
//          原始输入直接加过去
//
// 为什么重要:
//   24 层网络，如果每层都做复杂变换，信号会逐渐消失
//   残差连接让原始信息直接"跳过"子层
//   即使子层没学到有用的东西，信息也不会丢失
//
// 类比: 一条高速公路穿过城市
//   没有残差: 必须穿过每条小路 → 慢，容易迷路
//   有残差: 高速公路直通 + 可以选择在小路停留

print("--- 4. 残差连接 ---")
print("output = x + sublayer(norm(x))")
print("原始信息直接\"跳过\"子层，梯度也能顺畅回流")
print()

// ═══════════════════════════════════════════════════════
// 5. 真实模型: 跑完一整层 Transformer
// ═══════════════════════════════════════════════════════

print("--- 5. 真实模型: 完整 Transformer 层 ---")
print()

// 加载模型 (同 Step05)
let sourceFile = URL(fileURLWithPath: #file)
let learnLMRoot = sourceFile
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
let hubApi = HubApi(downloadBase: learnLMRoot)
let modelsDir = learnLMRoot.appendingPathComponent("models")
print("模型路径: \(modelsDir.path)")

let modelName = "Qwen/Qwen2.5-0.5B"
let tokenizer = try await AutoTokenizer.from(pretrained: modelName, hubApi: hubApi)
let repo = Hub.Repo(id: modelName)
let modelFolder = try await hubApi.snapshot(from: repo)

let fm = FileManager.default
let files = try fm.contentsOfDirectory(at: modelFolder, includingPropertiesForKeys: nil)
guard let safetensorsURL = files.first(where: {
    $0.pathExtension == "safetensors" && !$0.lastPathComponent.contains("index")
}) else { fatalError("找不到 safetensors") }

let fh = try FileHandle(forReadingFrom: safetensorsURL)
defer { try? fh.close() }

let hsData = fh.readData(ofLength: 8)
var hs: UInt64 = 0
for i in 0..<8 { hs |= UInt64(hsData[i]) << (i * 8) }
let hJSON = fh.readData(ofLength: Int(hs))
let header = try JSONSerialization.jsonObject(with: hJSON) as! [String: Any]
let dataBase = 8 + Int(hs)

func loadWeight(_ name: String) -> MLXArray {
    guard let info = header[name] as? [String: Any],
          let shape = info["shape"] as? [Int],
          let offsets = info["data_offsets"] as? [Int] else {
        fatalError("找不到权重: \(name)")
    }
    try! fh.seek(toOffset: UInt64(dataBase + offsets[0]))
    let data = fh.readData(ofLength: offsets[1] - offsets[0])
    let floats = data.withUnsafeBytes { ptr in
        ptr.bindMemory(to: Float16.self).map { Float($0) }
    }
    return MLXArray(floats, shape)
}

// 加载 config
let configData = try Data(contentsOf: modelFolder.appendingPathComponent("config.json"))
let config = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
let hiddenDim = config["hidden_size"] as! Int
let numHeads = config["num_attention_heads"] as! Int
let numKVHeads = config["num_key_value_heads"] as! Int
let headDim = hiddenDim / numHeads
let numLayers = config["num_hidden_layers"] as! Int

print("模型配置: \(numLayers) 层, hidden=\(hiddenDim), heads=\(numHeads), head_dim=\(headDim)")
print()

// Tokenize
let inputText = "猫追狗"
let inputIds = tokenizer.encode(text: inputText).map { $0 }
let seqLen = inputIds.count
print("输入: \"\(inputText)\" → \(inputIds)")
print()

// 读取 embeddings
let embedInfo = header["model.embed_tokens.weight"] as! [String: Any]
let embedShape = embedInfo["shape"] as! [Int]
let embedOffsets = embedInfo["data_offsets"] as! [Int]
let embedBase = dataBase + embedOffsets[0]

func readTokenVector(tokenId: Int) -> [Float] {
    let rowBytes = embedShape[1] * 2
    try! fh.seek(toOffset: UInt64(embedBase + tokenId * rowBytes))
    let data = fh.readData(ofLength: rowBytes)
    return data.withUnsafeBytes { ptr in
        ptr.bindMemory(to: Float16.self).map { Float($0) }
    }
}

var allEmbeddings = [Float]()
for id in inputIds { allEmbeddings.append(contentsOf: readTokenVector(tokenId: id)) }
var hiddenState = MLXArray(allEmbeddings, [seqLen, hiddenDim])
print("Embedding: shape=\(hiddenState.shape)")

// RoPE 函数
func applyRoPE(_ x: MLXArray, startPos: Int = 0) -> MLXArray {
    let len = x.shape[0]
    let dim = x.shape[1]
    let halfDim = dim / 2
    let freqs = (0..<halfDim).map { 1.0 / pow(10000.0, Float(2 * $0) / Float(dim)) }
    var angleValues = [Float]()
    for pos in 0..<len {
        for i in 0..<halfDim { angleValues.append(Float(startPos + pos) * freqs[i]) }
    }
    let angles = MLXArray(angleValues, [len, halfDim])
    let cosA = cos(angles)
    let sinA = sin(angles)
    let x1 = x[0..<len, 0..<halfDim]
    let x2 = x[0..<len, halfDim..<dim]
    return concatenated([x1 * cosA - x2 * sinA, x1 * sinA + x2 * cosA], axis: 1)
}

// 完整 Attention 函数
func attention(_ x: MLXArray, qW: MLXArray, kW: MLXArray, vW: MLXArray, oW: MLXArray,
               nHeads: Int, nKVHeads: Int, hDim: Int) -> MLXArray {
    let len = x.shape[0]
    let realHeadDim = hDim / nHeads
    let Q = matmul(x, qW.T).reshaped([len, nHeads, realHeadDim])
    let K = matmul(x, kW.T).reshaped([len, nKVHeads, realHeadDim])
    let V = matmul(x, vW.T).reshaped([len, nKVHeads, realHeadDim])

    var headOutputs: [MLXArray] = []
    // GQA: 如果 numHeads > numKVHeads, 多个 Q 头共享一个 KV 头
    let kvGroupSize = nHeads / nKVHeads
    for h in 0..<nHeads {
        let kvH = h / kvGroupSize
        let qh = Q[0..<len, h, 0..<realHeadDim].reshaped([len, realHeadDim])
        let kh = K[0..<len, kvH, 0..<realHeadDim].reshaped([len, realHeadDim])
        let vh = V[0..<len, kvH, 0..<realHeadDim].reshaped([len, realHeadDim])

        let rq = applyRoPE(qh)
        let rk = applyRoPE(kh)

        let scores = matmul(rq, rk.T) / sqrt(Float(realHeadDim))
        let weights = softmax(scores, axis: 1)
        headOutputs.append(matmul(weights, vh))
    }
    let concat = concatenated(headOutputs, axis: 1)
    return matmul(concat, oW.T)
}

// 加载第 0 层的全部权重并运行
let layer = 0
print()
print("=== 第 \(layer) 层完整前向传播 ===")
print()

let attnNormW = loadWeight("model.layers.\(layer).input_layernorm.weight")
let qW = loadWeight("model.layers.\(layer).self_attn.q_proj.weight")
let kW = loadWeight("model.layers.\(layer).self_attn.k_proj.weight")
let vW = loadWeight("model.layers.\(layer).self_attn.v_proj.weight")
let oW = loadWeight("model.layers.\(layer).self_attn.o_proj.weight")
let ffnNormW = loadWeight("model.layers.\(layer).post_attention_layernorm.weight")
let gateW = loadWeight("model.layers.\(layer).mlp.gate_proj.weight")
let upW = loadWeight("model.layers.\(layer).mlp.up_proj.weight")
let downW = loadWeight("model.layers.\(layer).mlp.down_proj.weight")

print("权重已加载 (Attention 4个 + FFN 3个 + Norm 2个)")

// Step 1: Attention 子层 (Pre-Norm + Attention + 残差)
let normed1 = rmsNorm(hiddenState, weight: attnNormW)
let attnOut = attention(normed1, qW: qW, kW: kW, vW: vW, oW: oW,
                        nHeads: numHeads, nKVHeads: numKVHeads, hDim: hiddenDim)
hiddenState = hiddenState + attnOut  // 残差连接!
eval(hiddenState)
print("Step 1: Attention 子层完成 (含残差), shape=\(hiddenState.shape)")

// Step 2: FFN 子层 (Pre-Norm + FFN + 残差)
let normed2 = rmsNorm(hiddenState, weight: ffnNormW)
let ffnOut = ffn(normed2, gateW: gateW, upW: upW, downW: downW)
hiddenState = hiddenState + ffnOut  // 残差连接!
eval(hiddenState)
print("Step 2: FFN 子层完成 (含残差), shape=\(hiddenState.shape)")
print()
print("第 \(layer) 层 Transformer 输出: shape=\(hiddenState.shape)")
print("  (shape 不变 — 这就是为什么可以堆叠 \(numLayers) 层)")
print()

// ═══════════════════════════════════════════════════════
// 6. lm_head — 从隐藏状态到词表概率
// ═══════════════════════════════════════════════════════
//
// 24 层 Transformer 跑完后:
//   hiddenState: [seq_len, 896] — 每个 token 位置的上下文感知向量
//
// lm_head 把这个向量投影到词表大小:
//   logits = hiddenState × lm_head^T
//   [seq, 896] × [896, 151936] = [seq, 151936]
//
// 每个 token 位置得到 151936 个分数 (logits)
// softmax 后就是概率 → 选概率最高的 → 下一个 token
//
// 注意: 这里只跑了一层，真实推理要跑全部 24 层
// 为了演示 lm_head 的效果，我们直接用第 0 层的输出

print("--- 6. lm_head 投影 ---")
print()

// 最终 RMSNorm
let finalNormW = loadWeight("model.norm.weight")
let finalNormed = rmsNorm(hiddenState, weight: finalNormW)

// lm_head 投影 (只对最后一个 token)
let lastHidden = finalNormed[finalNormed.shape[0] - 1].reshaped([1, hiddenDim])
print("最后一个 token 的隐藏状态: \(lastHidden.shape)")

// 加载 lm_head — 注意这个矩阵很大 [151936, 896]
print("正在加载 lm_head 权重 (~272MB, 转 FP32 需几秒)...")
let lmHeadW = loadWeight("lm_head.weight")
print("lm_head: \(lmHeadW.shape)")

let logits = matmul(lastHidden, lmHeadW.T)  // [1, 896] × [896, 151936] = [1, 151936]
eval(logits)
print("Logits: shape=\(logits.shape)  (每个词表 token 一个分数)")
print()

// Top-5 最可能的下一个 token
let flatLogits = logits.reshaped([logits.shape[1]])
var topIndices = [Int](0..<5)
var topValues = [Float](repeating: -.infinity, count: 5)

// 简单 top-5: 遍历 logits 找最大 5 个
// (真实推理用更高效的方法，这里为了教学清晰)
let logitsArr = (0..<logits.shape[1]).map { (i: Int) -> Float in
    let val = flatLogits[i]
    eval(val)
    return val.item(Float.self) ?? -.infinity
}

for i in logitsArr.indices {
    let v = logitsArr[i]
    let minIdx = topValues.indices.min(by: { topValues[$0] < topValues[$1] })!
    if v > topValues[minIdx] {
        topValues[minIdx] = v
        topIndices[minIdx] = i
    }
}

// 按分数排序
let sorted = zip(topIndices, topValues).sorted { $0.1 > $1.1 }
print("预测下一个 token Top-5 (只跑了 1/24 层，结果不理想是正常的):")
for (idx, val) in sorted {
    let token = tokenizer.decode(tokens: [idx])
    print("  \(String(format: "%6d", idx)): \(String(format: "%8.2f", val))  \"\(token)\"")
}
print()
print("(跑全部 24 层后预测会准确得多 — Step09 会实现完整推理)")
print()

print("=== Step06 完成! ===")
print("下一步: Step07_Sampling — 如何从 logits 选出下一个 token")
