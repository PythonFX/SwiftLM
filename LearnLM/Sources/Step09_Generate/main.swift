// Step09: Generate — 端到端文本生成，把所有步骤串起来
//
// 学习目标:
//   1. 完整的文本生成流程: tokenize → embed → transformer × N → sample → detokenize
//   2. Pre-fill + Decode 两阶段实现
//   3. 停止条件: EOS token / 最大长度
//   4. 用真实模型生成文本
//
// 运行: cd LearnLM && swift run Step09_Generate

import Foundation
import MLX
import Hub
import Tokenizers

print("=== Step09: Generate ===")
print()

// ═══════════════════════════════════════════════════════
// 1. 完整生成流程
// ═══════════════════════════════════════════════════════
//
//   用户输入: "中国的首都是"
//     │
//     ▼  Tokenizer (Step02)
//   [10483, 4517, 3091, 46321]    ← token IDs
//     │
//     ▼  Embedding 查找表 (Step04)
//   [[0.12, ...], [0.34, ...], ...]  ← [4, 896] 语义向量
//     │
//     ▼  Transformer × 24 层 (Step05-06)
//   每层: RMSNorm → Attention(RoPE) → 残差 → RMSNorm → FFN → 残差
//   输出还是 [4, 896]，但每个向量融合了完整上下文
//     │
//     ▼  lm_head 投影 (Step03)
//   [4, 151936] — 每个位置 151936 个 logits
//     │
//     ▼  取最后一个位置 (Step07)
//   [151936] → softmax → 概率 → 选一个 token
//     │
//     ▼  "北京" (token 10504)
//
// 然后把 "北京" 加到输入里，重复上述过程，直到遇到 EOS

print("--- 1. 完整流程 ---")
print("tokenize → embed → transformer×24 → lm_head → sample → detokenize")
print("重复直到: 遇到 EOS 或达到最大长度")
print()

// ═══════════════════════════════════════════════════════
// 2. 工具函数 — 从前面步骤复用
// ═══════════════════════════════════════════════════════

func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let x2 = x * x
    let lastAxis = x.shape.count - 1
    let meanX2 = mean(x2, axis: lastAxis)
    let rms = sqrt(meanX2 + MLXArray(Float(eps)))
    return x / rms.expandedDimensions(axis: lastAxis) * weight
}

func silu(_ x: MLXArray) -> MLXArray {
    return x * sigmoid(x)
}

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

// ═══════════════════════════════════════════════════════
// 3. 加载模型
// ═══════════════════════════════════════════════════════

print("--- 3. 加载模型 ---")

let sourceFile = URL(fileURLWithPath: #file)
let learnLMRoot = sourceFile
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
let hubApi = HubApi(downloadBase: learnLMRoot)
let modelsDir = learnLMRoot.appendingPathComponent("models")
print("模型路径: \(modelsDir.path)")

let modelName = "Qwen/Qwen2.5-0.5B"
print("模型: \(modelName)")
print()

let tokenizer = try await AutoTokenizer.from(pretrained: modelName, hubApi: hubApi)
let repo = Hub.Repo(id: modelName)
let modelFolder = try await hubApi.snapshot(from: repo)

// 解析 safetensors
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

// Config
let configData = try Data(contentsOf: modelFolder.appendingPathComponent("config.json"))
let config = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
let hiddenDim = config["hidden_size"] as! Int
let numHeads = config["num_attention_heads"] as! Int
let numKVHeads = config["num_key_value_heads"] as! Int
let headDim = hiddenDim / numHeads
let numLayers = config["num_hidden_layers"] as! Int

print("配置: \(numLayers) 层, hidden=\(hiddenDim), heads=\(numHeads)/kv=\(numKVHeads)")

// ═══════════════════════════════════════════════════════
// 4. 加载全部权重
// ═══════════════════════════════════════════════════════

print()
print("正在加载权重...")

// Embedding: 只存权重矩阵，按需查行
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

// 所有层的权重
struct LayerWeights {
    let attnNormW: MLXArray
    let qW, kW, vW, oW: MLXArray
    let ffnNormW: MLXArray
    let gateW, upW, downW: MLXArray
}

var layers: [LayerWeights] = []
for l in 0..<numLayers {
    layers.append(LayerWeights(
        attnNormW: loadWeight("model.layers.\(l).input_layernorm.weight"),
        qW: loadWeight("model.layers.\(l).self_attn.q_proj.weight"),
        kW: loadWeight("model.layers.\(l).self_attn.k_proj.weight"),
        vW: loadWeight("model.layers.\(l).self_attn.v_proj.weight"),
        oW: loadWeight("model.layers.\(l).self_attn.o_proj.weight"),
        ffnNormW: loadWeight("model.layers.\(l).post_attention_layernorm.weight"),
        gateW: loadWeight("model.layers.\(l).mlp.gate_proj.weight"),
        upW: loadWeight("model.layers.\(l).mlp.up_proj.weight"),
        downW: loadWeight("model.layers.\(l).mlp.down_proj.weight")
    ))
}
let finalNormW = loadWeight("model.norm.weight")
print("lm_head 加载中 (~272MB, 需几秒)...")
let lmHeadW = loadWeight("lm_head.weight")

print("全部权重加载完成! (\(numLayers) 层 + embedding + lm_head)")
print()

// ═══════════════════════════════════════════════════════
// 5. 前向传播函数
// ═══════════════════════════════════════════════════════

func attention(_ x: MLXArray, lw: LayerWeights, startPos: Int) -> MLXArray {
    let len = x.shape[0]
    let Q = matmul(x, lw.qW.T).reshaped([len, numHeads, headDim])
    let K = matmul(x, lw.kW.T).reshaped([len, numKVHeads, headDim])
    let V = matmul(x, lw.vW.T).reshaped([len, numKVHeads, headDim])

    var headOutputs: [MLXArray] = []
    let kvGroupSize = numHeads / numKVHeads
    for h in 0..<numHeads {
        let kvH = h / kvGroupSize
        let qh = Q[0..<len, h, 0..<headDim].reshaped([len, headDim])
        let kh = K[0..<len, kvH, 0..<headDim].reshaped([len, headDim])
        let vh = V[0..<len, kvH, 0..<headDim].reshaped([len, headDim])

        let rq = applyRoPE(qh, startPos: startPos)
        let rk = applyRoPE(kh, startPos: startPos)

        let scores = matmul(rq, rk.T) / sqrt(Float(headDim))

        // Causal mask: 只能看到当前位置及之前的位置
        if len > 1 {
            let mask = MLXArray((0..<len).flatMap { i in
                (0..<len).map { j in Float(j <= i ? 0.0 : -.infinity) }
            }, [len, len])
            let maskedScores = scores + mask
            let weights = softmax(maskedScores, axis: 1)
            headOutputs.append(matmul(weights, vh))
        } else {
            let weights = softmax(scores, axis: 1)
            headOutputs.append(matmul(weights, vh))
        }
    }
    let concat = concatenated(headOutputs, axis: 1)
    return matmul(concat, lw.oW.T)
}

func ffn(_ x: MLXArray, lw: LayerWeights) -> MLXArray {
    let gate = silu(matmul(x, lw.gateW.T))
    let up = matmul(x, lw.upW.T)
    return matmul(gate * up, lw.downW.T)
}

// 完整 forward: 输入 token IDs → 输出 logits
func forward(_ tokenIds: [Int], startPos: Int = 0) -> MLXArray {
    // Embedding lookup
    var allEmb = [Float]()
    for id in tokenIds { allEmb.append(contentsOf: readTokenVector(tokenId: id)) }
    var hidden = MLXArray(allEmb, [tokenIds.count, hiddenDim])

    // Transformer layers
    for lw in layers {
        let normed1 = rmsNorm(hidden, weight: lw.attnNormW)
        let attnOut = attention(normed1, lw: lw, startPos: startPos)
        hidden = hidden + attnOut

        let normed2 = rmsNorm(hidden, weight: lw.ffnNormW)
        let ffnOut = ffn(normed2, lw: lw)
        hidden = hidden + ffnOut
    }

    // Final norm
    hidden = rmsNorm(hidden, weight: finalNormW)

    // lm_head (只对最后一个 token)
    let lastHidden = hidden[hidden.shape[0] - 1].reshaped([1, hiddenDim])
    return matmul(lastHidden, lmHeadW.T)  // [1, vocab_size]
}

// ═══════════════════════════════════════════════════════
// 6. 文本生成!
// ═══════════════════════════════════════════════════════

print("--- 6. 文本生成 ---")
print()

func generate(prompt: String, maxTokens: Int = 30, temperature: Float = 0.7) {
    print("提示: \"\(prompt)\"")
    print()

    // Tokenize
    var tokenIds = tokenizer.encode(text: prompt).map { $0 }
    print("Token IDs: \(tokenIds)")

    var generated = ""
    let eosId = tokenizer.eosTokenId

    for step in 0..<maxTokens {
        // Forward pass
        let logits = forward(tokenIds, startPos: 0)
        let flatLogits = logits.reshaped([logits.shape[1]])

        // Temperature + argMax (simplified greedy for reliability)
        let nextId: Int
        if temperature < 0.01 {
            // Pure greedy
            let idx = argMax(flatLogits)
            eval(idx)
            nextId = idx.item(Int.self)
        } else {
            // Temperature sampling
            let scaled = flatLogits / MLXArray(temperature)
            let probs = softmax(scaled, axis: 0)
            // Use argMax as approximation (true sampling needs categorical distribution)
            let idx = argMax(probs)
            eval(idx)
            nextId = idx.item(Int.self)
        }

        // Check EOS
        if nextId == eosId {
            print("  [EOS] 生成结束")
            break
        }

        let tokenStr = tokenizer.decode(tokens: [nextId])
        generated += tokenStr
        print("  Step \(step + 1): token \(nextId) → \"\(tokenStr)\"")

        // Append for next iteration
        tokenIds.append(nextId)

        // Safety: prevent infinite loops
        if step >= maxTokens - 1 {
            print("  [达到最大长度 \(maxTokens)]")
        }
    }

    print()
    print("完整输出: \"\(prompt)\(generated)\"")
    print()
}

// 运行生成!
generate(prompt: "中国的首都是", maxTokens: 15)
print("─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─")
print()
generate(prompt: "1+1=", maxTokens: 10)

// ═══════════════════════════════════════════════════════
// 7. 总结 — 你已经学会了什么
// ═══════════════════════════════════════════════════════
//
// Step01: MLXArray, GPU计算, 矩阵乘法
// Step02: Tokenizer, 文字 ↔ 数字
// Step03: Safetensors, 模型文件结构, 权重名映射
// Step04: Embedding, token ID → 语义向量, 余弦相似度
// Step05: Attention, Q/K/V, 多头注意力, RoPE
// Step06: Transformer Block, RMSNorm, FFN, 残差连接
// Step07: Sampling, Temperature, Top-k, Top-p
// Step08: KV Cache, Pre-fill/Decode, 内存分析
// Step09: Generate, 端到端推理
//
// 你现在理解了 LLM 推理的完整数据流:
//   文字 → Tokenize → Embed → Transformer×N → Sample → Detokenize → 文字
//
// 接下来可以:
//   - 优化: 量化(GPTQ/AWQ)、KV Cache实现、批量推理
//   - 扩展: MoE模型(SSD offload)、多模态(VLM)、长上下文
//   - 部署: SwiftLM 服务器、SwiftBuddy iOS app

print("=== Step09 完成! ===")
print()
print("恭喜! 你已经从零理解了 LLM 推理的完整流程:")
print("  Step01 MLX基础 → Step02 Tokenizer → Step03 模型结构")
print("  → Step04 Embedding → Step05 Attention → Step06 Transformer")
print("  → Step07 Sampling → Step08 KVCache → Step09 Generate")
print()
print("下一步: 阅读 SwiftLM 源码 (Sources/SwiftLM/Server.swift)")
print("  学习生产级推理服务器的实现!")
