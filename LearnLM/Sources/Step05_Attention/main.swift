// Step05: Attention — 让 token 之间交换信息
//
// 学习目标:
//   1. 为什么需要 Attention — embedding 只知道"我是谁"，不知道上下文
//   2. Q, K, V 投影 — 三种不同的线性变换
//   3. 缩放点积注意力 (Scaled Dot-Product Attention)
//   4. 多头注意力 (Multi-Head Attention)
//   5. RoPE 旋转位置编码
//   6. 用真实模型权重运行 Attention
//
// 运行: cd LearnLM && swift run Step05_Attention

import Foundation
import MLX
import Hub
import Tokenizers

print("=== Step05: Attention ===")
print()

// ═══════════════════════════════════════════════════════
// 1. 为什么需要 Attention
// ═══════════════════════════════════════════════════════
//
// Step04 结束时，每个 token 有了独立的语义向量 (Embedding):
//   "猫":  [0.12, -0.34, ..., 0.78]  (896维)
//   "追":  [0.56,  0.23, ..., 0.45]  (896维)
//   "狗":  [-0.11, 0.67, ..., 0.34]  (896维)
//
// 问题: 这三个向量完全独立！"猫"不知道自己被"追"，"狗"不知道"猫"在追自己
//
// Attention 做的事: 让每个 token "看" 到所有其他 token
//   根据相关性，把其他 token 的信息融合进自己的向量
//   "猫"的新向量 = 我是谁 + 上下文（我被追、追的是狗）

print("--- 1. 为什么需要 Attention ---")
print("Embedding: 每个 token 只知道\"我是谁\"")
print("Attention:  每个 token 融合\"其他人是谁\" → 上下文感知")
print()

// ═══════════════════════════════════════════════════════
// 2. Q, K, V 投影 — 三种视角看同一个输入
// ═══════════════════════════════════════════════════════
//
// 回顾 Step03: 每个注意力层有 4 个权重矩阵 W_q, W_k, W_v, W_o
//
// 对输入 X (每个 token 的向量) 做三次不同的线性变换:
//   Q = X × W_q   "我在找什么？" (Query — 查询)
//   K = X × W_k   "我有什么特征？" (Key — 被查的索引)
//   V = X × W_v   "我的实际内容" (Value — 被传递的信息)
//
// 同一条输入，乘不同矩阵，得到三种不同视角
//   W_q 让 "猫" 学会问: "谁在做动作？谁是被追的？"
//   W_k 让 "狗" 学会表达: "我是可以被追的对象"
//   W_v 让 "狗" 学会传递: "我的语义内容是动物"

print("--- 2. Q, K, V 投影 (Mini Demo) ---")
print()

// 3 个 token，每个 6 维
let X = MLXArray([Float]([
     1.0,  0.5,  0.2, -0.3,  0.8,  0.1,   // token 0: "猫"
     0.3, -0.1,  0.9,  0.4, -0.2,  0.6,   // token 1: "追"
    -0.5,  0.7, -0.4,  0.1,  0.3,  0.9,   // token 2: "狗"
]), [3, 6])

// 模拟 W_q, W_k, W_v (6×6 矩阵，真实模型是 896×896)
let Wq = MLXArray([Float]([
     0.4, -0.2,  0.3,  0.1, -0.5,  0.2,
    -0.1,  0.6, -0.3,  0.4,  0.2, -0.1,
     0.3,  0.1,  0.5, -0.2,  0.4, -0.3,
    -0.4,  0.3, -0.1,  0.6, -0.2,  0.5,
     0.2, -0.4,  0.2,  0.3,  0.1, -0.6,
    -0.3,  0.5, -0.4,  0.1,  0.6,  0.2,
]), [6, 6])
let Wk = MLXArray([Float]([
     0.1,  0.3, -0.5,  0.2,  0.4, -0.1,
     0.5, -0.2,  0.3, -0.4,  0.1,  0.6,
    -0.3,  0.4,  0.2,  0.5, -0.1, -0.3,
     0.2, -0.1,  0.6,  0.3, -0.5,  0.4,
    -0.4,  0.5, -0.2,  0.1,  0.3, -0.6,
     0.6, -0.3,  0.1, -0.2,  0.4,  0.5,
]), [6, 6])
let Wv = MLXArray([Float]([
     0.3,  0.2, -0.4,  0.5, -0.1,  0.3,
    -0.2,  0.4,  0.1, -0.3,  0.6, -0.2,
     0.5, -0.1,  0.3,  0.2, -0.4,  0.6,
    -0.3,  0.6, -0.2,  0.4,  0.1, -0.5,
     0.1, -0.3,  0.5, -0.1,  0.2,  0.4,
    -0.4,  0.2,  0.6, -0.5,  0.3, -0.1,
]), [6, 6])

// 线性投影: Q = X @ W_q^T (Step03 讲过: 权重是 [out, in]，所以要转置)
let Q = matmul(X, Wq.T)
let K = matmul(X, Wk.T)
let V = matmul(X, Wv.T)

eval(Q); eval(K); eval(V)

print("输入 X:  shape=\(X.shape)  [3 tokens × 6 dim]")
print("Q (Query): shape=\(Q.shape)  \"我在找什么\"")
print("K (Key):   shape=\(K.shape)  \"我有什么特征\"")
print("V (Value): shape=\(V.shape)  \"我的实际内容\"")
print()

// ═══════════════════════════════════════════════════════
// 3. 缩放点积注意力 — 谁和谁相关
// ═══════════════════════════════════════════════════════
//
// 核心公式:
//   Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
//
// 分步:
//   1. scores = Q × K^T          → 每个 token 对其他 token 的"关注度"分数
//   2. scaled = scores / √d_k    → 缩放，防止梯度消失
//   3. weights = softmax(scaled) → 归一化为概率分布 (每行加起来=1)
//   4. output = weights × V      → 用概率加权 V，得到融合了上下文的向量
//
// 直觉 (以 "猫追狗" 为例):
//   scores 是 3×3 矩阵，scores[i][j] = token_i 对 token_j 的关注分数
//   softmax 后:
//     "猫" 看 "猫" 0.15, 看 "追" 0.55, 看 "狗" 0.30
//     "追" 看 "猫" 0.40, 看 "追" 0.10, 看 "狗" 0.50
//     → "猫" 最关注 "追" (55%)，因为猫和追关系最密切
//   output = 加权 V → "猫"的新向量融合了"追"和"狗"的信息

print("--- 3. 缩放点积注意力 ---")
print()

// Step 1: scores = Q @ K^T
let scores = matmul(Q, K.T)
print("Step 1: scores = Q × K^T  shape=\(scores.shape)")
print("  \(scores)")
print()

// Step 2: 缩放
let dK = Float(Q.shape[1])
let scaledScores = scores / sqrt(dK)
print("Step 2: scores / √\(Int(dK)) = scores / \(String(format: "%.2f", sqrt(dK)))")
print()

// Step 3: softmax → 注意力权重 (每行加起来=1)
let attnWeights = softmax(scaledScores, axis: 1)
eval(attnWeights)
print("Step 3: softmax → 注意力权重")
print("  \(attnWeights)")
print("  (每行是一个 token 对所有 token 的关注度，加起来=1)")
print()

// Step 4: output = weights × V
let attnOutput = matmul(attnWeights, V)
eval(attnOutput)
print("Step 4: output = weights × V  shape=\(attnOutput.shape)")
print("  \(attnOutput)")
print("  (每个 token 的向量现在融合了其他 token 的信息)")
print()

// ═══════════════════════════════════════════════════════
// 4. 多头注意力 — 从多个角度看关系
// ═══════════════════════════════════════════════════════
//
// 单头注意力: 整个 6 维用一个头 → 只能捕捉一种关系
// 多头注意力: 把 6 维拆成 2 个头 × 3 维 → 每个头关注不同关系
//
// Qwen2.5-0.5B: 896 维 → 14 头 × 64 维/头
//
// 为什么多头:
//   头 1 可能学到语法关系 (主语-谓语)
//   头 2 可能学到语义关系 (动物-动作)
//   头 3 可能学到位置关系 (相邻词)
//   多个头综合 → 更丰富的上下文理解

print("--- 4. 多头注意力 (Mini Demo) ---")
print()

let numHeads = 2
let headDim = 3  // 6 / 2 = 3

// 切分多头: [3, 6] → [3, 2, 3] (3 tokens, 2 heads, 3 dim/head)
let Qm = Q.reshaped([3, numHeads, headDim])
let Km = K.reshaped([3, numHeads, headDim])
let Vm = V.reshaped([3, numHeads, headDim])

print("Q 切分多头: \(Q.shape) → \(Qm.shape)  [tokens, heads, dim/head]")
print()

// 每个头独立计算注意力
var headOutputs: [MLXArray] = []
for h in 0..<numHeads {
    let qh = Qm[0..., h, 0..<headDim].reshaped([3, headDim])
    let kh = Km[0..., h, 0..<headDim].reshaped([3, headDim])
    let vh = Vm[0..., h, 0..<headDim].reshaped([3, headDim])

    let s = matmul(qh, kh.T) / sqrt(Float(headDim))
    let w = softmax(s, axis: 1)
    let out = matmul(w, vh)
    eval(out)
    headOutputs.append(out)
    print("  Head \(h): attention output shape=\(out.shape)")
    print("    \(out)")
}
print()

// 拼接多头: [3, 3] × 2 → [3, 6]
let multiHeadOut = concatenated(headOutputs, axis: 1)
print("拼接多头: shape=\(multiHeadOut.shape)")
print()

// 输出投影: output = concat_heads × W_o
let Wo = MLXArray([Float]([
     0.2, -0.3,  0.5,  0.1, -0.4,  0.3,
     0.4,  0.1, -0.2,  0.5,  0.3, -0.1,
    -0.3,  0.4,  0.1, -0.2,  0.6,  0.2,
     0.1, -0.5,  0.3,  0.4, -0.1,  0.6,
    -0.5,  0.2, -0.4,  0.3,  0.1,  0.4,
     0.3,  0.6, -0.1, -0.3,  0.2,  0.5,
]), [6, 6])
let projOutput = matmul(multiHeadOut, Wo.T)
eval(projOutput)
print("输出投影 W_o: \(multiHeadOut.shape) → \(projOutput.shape)")
print()

// ═══════════════════════════════════════════════════════
// 5. RoPE — 旋转位置编码
// ═══════════════════════════════════════════════════════
//
// 问题: Attention 本身不知道 token 的位置
//   "猫追狗" 和 "狗追猫" 的 Q×K^T 结果相同 (只要词一样)
//
// RoPE 解决: 对 Q 和 K 做旋转变换，旋转角度 = 位置 × 频率
//
// 原理 (2 维为例):
//   位置 0 的向量 → 不旋转
//   位置 1 的向量 → 旋转 θ
//   位置 2 的向量 → 旋转 2θ
//
//   旋转后 Q·K 的点积只取决于相对位置 (位置差)
//   不取决于绝对位置 → 天然的"相对位置编码"
//
// 实际实现: 896 维每 2 维一对 (共 448 对)
//   每对的旋转频率不同: 前面的对频率高 (捕捉相邻位置)
//                         后面的对频率低 (捕捉远距离位置)

print("--- 5. RoPE 旋转位置编码 ---")
print()

func applyRoPE(_ x: MLXArray, startPos: Int = 0) -> MLXArray {
    // x: [seqLen, headDim] (单头)
    let seqLen = x.shape[0]
    let dim = x.shape[1]
    let halfDim = dim / 2

    // 频率: θ_i = 1 / (10000^(2i/d))
    let freqs = (0..<halfDim).map { i -> Float in
        1.0 / pow(10000.0, Float(2 * i) / Float(dim))
    }

    // 位置 × 频率 → 角度
    // angles[seq][pair] = (startPos + seq) * freqs[pair]
    var angleValues = [Float]()
    for pos in 0..<seqLen {
        for i in 0..<halfDim {
            angleValues.append(Float(startPos + pos) * freqs[i])
        }
    }
    let angles = MLXArray(angleValues, [seqLen, halfDim])
    let cosA = cos(angles)
    let sinA = sin(angles)

    // 把 x 分成前后两半: x1 = 前 halfDim, x2 = 后 halfDim
    let x1 = x[0..<seqLen, 0..<halfDim]
    let x2 = x[0..<seqLen, halfDim..<dim]

    // 旋转: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = x1 * cosA - x2 * sinA
    let r2 = x1 * sinA + x2 * cosA

    return concatenated([r1, r2], axis: 1)
}

// Mini demo: 3 个 token，4 维，看 RoPE 如何影响点积
let demoQ = MLXArray([Float]([1.0, 0.0, 0.5, -0.3]), [1, 4])
print("位置 0 的 Q 向量: \(demoQ)")
let ropeQ0 = applyRoPE(demoQ, startPos: 0)
let ropeQ1 = applyRoPE(demoQ, startPos: 1)
let ropeQ5 = applyRoPE(demoQ, startPos: 5)
eval(ropeQ0); eval(ropeQ1); eval(ropeQ5)

print("旋转后:")
print("  位置 0: \(ropeQ0)")
print("  位置 1: \(ropeQ1)")
print("  位置 5: \(ropeQ5)")

// 同一个向量，不同位置 → 旋转后不同 → 点积不同
let dot01 = sum(ropeQ0 * ropeQ1)
let dot05 = sum(ropeQ0 * ropeQ5)
eval(dot01); eval(dot05)
print("  Q(pos0)·Q(pos1) = \(dot01)  ← 相邻，点积高")
print("  Q(pos0)·Q(pos5) = \(dot05)  ← 远离，点积低")
print("  (RoPE 让相近位置的向量更相关)")
print()

// ═══════════════════════════════════════════════════════
// 6. 真实模型 Attention — 用 Qwen2.5-0.5B 的权重
// ═══════════════════════════════════════════════════════

print("--- 6. 真实模型 Attention ---")
print()

// 加载 Tokenizer + 模型
let sourceFile = URL(fileURLWithPath: #file)
let learnLMRoot = sourceFile
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
let hubApi = HubApi(downloadBase: learnLMRoot)
let modelsDir = learnLMRoot.appendingPathComponent("models")
print("模型下载路径: \(modelsDir.path)")

let modelName = "Qwen/Qwen2.5-0.5B"
let tokenizer = try await AutoTokenizer.from(pretrained: modelName, hubApi: hubApi)
let repo = Hub.Repo(id: modelName)
let modelFolder = try await hubApi.snapshot(from: repo)

let fm = FileManager.default
let files = try fm.contentsOfDirectory(at: modelFolder, includingPropertiesForKeys: nil)
guard let safetensorsURL = files.first(where: {
    $0.pathExtension == "safetensors" && !$0.lastPathComponent.contains("index")
}) else { fatalError("找不到 safetensors") }

// 解析 safetensors header
let fh = try FileHandle(forReadingFrom: safetensorsURL)
defer { try? fh.close() }

let hsData = fh.readData(ofLength: 8)
var hs: UInt64 = 0
for i in 0..<8 { hs |= UInt64(hsData[i]) << (i * 8) }
let hJSON = fh.readData(ofLength: Int(hs))
let header = try JSONSerialization.jsonObject(with: hJSON) as! [String: Any]
let dataBase = 8 + Int(hs)

// 通用权重加载函数
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

// 加载 config 参数
let configURL = modelFolder.appendingPathComponent("config.json")
let configData = try Data(contentsOf: configURL)
let config = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
let hiddenDim = config["hidden_size"] as! Int
let numAttnHeads = config["num_attention_heads"] as! Int
let numKVHeads = config["num_key_value_heads"] as! Int
let headDimVal = hiddenDim / numAttnHeads
let numRealLayers = config["num_hidden_layers"] as! Int
print("config: hidden=\(hiddenDim), heads=\(numAttnHeads), kv_heads=\(numKVHeads), head_dim=\(headDimVal)")
print()

// Tokenize 输入
let inputText = "猫追狗"
let inputIds = tokenizer.encode(text: inputText).map { $0 }
print("输入: \"\(inputText)\" → token IDs: \(inputIds)")
print()

// 读取 token embedding (Step04 方法)
let embedKey = "model.embed_tokens.weight"
let embedInfo = header[embedKey] as! [String: Any]
let embedShape = embedInfo["shape"] as! [Int]
let embedOffsets = embedInfo["data_offsets"] as! [Int]
let embedBaseOffset = dataBase + embedOffsets[0]

func readTokenVector(tokenId: Int) -> [Float] {
    let rowBytes = embedShape[1] * 2
    try! fh.seek(toOffset: UInt64(embedBaseOffset + tokenId * rowBytes))
    let data = fh.readData(ofLength: rowBytes)
    return data.withUnsafeBytes { ptr in
        ptr.bindMemory(to: Float16.self).map { Float($0) }
    }
}

// 构建 embedding 矩阵 [seqLen, hiddenDim]
var allEmbeddings = [Float]()
for id in inputIds {
    allEmbeddings.append(contentsOf: readTokenVector(tokenId: id))
}
let embeddings = MLXArray(allEmbeddings, [inputIds.count, hiddenDim])
print("Embedding: shape=\(embeddings.shape)")
print()

// 加载第 0 层的 Attention 权重
let layer = 0
let normWeight = loadWeight("model.layers.\(layer).input_layernorm.weight")
let qWeight = loadWeight("model.layers.\(layer).self_attn.q_proj.weight")
let kWeight = loadWeight("model.layers.\(layer).self_attn.k_proj.weight")
let vWeight = loadWeight("model.layers.\(layer).self_attn.v_proj.weight")
let oWeight = loadWeight("model.layers.\(layer).self_attn.o_proj.weight")

print("第\(layer)层 Attention 权重已加载:")
print("  q_proj: \(qWeight.shape), k_proj: \(kWeight.shape)")
print("  v_proj: \(vWeight.shape), o_proj: \(oWeight.shape)")
print()

// RMSNorm (Step06 会详细讲)
func rmsNorm(_ x: MLXArray, weight: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let x2 = x * x
    let lastAxis = x.shape.count - 1
    let meanX2 = mean(x2, axis: lastAxis)
    let rms = sqrt(meanX2 + MLXArray(Float(eps)))
    return x / rms.expandedDimensions(axis: lastAxis) * weight
}

// Step 1: RMSNorm
let normed = rmsNorm(embeddings, weight: normWeight)
eval(normed)
print("RMSNorm: \(embeddings.shape) → \(normed.shape)")

// Step 2: Q, K, V 投影
let realQ = matmul(normed, qWeight.T)
let realK = matmul(normed, kWeight.T)
let realV = matmul(normed, vWeight.T)
eval(realQ); eval(realK); eval(realV)
print("Q, K, V 投影: → Q=\(realQ.shape), K=\(realK.shape), V=\(realV.shape)")

// Step 3: 切分多头 + RoPE
let seqLen = inputIds.count
let realQm = realQ.reshaped([seqLen, numAttnHeads, headDimVal])
let realKm = realK.reshaped([seqLen, numKVHeads, headDimVal])
let realVm = realV.reshaped([seqLen, numKVHeads, headDimVal])

// 对每个头的 Q 和 K 应用 RoPE
var ropeQList: [MLXArray] = []
var ropeKList: [MLXArray] = []
for h in 0..<numKVHeads {
    let qh = realQm[0..<seqLen, h, 0..<headDimVal].reshaped([seqLen, headDimVal])
    let kh = realKm[0..<seqLen, h, 0..<headDimVal].reshaped([seqLen, headDimVal])
    ropeQList.append(applyRoPE(qh))
    ropeKList.append(applyRoPE(kh))
}
print("RoPE: Q 和 K 旋转完成 (每个头独立)")

// Step 4: 多头注意力计算 (用头 0 展示)
let q0 = ropeQList[0]
let k0 = ropeKList[0]
let v0 = realVm[0..<seqLen, 0, 0..<headDimVal].reshaped([seqLen, headDimVal])

let realScores = matmul(q0, k0.T) / sqrt(Float(headDimVal))
let realAttn = softmax(realScores, axis: 1)
eval(realAttn)
print()
print("Head 0 的注意力权重 (3×3):")
print("         \"\(inputText[inputText.index(inputText.startIndex, offsetBy: 0)])\""
    + "     \"\(inputText[inputText.index(inputText.startIndex, offsetBy: 1)])\""
    + "     \"\(inputText[inputText.index(inputText.startIndex, offsetBy: 2)])\"")
let tokens = Array(inputText)
for i in 0..<seqLen {
    let row = realAttn[i]
    eval(row)
    print("  \"\(tokens[i])\"   \(row)")
}
print()

print("=== Step05 完成! ===")
print("下一步: Step06_Transformer — 完整的 Transformer 层 (Attention + FFN + 残差)")
