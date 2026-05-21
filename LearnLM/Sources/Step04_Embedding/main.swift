// Step04: Embedding — 把 token ID 变成语义向量
//
// 学习目标:
//   1. Embedding 是什么 — 为什么需要把整数变成向量
//   2. Embedding 本质是查找表 (lookup table)
//   3. 余弦相似度 — 衡量向量语义相似程度
//   4. 从真实模型加载 Embedding 权重
//   5. 验证: 语义相近的 token 向量确实相近
//   6. 位置编码 (Positional Encoding) 的概念
//
// 运行: cd LearnLM && swift run Step04_Embedding

import Foundation
import MLX
import Hub
import Tokenizers

print("=== Step04: Embedding ===")
print()

// ═══════════════════════════════════════════════════════
// 1. 为什么需要 Embedding
// ═══════════════════════════════════════════════════════
//
// 回顾 Step02: Tokenizer 把文字变成整数 ID
//   "你好世界" → [8848, 1039, 4517, 3091]
//
// 问题: 这些整数没有数学意义
//   8848 > 1039 不代表 "你" 比 "好" 更重要
//   8848 - 1039 = 7809 这个差值毫无含义
//   但 LLM 的所有计算都是矩阵乘法和加法
//   整数 ID 没法做有意义的矩阵运算
//
// Embedding 的作用:
//   把每个整数 ID 映射到一个高维浮点向量
//   这些向量在训练过程中学习 — 语义相近的 token 自动靠近
//
//   "猫" → [0.12, -0.34, 0.56, ..., 0.78]  (896维)
//   "狗" → [0.15, -0.30, 0.52, ..., 0.81]  (896维)
//          ↑ 语义相近，向量也相近
//
//   "猫"   → [0.12, -0.34, 0.56, ..., 0.78]  (896维)
//   "桌子" → [-0.45, 0.23, -0.67, ..., 0.12] (896维)
//          ↑ 语义无关，向量距离远

print("--- 1. 为什么需要 Embedding ---")
print()
print("Tokenizer 输出: 整数 ID (没有数学意义)")
print("Embedding 输出: 浮点向量 (有语义信息)")
print("核心: 语义相近的 token → 向量距离近")
print()

// ═══════════════════════════════════════════════════════
// 2. Embedding 的数学本质 — 查找表
// ═══════════════════════════════════════════════════════
//
// Embedding 就是一个二维矩阵 E [vocab_size, hidden_dim]
//   Qwen2.5-0.5B: E [151936, 896]
//
//   每行是一个 token 的向量:
//     E[0]     = [0.12, -0.34, ...]  ← token 0
//     E[8848]  = [0.23, -0.11, ...]  ← token 8848 ("你")
//     E[1039]  = [0.45,  0.67, ...]  ← token 1039 ("好")
//
//   查找操作 = 数组索引: embedding(id) = E[id]
//   没有乘法！就是取一行出来
//
//   批量查找 (推理时的实际操作):
//     input_ids = [8848, 1039]       ← 两个 token
//     result = E[input_ids]          ← 取两行
//     → shape [2, 896]              ← 两个 896 维向量
//
//   896 维看起来很多，但这些维度各自编码不同侧面:
//     某几维 → 词性 (名词/动词/形容词)
//     某几维 → 语义类别 (动物/颜色/数字)
//     某几维 → 情感倾向 (正面/负面/中性)
//     大部分维度 → 人类无法解释的隐含特征
//     没有哪一维是手动设计的 — 全部从训练中自动涌现

print("--- 2. 数学本质: 查找表 ---")
print("E [vocab_size, hidden_dim] — 每行是一个 token 的向量")
print("embedding(id) = E[id] — 数组索引，没有乘法")
print()

// ═══════════════════════════════════════════════════════
// 3. Mini 演示: 手动 Embedding 查找
// ═══════════════════════════════════════════════════════
//
// 用一个 6×4 的小矩阵模拟 Embedding
// (真实模型是 151936×896，原理完全一样)

print("--- 3. Mini 演示 ---")
print()

let miniEmb = MLXArray([Float]([
     0.8,  0.9,  0.1,  0.2,   // token 0: "猫"
     0.7,  0.8,  0.2,  0.3,   // token 1: "狗"
     0.1,  0.2,  0.9,  0.8,   // token 2: "跑"
     0.2,  0.3,  0.1,  0.2,   // token 3: "桌子"
     0.6,  0.5,  0.8,  0.9,   // token 4: "开心"
     0.5,  0.4,  0.7,  0.8,   // token 5: "快乐"
]), [6, 4])

print("Mini Embedding [6 tokens × 4 dim]:")
print("  \(miniEmb)")
print()

// 单个查找: token 0 ("猫")
let vCat = miniEmb[0]
let vDog = miniEmb[1]
let vTable = miniEmb[3]
print("token 0 (\"猫\"):   \(vCat)")
print("token 1 (\"狗\"):   \(vDog)")
print("token 3 (\"桌子\"): \(vTable)")
print("  \"猫\" 和 \"狗\" 的数值接近 → 向量相近")
print("  \"猫\" 和 \"桌子\" 的数值差异大 → 向量距离远")
print()

// ═══════════════════════════════════════════════════════
// 4. 余弦相似度 — 量化"向量有多近"
// ═══════════════════════════════════════════════════════
//
// 直觉: 两个向量的"方向"越一致，相似度越高
//        向量长度不重要，只看方向
//
// 公式: cos_sim(a, b) = (a·b) / (|a| × |b|)
//
//   a·b = Σ(a[i] × b[i])       — 点积 (衡量方向一致性)
//   |a| = √(Σ(a[i]²))          — 向量长度 (L2范数)
//
// 结果范围 [-1, 1]:
//   1.0  → 方向完全相同 (最相似)
//   0.0  → 完全正交 (无关)
//  -1.0  → 方向完全相反
//
// 为什么不直接用欧氏距离?
//   因为向量长度受词频影响 — 高频词的向量通常更长
//   余弦相似度归一化了长度，只比较方向
//   语义关系主要由方向决定，不是长度

print("--- 4. 余弦相似度 ---")
print()

func cosineSim(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
    let dotProduct = sum(a * b)
    let normA = sqrt(sum(a * a))
    let normB = sqrt(sum(b * b))
    return dotProduct / (normA * normB)
}

// Mini demo (手动设置的数据，模拟语义关系)
let simCatDog = cosineSim(vCat, vDog)
let simCatTable = cosineSim(vCat, vTable)
let simHappyJoy = cosineSim(miniEmb[4], miniEmb[5])

eval(simCatDog)
eval(simCatTable)
eval(simHappyJoy)

print("(手动设置的数据，模拟真实语义关系)")
print("  \"猫\" vs \"狗\":   \(simCatDog)  ← 相似 (都是动物)")
print("  \"猫\" vs \"桌子\": \(simCatTable)  ← 不相似 (无关)")
print("  \"开心\" vs \"快乐\": \(simHappyJoy)  ← 很相似 (同义词)")
print()

// ═══════════════════════════════════════════════════════
// 5. 加载真实 Tokenizer + 模型
// ═══════════════════════════════════════════════════════

print("--- 5. 加载真实模型 ---")
print()

// Models 目录 — 所有步骤共用同一个下载路径: LearnLM/models/
// HubApi 内部会加 repo.type 前缀 ("models")，所以 downloadBase 设为 LearnLM/
// 最终路径: LearnLM/models/Qwen/Qwen2.5-0.5B/
let sourceFile = URL(fileURLWithPath: #file)
let learnLMRoot = sourceFile
    .deletingLastPathComponent()  // Step04_Embedding
    .deletingLastPathComponent()  // Sources
    .deletingLastPathComponent()  // LearnLM

let hubApi = HubApi(downloadBase: learnLMRoot)

let modelName = "Qwen/Qwen2.5-0.5B"
let modelsDir = learnLMRoot.appendingPathComponent("models")
print("模型下载路径: \(modelsDir.path)")
print("正在加载 Tokenizer 和模型权重...")
print("(首次运行需下载 ~1GB，后续使用缓存)")
print()

// 加载 Tokenizer (Step02) — 使用自定义下载路径
let tokenizer = try await AutoTokenizer.from(pretrained: modelName, hubApi: hubApi)

// 下载模型文件 — 使用同一个 hubApi
let repo = Hub.Repo(id: modelName)
let modelFolder = try await hubApi.snapshot(from: repo)

// 找到 safetensors 文件
let fm = FileManager.default
let files = try fm.contentsOfDirectory(at: modelFolder, includingPropertiesForKeys: nil)
guard let safetensorsURL = files.first(where: {
    $0.pathExtension == "safetensors" && !$0.lastPathComponent.contains("index")
}) else {
    fatalError("找不到 safetensors 文件")
}

print("模型文件: \(safetensorsURL.lastPathComponent)")
print()

// ═══════════════════════════════════════════════════════
// 6. 解析 safetensors — 找到 Embedding 权重
// ═══════════════════════════════════════════════════════
//
// 复用 Step03 的知识: 解析 safetensors 二进制格式
//   [8字节 header大小] [JSON header] [张量数据]
//
// 关键: 我们不把整个 272MB 的 Embedding 加载到内存
// 而是只读取需要的 token 行 (每行仅 ~1.8KB)

print("--- 6. 解析 safetensors ---")
print()

let fileHandle = try FileHandle(forReadingFrom: safetensorsURL)
defer { try? fileHandle.close() }

// 读 header size (前8字节, UInt64 little-endian)
let headerSizeData = fileHandle.readData(ofLength: 8)
var headerSize: UInt64 = 0
for i in 0..<8 {
    headerSize |= UInt64(headerSizeData[i]) << (i * 8)
}

// 读 JSON header
let headerJSONData = fileHandle.readData(ofLength: Int(headerSize))
let header = try JSONSerialization.jsonObject(with: headerJSONData) as! [String: Any]

// 找到 embed_tokens.weight
let embedKey = "model.embed_tokens.weight"
guard let embedInfo = header[embedKey] as? [String: Any],
      let shape = embedInfo["shape"] as? [Int],
      let dtype = embedInfo["dtype"] as? String,
      let offsets = embedInfo["data_offsets"] as? [Int] else {
    fatalError("找不到 \(embedKey)")
}

let realVocabSize = shape[0]
let realHiddenDim = shape[1]
let dataBaseOffset = 8 + Int(headerSize) + offsets[0]

print("Embedding 权重:")
print("  key:    \(embedKey)")
print("  dtype:  \(dtype)")
print("  shape:  [\(realVocabSize), \(realHiddenDim)]")
print("  大小:   ~\((offsets[1] - offsets[0]) / 1024 / 1024)MB")
print()

// ═══════════════════════════════════════════════════════
// 7. 真实 token 的 Embedding 向量 — 只读需要的行
// ═══════════════════════════════════════════════════════
//
// safetensors 存储的是 row-major (行优先) 的 FP16 数据
//   每行 = hidden_dim 个 FP16 值 = 896 × 2 = 1792 字节
//   token i 的起始偏移 = dataBaseOffset + i × 1792
//
// 所以只读一行: seek → read 1792 bytes → 转成 [Float]
// 不需要加载整个 272MB

print("--- 7. 真实 token 向量 ---")
print()

// 读取单个 token 的 embedding 向量
func readTokenVector(tokenId: Int) -> (floats: [Float], array: MLXArray) {
    let rowBytes = realHiddenDim * 2  // FP16 = 2 bytes per element
    let offset = UInt64(dataBaseOffset + tokenId * rowBytes)
    try! fileHandle.seek(toOffset: offset)
    let data = fileHandle.readData(ofLength: rowBytes)

    // FP16 (Float16) → Float32
    let floats = data.withUnsafeBytes { ptr in
        ptr.bindMemory(to: Float16.self).map { Float($0) }
    }
    return (floats, MLXArray(floats, [realHiddenDim]))
}

// 准备对比的词对 (用常见单字，确保是单个 token)
let wordPairs: [(String, String, String)] = [
    ("猫", "狗", "都是动物"),
    ("猫", "山", "无关概念"),
    ("大", "小", "都是大小描述"),
    ("快", "慢", "都是速度描述"),
    ("冷", "热", "都是温度描述"),
    ("王", "后", "皇室相关"),
]

print("真实 Qwen2.5-0.5B Embedding 对比:")
print()

for (wordA, wordB, desc) in wordPairs {
    // Tokenize — 取第一个 token ID
    let idsA = tokenizer.encode(text: wordA)
    let idsB = tokenizer.encode(text: wordB)
    let idA = idsA[0]
    let idB = idsB[0]

    // 读取向量 (每个只读 ~1.8KB)
    let vecA = readTokenVector(tokenId: idA)
    let vecB = readTokenVector(tokenId: idB)

    // 计算余弦相似度
    let sim = cosineSim(vecA.array, vecB.array)
    eval(sim)

    // 显示前 6 维 (896 维太多，只看开头)
    let firstA = Array(vecA.floats.prefix(6)).map { String(format: "%+.3f", $0) }
    let firstB = Array(vecB.floats.prefix(6)).map { String(format: "%+.3f", $0) }

    print("  \"\(wordA)\" (ID=\(String(format: "%5d", idA))): [\(firstA.joined(separator: ", ")), ...]")
    print("  \"\(wordB)\" (ID=\(String(format: "%5d", idB))): [\(firstB.joined(separator: ", ")), ...]")
    print("  余弦相似度: \(sim)  ← \(desc)")
    print()
}

// ═══════════════════════════════════════════════════════
// 8. 经典验证: king - man + woman ≈ queen
// ═══════════════════════════════════════════════════════
//
// Embedding 最著名的性质: 向量运算可以类比语义关系
//   king - man + woman ≈ queen
//   即: "国王" 减去 "男人" 加上 "女人" ≈ "王后"
//
// 原理: "king" 和 "man" 的向量差编码了 "皇室" 语义
//        把这个差加到 "woman" 上 → "女性皇室" ≈ "queen"
//
// 这个性质不是设计出来的，是训练自动涌现的
// 0.5B 小模型可能效果不明显，但大模型确实能做到

print("--- 8. 经典验证: king - man + woman ≈ queen ---")
print()

let wordAnalogy = [("king", "queen"), ("man", "woman")]
var analogyVecs: [(String, [Float], MLXArray)] = []

for (w1, w2) in wordAnalogy {
    let id1 = tokenizer.encode(text: w1)[0]
    let id2 = tokenizer.encode(text: w2)[0]
    let v1 = readTokenVector(tokenId: id1)
    let v2 = readTokenVector(tokenId: id2)
    analogyVecs.append((w1, v1.floats, v1.array))
    analogyVecs.append((w2, v2.floats, v2.array))

    let sim = cosineSim(v1.array, v2.array)
    eval(sim)
    print("  \"\(w1)\" vs \"\(w2)\": 余弦相似度 = \(sim)")
}

// king - man + woman
let king = analogyVecs[0].2
let man = analogyVecs[1].2
let woman = analogyVecs[3].2
let target = king - man + woman
eval(target)

print()
print("  king - man + woman 的结果向量 vs queen:")
let queen = analogyVecs[2].2
let analogySim = cosineSim(target, queen)
eval(analogySim)
print("  余弦相似度 = \(analogySim)")
print()
print("  (0.5B 小模型的类比能力有限，7B+ 模型效果更好)")
print()

// ═══════════════════════════════════════════════════════
// 9. 位置编码 — Embedding 缺失的那一环
// ═══════════════════════════════════════════════════════
//
// 问题: Embedding 只有语义信息，没有位置信息
//
//   "猫追狗" → embed("猫"), embed("追"), embed("狗")
//              3 个向量，知道每个词的含义
//              但不知道 "猫" 在第1位、"追" 在第2位
//
//   "狗追猫" → embed("狗"), embed("追"), embed("猫")
//              和上面用到的向量完全相同，只是顺序不同
//              模型怎么区分这两种不同的句子？
//
//   答案: 位置编码 — 给每个位置附加一个"位置信号"
//
// ┌──────────────────────────────────────────────────────────┐
// │ 绝对位置编码 (原始 Transformer / GPT-2)                   │
// ├──────────────────────────────────────────────────────────┤
// │                                                          │
// │   类似 Embedding，也有一个查找表:                          │
// │     pos_table[0] = [0.1, 0.5, ...]  ← 位置 0 的编码      │
// │     pos_table[1] = [0.3, 0.2, ...]  ← 位置 1 的编码      │
// │     pos_table[2] = [0.7, 0.8, ...]  ← 位置 2 的编码      │
// │                                                          │
// │   最终输入 = token_embedding + position_embedding         │
// │     "猫追狗":                                            │
// │       [embed("猫")+pos[0], embed("追")+pos[1],            │
// │        embed("狗")+pos[2]]                               │
// │     "狗追猫":                                            │
// │       [embed("狗")+pos[0], embed("追")+pos[1],            │
// │        embed("猫")+pos[2]]                               │
// │                                                          │
// │   同一个 "猫" 在位置 0 和位置 2 的向量不同了!              │
// │                                                          │
// ├──────────────────────────────────────────────────────────┤
// │ RoPE — 旋转位置编码 (Qwen2.5 / LLaMA / Mistral 使用)     │
// ├──────────────────────────────────────────────────────────┤
// │                                                          │
// │   RoPE 不加偏移，而是旋转向量:                             │
// │     位置 0 的向量 → 不旋转                                │
// │     位置 1 的向量 → 旋转 θ                                │
// │     位置 2 的向量 → 旋转 2θ                               │
// │     位置 i 的向量 → 旋转 i×θ                              │
// │                                                          │
// │   在 896 维空间里，每 2 维组成一个旋转平面 (共 448 个)     │
// │   每个平面的旋转角度不同，维度越低角度越大:                │
// │     平面 0:   θ₀ × pos   (大角度，捕捉相邻位置)           │
// │     平面 1:   θ₁ × pos                               │
// │     ...                                                  │
// │     平面 447: θ₄₄₇ × pos (小角度，捕捉远距离位置)         │
// │                                                          │
// │   为什么用旋转:                                           │
// │     内积 (点积) 有旋转不变性                              │
// │     dot(rotate(q, pos_a), rotate(k, pos_b))              │
// │     只取决于 pos_a - pos_b (相对位置)                     │
// │     不取决于 pos_a 和 pos_b 各自的绝对值                   │
// │                                                          │
// │   这就是 "相对位置编码" — 关注"相距多远"而非"各自在哪"     │
// │                                                          │
// │   直觉: 两个向量旋转后做点积 → 结果只跟旋转角度差有关      │
// │     就像两个钟表的指针，角度差 = 时间差，                  │
// │     跟各自指向几点无关                                    │
// │                                                          │
// │   RoPE 的优势:                                           │
// │     1. 不需要额外的位置查找表 (省参数)                    │
// │     2. 天然支持外推 (训练时没见过的长度也能用)             │
// │     3. 相对位置编码更符合语言的本质                       │
// │        ("猫追狗" — 重要的是 "猫" 在 "追" 前面一位，        │
// │         而不是 "猫" 在绝对位置 0)                         │
// │                                                          │
// │   RoPE 在推理流程中的位置:                                │
// │     Embedding 输出 → 不加位置信息 (原始向量)              │
// │     进入 Attention 层后 → 在 Q 和 K 上应用旋转            │
// │     所以 RoPE 不是在 Embedding 阶段加的，是在 Attention 里 │
// │     (Step05 会详细实现)                                   │
// │                                                          │
// └──────────────────────────────────────────────────────────┘

print("--- 9. 位置编码概念 ---")
print()
print("问题: Embedding 只有语义，没有位置")
print("  \"猫追狗\" 和 \"狗追猫\" 用到的向量相同，顺序不同")
print("  模型怎么区分? → 位置编码")
print()
print("RoPE (Qwen2.5 使用):")
print("  不在 Embedding 阶段加位置")
print("  而是在 Attention 层里，对 Q 和 K 做旋转变换")
print("  旋转角度 = 位置 × 频率")
print("  结果: Q·K 的点积只取决于相对位置 (相距多远)")
print("  详见 Step05_Attention")
print()

// ═══════════════════════════════════════════════════════
// 10. 总结 — 数据流全景
// ═══════════════════════════════════════════════════════
//
// 到目前为止的数据流:
//
//   "猫追狗"
//     │
//     ▼  Tokenizer (Step02)
//   [8848, 1039, 4517]          ← token IDs
//     │
//     ▼  Embedding 查找表 (Step04, 本步)
//   [[0.12, ...],               ← shape [3, 896]
//    [0.34, ...],                  每行是一个 token 的 896 维向量
//    [0.56, ...]]
//     │
//     ▼  Transformer × N 层 (Step05-06)
//   每层做:
//     1. Attention (Step05) — 让 token 之间交换信息
//        + RoPE 位置编码 (在这一步加位置信息)
//     2. FFN (Step06) — 非线性变换，增加表达能力
//   输出还是 [3, 896]，但每个向量现在融合了上下文
//     │
//     ▼  lm_head (Step03 讲过)
//   [3, 896] × [896, 151936] → [3, 151936]  ← 每个位置输出词表概率
//     │
//     ▼  Sampling (Step07)
//   取最后一个位置的预测 → 下一个 token
//
// 下一步: Step05_Attention — 学习 Attention 如何让 token 之间交换信息

print("--- 10. 数据流总结 ---")
print()
print("\"猫追狗\"")
print("  → Tokenizer: [8848, 1039, 4517]")
print("  → Embedding:  [[0.12,...], [0.34,...], [0.56,...]]  shape [3, 896]")
print("  → Transformer×24 层: 每层 Attention + FFN")
print("  → lm_head:    [3, 151936]  (每个位置输出词表概率)")
print("  → Sampling:   选概率最高的 → 下一个 token")
print()
print("当前进度: Tokenizer ✓  Embedding ✓")
print("下一步:   Step05_Attention")
print()
print("=== Step04 完成! ===")
