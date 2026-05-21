// Step02: Tokenizer — 把文字变成数字，理解 Tokenization
//
// 学习目标:
//   1. Tokenizer 是什么 — LLM 的 "翻译器"，文字 ↔ 数字
//   2. encode: 文字 → token IDs (整数数组)
//   3. decode: token IDs → 文字
//   4. tokenize: 文字 → token 字符串 (中间形态)
//   5. BPE 算法原理 — 最常用的 tokenizer 方法
//
// Tokenizer 在 LLM 中的位置:
//   输入文字 → [Tokenizer] → token IDs → [Model] → 输出 token IDs → [Tokenizer] → 输出文字
//
// 运行: cd LearnLM && swift run Step02_Tokenizer

import Foundation
import Tokenizers
import Hub

// ═══════════════════════════════════════════════════════════════
// 1. 什么是 Tokenization
// ═══════════════════════════════════════════════════════════════
//
// LLM 不能直接理解文字，它只能处理数字。
// Tokenizer 的工作就是把文字转换成数字 (token IDs)，反过来也能把数字还原成文字。
//
// "Token" 不等于 "字" 也不等于 "词"：
//   - 英文: "Hello" 可能是一个 token，"unbelievable" 可能被拆成 "un" + "believ" + "able"
//   - 中文: "你好" 可能是一个 token，也可能拆成 "你" + "好"
//   - 不同的 tokenizer 有不同的切分策略
//
// 常见的 tokenization 方法:
//   - BPE (Byte-Pair Encoding): GPT/Qwen/Llama 等主流模型使用
//   - WordPiece: BERT 使用
//   - Unigram: T5/ALBERT 使用
//   - SentencePiece: 多用于多语言模型
//
// 这里我们使用 HuggingFace 的 swift-transformers 库来加载现成的 tokenizer。
// 在实际 LLM 推理中，我们不需要自己实现 BPE — 用现成的库即可。

print("=== Step02: Tokenizer ===")
print()

// ═══════════════════════════════════════════════════════════════
// 2. 加载 HuggingFace Tokenizer
// ═══════════════════════════════════════════════════════════════
//
// Tokenizer 需要两个配置文件:
//   - tokenizer.json: 词汇表 (vocabulary) + 合并规则 (merges)
//   - tokenizer_config.json: 特殊 token 定义、chat template 等
//
// 这些文件随模型一起发布在 HuggingFace Hub 上。
// AutoTokenizer 会自动从 Hub 下载这些文件。

let modelName = "Qwen/Qwen2.5-0.5B"  // 小模型，下载快

print("--- 2. 加载 Tokenizer ---")
print("正在从 HuggingFace Hub 下载 tokenizer: \(modelName)")
print("(首次运行需要下载 ~2MB 的 tokenizer 文件，请稍候...)")
print()

let tokenizer = try await AutoTokenizer.from(pretrained: modelName)
print("Tokenizer 加载成功!")
print()

// ═══════════════════════════════════════════════════════════════
// 3. Encode: 文字 → Token IDs
// ═══════════════════════════════════════════════════════════════
//
// encode() 把文字转成整数数组 (token IDs)
// 每个 token ID 对应词汇表 (vocabulary) 中的一个 token

print("--- 3. Encode: 文字 → Token IDs ---")

let chineseText = "你好世界"
let chineseIds = tokenizer.encode(text: chineseText)
print("文字: \"\(chineseText)\"")
print("IDs:  \(chineseIds)")
print()

let englishText = "Hello, world!"
let englishIds = tokenizer.encode(text: englishText)
print("文字: \"\(englishText)\"")
print("IDs:  \(englishIds)")
print()

let mixedText = "Swift是一门很棒的语言!"
let mixedIds = tokenizer.encode(text: mixedText)
print("文字: \"\(mixedText)\"")
print("IDs:  \(mixedIds)")
print()

// 注意: encode 默认会添加 special tokens (如 BOS token)
// BOS = Beginning Of Sequence, EOS = End Of Sequence
print("特殊 Token:")
print("  BOS token: \(tokenizer.bosToken ?? "无") (ID: \(tokenizer.bosTokenId.map(String.init) ?? "无"))")
print("  EOS token: \(tokenizer.eosToken ?? "无") (ID: \(tokenizer.eosTokenId.map(String.init) ?? "无"))")
print()

// ═══════════════════════════════════════════════════════════════
// 4. Decode: Token IDs → 文字
// ═══════════════════════════════════════════════════════════════
//
// decode() 把 token IDs 还原成文字

print("--- 4. Decode: Token IDs → 文字 ---")

let decodedChinese = tokenizer.decode(tokens: chineseIds)
print("IDs: \(chineseIds)")
print("文字: \"\(decodedChinese)\"")
print()

let decodedEnglish = tokenizer.decode(tokens: englishIds)
print("IDs: \(englishIds)")
print("文字: \"\(decodedEnglish)\"")
print()

// Round-trip 验证: encode → decode 应该还原原文
let roundTrip = tokenizer.decode(tokens: tokenizer.encode(text: "测试一下"))
print("Round-trip 验证:")
print("  原文: \"测试一下\"")
print("  encode → decode: \"\(roundTrip)\"")
print()

// ═══════════════════════════════════════════════════════════════
// 5. Tokenize: 文字 → Token 字符串 (中间形态)
// ═══════════════════════════════════════════════════════════════
//
// tokenize() 返回 token 的字符串形式，是 encode 的中间步骤:
//   文字 → tokenize (token字符串) → 查词汇表 → token IDs
//
// 这一步让我们能看到文字被切分成了哪些 token

print("--- 5. Tokenize: 查看 Token 切分 ---")

let tokens1 = tokenizer.tokenize(text: "unbelievable")
print("\"unbelievable\" → \(tokens1)")

let tokens2 = tokenizer.tokenize(text: "人工智能")
print("\"人工智能\" → \(tokens2)")

let tokens3 = tokenizer.tokenize(text: "Hello world")
print("\"Hello world\" → \(tokens3)")
print()

// ═══════════════════════════════════════════════════════════════
// 6. Token ID ↔ Token 字符串 的转换
// ═══════════════════════════════════════════════════════════════
//
// 词汇表 (vocabulary) 就是 token字符串 ↔ ID 的映射表
// Qwen2.5 的词汇表大约有 151,643 个 token
//
// 重要: BPE tokenizer 内部的 token 字符串不是原始文字!
// 中文 "你" 的 UTF-8 字节是 [0xE4, 0xBD, 0xA0]
// 经过 byte-to-unicode 映射后变成类似 "ä½" 这样的字符串
// 所以 convertTokenToId("你") 会返回 nil — 词汇表里的 key 不是 "你"

print("--- 6. Token ↔ ID 转换 ---")

// 正确方式: 先 tokenize 拿到内部 token 字符串，再查 ID
let demoTokens = tokenizer.tokenize(text: "你好")
print("\"你好\" 的内部 token 字符串: \(demoTokens)")
for token in demoTokens {
    if let id = tokenizer.convertTokenToId(token) {
        print("  Token \"\(token)\" → ID: \(id)")
    }
}

// 英文 token 通常可读性更好
if let theId = tokenizer.convertTokenToId("the") {
    print("Token \"the\" → ID: \(theId)")
}

// 反向: ID → Token 字符串 (可能是字节编码的)
if let token = tokenizer.convertIdToToken(151643) {
    print("ID 151643 → Token: \"\(token)\"  (这是 EOS token)")
}
print()

// ═══════════════════════════════════════════════════════════════
// 7. BPE 算法原理 (Byte-Pair Encoding)
// ═══════════════════════════════════════════════════════════════
//
// BPE 是最常用的 tokenization 算法，GPT/Qwen/Llama 都使用它。
//
// 核心思想:
//   1. 把所有文字先转成 UTF-8 字节序列
//   2. 从字符级别开始，反复合并出现频率最高的相邻字节对
//   3. 合并的顺序记录在 "merges" 列表中
//   4. 最终的词汇表 = 初始字节 + 所有合并结果
//
// 举例 (简化版):
//   训练语料: "low low lower lowest"
//
//   第0轮: l o w _ l o w _ l o w e r _ l o w e s t
//          (全是单字节)
//
//   第1轮: 发现 "l"+"o" 出现最多 → 合并为 "lo"
//          lo w _ lo w _ lo w e r _ lo w e s t
//
//   第2轮: 发现 "lo"+"w" 出现最多 → 合并为 "low"
//          low _ low _ low e r _ low e s t
//
//   第3轮: 发现 "e"+"s" 出现最多 → 合并为 "es"
//          low _ low _ low e r _ low es t
//
//   ... 继续合并，直到达到目标词汇表大小
//
// 为什么 BPE 有效:
//   - 高频词 (如 "the", "and") 被完整保留 → 1个token表示
//   - 低频词被拆成子词 (subword) → 仍然可以表示
//   - 任何文字都能处理 (因为 UTF-8 字节是有限的，不会出现 unknown token)
//
// 为什么不用按字/按词切分:
//   - 按字: 中文5万+汉字，词汇表爆炸；且 "你好" 两个 token 比一个 token 多跑一步推理
//   - 按词: 无法处理新词、错别字、混合语言
//   - 字节级 BPE: 高频中文词 ("你好"、"世界") 会被自动合并成单个 token，比逐字更高效
//
// Qwen2.5 的 BPE 词汇表 (~151K tokens):
//   - 常用英文单词: 1个token
//   - 常用中文词: 1-2个token
//   - 罕见词/名字: 被拆成多个子词token
//   - 特殊token: <|im_start|>, <|im_end|>, <|endoftext|> 等

print("--- 7. BPE 原理演示 ---")
print()

// 用一个长单词来展示 BPE 的切分效果
let longWord = "Supercalifragilisticexpialidocious"
let longTokens = tokenizer.tokenize(text: longWord)
print("长单词 \"\(longWord)\":")
print("  被切分成 \(longTokens.count) 个 token: \(longTokens)")
print("  (BPE 会把它拆成已知的子词组合)")
print()

// 对比: 常见词通常是一个 token
let commonWords = ["hello", "world", "the", "apple", "computer"]
print("常见词的 token 数量:")
for word in commonWords {
    let tokens = tokenizer.tokenize(text: word)
    print("  \"\(word)\" → \(tokens.count) token(s): \(tokens)")
}
print()

// ═══════════════════════════════════════════════════════════════
// 8. 实际应用: Token 数量与推理速度
// ═══════════════════════════════════════════════════════════════
//
// 理解 token 数量很重要，因为:
//   - LLM 按token数量计费 (API)
//   - 推理速度 = tokens/second (TPS)
//   - 上下文长度限制 = 最大 token 数 (如 32K, 128K)
//
// 词汇表大小对推理的影响:
//   - 更大词汇表 → 每条文本需要更少 token → 更少推理步骤 → 更快
//   - 代价: Embedding 和 lm_head 矩阵更大，占更多内存
//   - 行业趋势: GPT-2(50K) → Llama-2(32K) → Llama-3(128K) → Qwen2.5(151K)，越来越大
//
// 量化时词汇表的内存 "陷阱":
//   - 模型主体可以 4-bit 量化 (0.5 bytes/参数)
//   - 但 Embedding 和 lm_head 通常保持 FP16 (2 bytes/参数)，贵4倍
//   - 对小模型影响巨大:
//     Qwen2.5-0.5B 量化后，词汇表占内存 ~60%
//     Qwen3.5-4B   量化后，词汇表占内存 ~48%
//     Qwen2.5-72B  量化后，词汇表占内存  ~8%
//   - 模型越小，大词汇表的"税"越重
//   - 端侧部署策略: 裁剪词汇表 (151K→30K)、共享 Embedding/lm_head 权重
//     如 SmolLM-360M 用 49K 词汇表, MobileLLM-0.3B 用 32K 词汇表

print("--- 8. Token 数量与实际意义 ---")

let sentences = [
    "你好世界",
    "The quick brown fox jumps over the lazy dog.",
    "SwiftLM 是一个用 Swift 编写的 LLM 推理服务器，运行在 Apple Silicon 上。",
]

print("不同文本的 token 数量:")
for sentence in sentences {
    let ids = tokenizer.encode(text: sentence)
    print("  [\(ids.count) tokens] \(sentence)")
}
print()

// 中英混合的 token 效率
print("Token 效率对比:")
let same = "机器学习是人工智能的一个分支。"
let sameEn = "Machine learning is a branch of artificial intelligence."
print("  中文: \(tokenizer.encode(text: same).count) tokens")
print("  英文: \(tokenizer.encode(text: sameEn).count) tokens")
print("  (中文通常比英文需要更多 tokens，因为 UTF-8 编码)")
print()

print("=== Step02 完成! ===")
print("下一步: Step03_LoadWeights — 学习如何加载模型权重")
