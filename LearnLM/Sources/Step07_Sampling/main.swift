// Step07: Sampling — 从概率分布中选出下一个 token
//
// 学习目标:
//   1. Softmax — logits 转概率
//   2. Greedy — 选概率最高的 (确定性的)
//   3. Temperature — 控制随机性
//   4. Top-k — 只从概率最高的 k 个中选
//   5. Top-p (Nucleus) — 只从累积概率达到 p 的候选中选
//   6. 组合策略: Temperature + Top-k + Top-p
//
// 运行: cd LearnLM && swift run Step07_Sampling

import Foundation
import MLX

print("=== Step07: Sampling ===")
print()

// ═══════════════════════════════════════════════════════
// 1. 从 logits 到 token — 采样是最后一步
// ═══════════════════════════════════════════════════════
//
// 模型输出的 logits: [151936] 个分数 (每个词表 token 一个)
// 采样的任务: 从这些分数中选出一个 token ID 作为"下一个 token"
//
// 最简单的选法: argMax → 选分数最高的 (Greedy)
// 更好的选法:   把分数转成概率，按概率随机选 (Sampling)
//
// 为什么不总是选最高的:
//   "今天天气很___"
//   Greedy: 总是选 "好" → 每次回答都一样
//   Sampling: 可能选 "好"/"差"/"热"/"冷" → 更自然多样

print("--- 1. 采样的位置 ---")
print("logits [151936] → 采样策略 → 1 个 token ID")
print("Greedy: 总选最高 → 确定性但无聊")
print("Sampling: 按概率随机选 → 多样但可能不连贯")
print()

// ═══════════════════════════════════════════════════════
// 2. Softmax — 把分数变成概率
// ═══════════════════════════════════════════════════════
//
// softmax(x_i) = exp(x_i) / Σ exp(x_j)
//
// 性质:
//   1. 所有输出在 (0, 1) 之间
//   2. 所有输出加起来 = 1 (概率分布)
//   3. 保留相对大小: 分数高的 → 概率高
//
// 数值稳定性技巧: 先减去最大值
//   softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))
//   防止 exp(大数) = infinity

print("--- 2. Softmax ---")
print()

// 模拟 6 个 token 的 logits (简化版，真实是 151936 个)
let rawLogits = MLXArray([Float]([2.1, 5.3, 1.2, 3.8, 0.5, 4.1]))
let probs = softmax(rawLogits, axis: 0)
eval(probs)
print("Raw logits: [2.1, 5.3, 1.2, 3.8, 0.5, 4.1]")
print("Softmax:    \(probs)")
print("  (加起来=1, 最高分 5.3 → 最高概率)")
print()

// ═══════════════════════════════════════════════════════
// 3. Greedy — 贪心解码
// ═══════════════════════════════════════════════════════
//
// 选概率最高的 token: argMax(logits)
// 优点: 确定性的，可复现
// 缺点: 输出重复、无聊、缺乏创造性

print("--- 3. Greedy 解码 ---")

let greedyIdx = argMax(rawLogits)
eval(greedyIdx)
print("argMax(logits) = token \(greedyIdx) (概率最高的)")
print("  确定、可复现、但无聊")
print()

// ═══════════════════════════════════════════════════════
// 4. Temperature — 控制随机性
// ═══════════════════════════════════════════════════════
//
// logits / temperature → softmax → 概率
//
//   T < 1.0: 放大差异 → 更确定 (偏向高分 token)
//   T = 1.0: 原始分布
//   T > 1.0: 缩小差异 → 更随机 (低分 token 也有机会)
//
//   T → 0:   等价于 Greedy (只选最高分)
//   T → ∞:   等概率 (完全随机)
//
// 实际使用:
//   代码生成: T=0.2 (需要确定性)
//   对话:     T=0.7 (平衡连贯和多样)
//   创意写作: T=1.0-1.5 (鼓励多样性)

print("--- 4. Temperature ---")
print()

let temps: [Float] = [0.1, 0.5, 1.0, 2.0]
for t in temps {
    let scaled = rawLogits / MLXArray(t)
    let p = softmax(scaled, axis: 0)
    eval(p)
    print("T=\(String(format: "%.1f", t)): \(p)")
}
print("  T=0.1: 几乎全部概率集中在最高分 token (接近 greedy)")
print("  T=2.0: 概率分布趋于平坦 (更随机)")
print()

// ═══════════════════════════════════════════════════════
// 5. Top-k — 只考虑前 k 个
// ═══════════════════════════════════════════════════════
//
// 思路: 把 logits 中前 k 名之外的都设为 -∞
//   softmax(-∞) = 0 → 这些 token 概率为 0 → 不可能被选中
//
// 效果: 限制候选范围，避免选到离谱的 token
//   top_k=50: 只从概率最高的 50 个 token 中选
//   top_k=1:  等价于 Greedy

print("--- 5. Top-k ---")
print()

// 手动实现 top-k (教学用，真实推理用高效实现)
let logitsValues: [Float] = [2.1, 5.3, 1.2, 3.8, 0.5, 4.1]
let tokenNames = ["的", "好", "了", "天", "啊", "很"]

func topKFilter(_ logits: [Float], k: Int) -> [Float] {
    let sorted = logits.enumerated().sorted { $0.element > $1.element }
    let threshold = sorted[min(k, sorted.count) - 1].element
    return logits.map { $0 < threshold ? -.infinity : $0 }
}

for k in [2, 3, 6] {
    let filtered = topKFilter(logitsValues, k: k)
    let filteredMLX = MLXArray(filtered)
    let p = softmax(filteredMLX, axis: 0)
    eval(p)
    let pStr = (0..<6).map { i -> String in
        let label = filtered[i] == -.infinity ? "  -∞ " : String(format: "%5.3f", p[i].item(Float.self))
        return "\(tokenNames[i]):\(label)"
    }.joined(separator: " ")
    print("top_k=\(k): \(pStr)")
}
print("  top_k=2: 只有\"好\"和\"很\"有概率")
print("  top_k=6: 等于没有过滤 (全部 token)")
print()

// ═══════════════════════════════════════════════════════
// 6. Top-p (Nucleus Sampling) — 累积概率阈值
// ═══════════════════════════════════════════════════════
//
// 思路: 把 token 按概率从高到低排序，累积概率达到 p 后停止
//   只在这些 token 之间重新分配概率
//
// top_p=0.9: 最少需要多少个 token 让累积概率 ≥ 0.9，就只从这些选
//
// vs top-k:
//   top-k 固定选 k 个 → 不管概率分布是否集中
//   top-p 自适应 → 概率集中时少选几个，分散时多选几个
//
// 例子:
//   如果 top-2 token 的概率之和 = 0.92 → top_p=0.9 只需 2 个
//   如果 top-5 token 的概率之和 = 0.85 → top_p=0.9 需要 6+ 个

print("--- 6. Top-p (Nucleus Sampling) ---")
print()

func topPFilter(_ logits: [Float], p: Float) -> [Float] {
    // softmax
    let maxVal = logits.max()!
    let exps = logits.map { exp($0 - maxVal) }
    let sumExps = exps.reduce(0, +)
    let probs = exps.map { $0 / sumExps }

    // 按概率从高到低排序
    let indexed = probs.enumerated().sorted { $0.element > $1.element }
    var cumSum: Float = 0
    var keepIndices = Set<Int>()
    for (idx, prob) in indexed {
        cumSum += prob
        keepIndices.insert(idx)
        if cumSum >= p { break }
    }

    return logits.enumerated().map { i, v in keepIndices.contains(i) ? v : -.infinity }
}

for p in [0.5, 0.9] as [Float] {
    let filtered = topPFilter(logitsValues, p: p)
    let filteredMLX = MLXArray(filtered)
    let probsResult = softmax(filteredMLX, axis: 0)
    eval(probsResult)
    let pStr = (0..<6).map { i -> String in
        let label = filtered[i] == -.infinity ? "  -∞ " : String(format: "%5.3f", probsResult[i].item(Float.self))
        return "\(tokenNames[i]):\(label)"
    }.joined(separator: " ")
    print("top_p=\(String(format: "%.1f", p)): \(pStr)")
}
print()

// ═══════════════════════════════════════════════════════
// 7. 实际推理中的组合策略
// ═══════════════════════════════════════════════════════
//
// 典型配置:
//   1. logits / temperature (缩放)
//   2. Top-k 过滤 (限制候选数)
//   3. Top-p 过滤 (自适应截断)
//   4. Softmax → 概率
//   5. 按概率随机采样一个 token
//
// 重复惩罚 (Repetition Penalty):
//   如果某个 token 最近刚出现过 → 降低它的概率
//   防止输出重复内容
//
// 不同场景推荐:
//   ┌────────────┬───────┬───────┬───────┐
//   │ 场景       │ Temp  │ Top-k │ Top-p │
//   ├────────────┼───────┼───────┼───────┤
//   │ 代码生成   │ 0.2   │ -     │ 0.95  │
//   │ 日常对话   │ 0.7   │ -     │ 0.9   │
//   │ 创意写作   │ 1.0   │ 50    │ 0.95  │
//   │ 翻译       │ 0.3   │ -     │ 0.95  │
//   │ Greedy     │ →0    │ 1     │ -     │
//   └────────────┴───────┴───────┴───────┘

print("--- 7. 组合策略 ---")
print()
print("实际推理流程: logits → /Temperature → Top-k → Top-p → Softmax → 随机采样")
print()
print("推荐配置:")
print("  代码: T=0.2, top_p=0.95")
print("  对话: T=0.7, top_p=0.9")
print("  创意: T=1.0, top_k=50, top_p=0.95")
print()

// ═══════════════════════════════════════════════════════
// 8. Mini 演示: 不同采样策略的效果
// ═══════════════════════════════════════════════════════

print("--- 8. 采样效果对比 ---")
print()

// 模拟一个"今天天气很___"的 logits
// token:     好    不错   差    热    冷   晴朗   棒   糟糕
let weatherLogits: [Float] = [5.0, 3.5, 0.5, 2.0, 1.5, 4.0, 3.0, -1.0]
let weatherTokens = ["好", "不错", "差", "热", "冷", "晴朗", "棒", "糟糕"]

// Greedy
let greedyMLX = MLXArray(weatherLogits)
let greedyResult = argMax(greedyMLX)
eval(greedyResult)
print("Greedy: \"\(weatherTokens[greedyResult.item(Int.self)])\"")

// Temperature 采样
for t in [0.3, 0.7, 1.5] as [Float] {
    let scaled = MLXArray(weatherLogits.map { $0 / t })
    let p = softmax(scaled, axis: 0)
    eval(p)
    // 用随机采样 (简化: 按概率加权选)
    // 这里为了确定性展示，只看 top-3 概率
    let probsArr = (0..<8).map { i -> (Int, Float) in
        let val = p[i].item(Float.self)
        return (i, val)
    }.sorted { $0.1 > $1.1 }

    let top3 = probsArr.prefix(3).map { "\(weatherTokens[$0.0])(\(String(format: "%.1f%%", $0.1 * 100)))" }.joined(separator: ", ")
    print("T=\(String(format: "%.1f", t)): top3 = \(top3)")
}
print()

print("=== Step07 完成! ===")
print("下一步: Step08_KVCache — 避免重复计算的缓存技巧")
