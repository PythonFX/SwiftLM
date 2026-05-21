// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LearnLM",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "Step01_HelloMLX", targets: ["Step01_HelloMLX"]),
        .executable(name: "Step02_Tokenizer", targets: ["Step02_Tokenizer"]),
        .executable(name: "Step03_LoadWeights", targets: ["Step03_LoadWeights"]),
        .executable(name: "Step04_Embedding", targets: ["Step04_Embedding"]),
    ],
    dependencies: [
        // 官方Apple MLX Swift — 学习阶段用官方版即可
        // 后续优化阶段可切换到 ../mlx-swift (SharpAI fork with SSD streaming)
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.2.0")),
    ],
    targets: [
        // ── Step 1: MLX 基础 — 张量、GPU计算、矩阵乘法 ──
        .executableTarget(
            name: "Step01_HelloMLX",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/Step01_HelloMLX"
        ),
        // ── Step 2: Tokenizer — 文字 ↔ 数字转换 ──
        .executableTarget(
            name: "Step02_Tokenizer",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/Step02_Tokenizer"
        ),
        // ── Step 3: Load Weights — 模型文件结构与权重加载 ──
        .executableTarget(
            name: "Step03_LoadWeights",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
            ],
            path: "Sources/Step03_LoadWeights"
        ),
        // ── Step 4: Embedding — Token ID → 语义向量 ──
        .executableTarget(
            name: "Step04_Embedding",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/Step04_Embedding"
        ),
    ]
)
