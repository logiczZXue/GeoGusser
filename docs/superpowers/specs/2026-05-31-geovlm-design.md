# GeoVLM: 边缘端地理要素提取专用视觉语言模型

## 设计文档

**日期**: 2026-05-31
**状态**: 设计完成，待实现
**目标硬件**: DK-2500 (Intel Core Ultra 5 225U, 8GB LPDDR5X, ~10 TOPS NPU)

---

## 1. 动机

### 1.1 核心矛盾

图寻系统依赖 VLM（视觉语言模型）从街景图片中提取地理要素（气候带、地形、植被、城市化程度等），然后将这些要素输入 GeoKB 融合引擎进行空间定位。

通用 VLM（Qwen3-VL-4B-Instruct）能够产出可用的要素，但无法部署在 DK-2500（8GB 统一内存，无独显）上。2B 模型能跑但精度不足——参数从 4B 缩到 2B，要素提取准确率显著下降。

**根本矛盾**：VLM 要素提取能力与 VRAM 容量成正比，但边缘设备 VRAM 极度受限。

### 1.2 洞察

通用 VLM 的参数分配存在严重效率问题：

| 组件 | 参数量 | 与地理要素提取的相关性 |
|------|--------|------------------------|
| Vision Encoder (ViT) | 300M | 100% 相关 — 看图能力 |
| Projector | 50M | 100% 相关 — 特征转换 |
| Language Model | 1500M | ~5% 相关 — 其余是写诗、解方程、翻译等通用能力 |

砍掉 LLM 的 95% 无用容量，用专用轻量 Decoder 替代，将释放的 VRAM 投入到多尺度视觉等真正提升精度的方向。

### 1.3 设计哲学

受 Agent 架构哲学启发：**模型变薄，工具变厚**。世界知识和推理能力不应在模型参数中，而应在外部知识库（GeoKB）和传感器中。

---

## 2. 业界参考

| 来源 | 关键技术 | 借鉴点 |
|------|---------|--------|
| BLIP-2 (Salesforce, 2023) | Q-Former: 32 learnable queries + cross-attn, 100M 参数桥接视觉和语言 | 用 learnable queries 替代完整 LLM 的核心可行性证明 |
| DINOv2 (Meta, 2023) | 自监督 ViT 在细粒度识别上超越监督模型 | ViT 本身的视觉能力足够，不依赖 LLM 的知识 |
| Token Merging (Meta, 2022) | 合并相似 visual token，减少 50-60% 序列长度，精度损失 <1% | 多尺度高分辨率序列的压缩方案 |
| MobileCLIP (Apple, 2024) | 多尺度训练 + 结构重参数化，小模型匹配大模型 | 多尺度视觉是缩小精度差距的关键杠杆 |
| LLaVA-PruMerge (2024) | 剪枝 90% visual token，VLM 性能几乎不降 | visual token 极度冗余，安全剪枝的置信度 |
| TinyLLaVA (2024) | 小 VLM 通过高质量数据匹配大 VLM 的 85% | 数据质量 > 模型大小 |
| OpenVINO NNCF (Intel) | INT8 量化 ViT 精度损失 <0.5% | 部署量化的可行性 |
| DistilBERT (HuggingFace) | 三损失蒸馏 (CE + MLM + Cosine Embedding) | 多层损失联合优化 |

**核心结论**：visual token 极度冗余 + 多尺度是关键杠杆 + 数据质量 > 模型大小。

---

## 3. 架构总览

```
                         输入
                 ┌────────┴────────┐
                 ▼                  ▼
           ┌───────────┐    ┌──────────────┐
           │   图片      │    │  传感器        │
           │ (任意分辨率) │    │ elev/temp/humid│
           └─────┬─────┘    └──────┬───────┘
                 │                  │
   ┌─────────────┼──────┐           │
   ▼             ▼      ▼           │
┌──────┐  ┌──────┐ ┌──────┐         │
│224²  │  │448²  │ │672²  │         │
│ViT   │  │ViT   │ │ViT   │         │
│196tok│  │784tok│ │1764t │         │
└──┬───┘  └──┬───┘ └──┬───┘         │
   │         │ToMe    │ToMe         │
   │         ▼314     ▼530          │
   │  ┌──────────────┐  │           │
   └─▶│ Scale Fusion │◄─┘           │
      │ ~520 tokens  │              │
      └──────┬───────┘              │
             │                      │
      ┌──────▼──────┐    ┌──────────▼─────┐
      │ Q-Former    │◄───│ Sensor Encoder  │
      │ 8 Field Qs  │    │ [3 tokens, 512] │
      │ 3 Blocks    │    └────────────────┘
      └──────┬──────┘
             │
      ┌──────▼──────────┐
      │ Constraint Graph │
      │ GNN 8 Nodes      │
      └──────┬───────────┘
             │
      ┌──────▼──────┐
      │ 8 Heads     │
      │ + Confidence │
      └──────┬──────┘
             │
      ┌──────▼──────────┐    ┌────────────┐
      │ Confidence<0.5? │───▶│ Crop 672²  │→ 重回 ViT
      └──────┬──────────┘    │ 精炼低置信  │
             │ no             └─────┬──────┘
      ┌──────▼──────┐              │
      │  结构化输出   │◄─────────────┘
      │  elements {} │
      └──────┬──────┘
             │
      ┌──────▼──────────┐
      │ 现有融合引擎      │
      │ fuse_elements_v3 │
      │ GeoKB + Scoring  │
      │ Grid Search     │
      └──────┬──────────┘
             │
      ┌──────▼──────┐
      │   (lat,lng)  │
      │ + uncertainty │
      │ + confidence  │
      └──────────────┘
```

---

## 4. 组件详设

### 4.1 多尺度视觉编码器

**动机**：ViT 输入 224×224 时，树叶纹理（~3px）、岩石纹理（~2px）和建筑装饰细节（~5px）在 patch grid 中完全不可区分。8B 的 ViT 更大、原生分辨率更高，这是精度优势的主要来源。多尺度视觉用更少的参数和计算量复现这一优势。

**设计**：
- 三尺度共享 ViT：224²（粗/整体景观）、448²（中/建筑和植被结构）、672²（细/纹理和细节）
- 共享权重 — 同一双眼睛在不同距离看同一张图。独立三尺度 ViT 需要 900M 参数，共享只需 300M
- Token Merging (ToMe)：中尺度 784→314 tokens (60% 压缩)，细尺度 1764→530 tokens (70% 压缩)
- Scale Fusion：给每个 token 加可学习的 scale embedding [COARSE/MID/FINE]，经 1 层轻量 Self-Attention 让不同尺度互相交流，再压缩到 ~520 tokens
- 输出：[520, 512] unified visual features

**关键洞察**：ToMe 的相似度阈值自动将天空、路面等冗余区域的数百个几乎相同的蓝色 token 合并为少数几个。纹理信息主要集中在植被边缘、建筑立面和岩石表面——压缩 60-70% 几乎不损失有用信息。

### 4.2 Sensor Encoder

**设计**：
- 输入：[elevation (float), temperature (float), humidity (float)]
- MLP: 3 → 64 → 128 → 512
- 输出：[3, 512] sensor tokens
- 训练时注入噪声（elev ±15%, temp ±2°C, humidity ±5%）以增强鲁棒性

### 4.3 Field-Aware Q-Former (核心创新)

**动机**：BLIP-2 证明 100M 参数的 Q-Former 足以桥接视觉和语言。GeoVLM 将其改造为地理要素专用——8 个 learnable query 分别对应 8 个地理要素字段。

**设计**：
- 8 个 Field Queries [8, 512]，初始化为 N(0, 0.02)
- 每个 query 加上可学习的 Field-Type Embedding（类似位置编码，标识该 query 的"身份"）：
  - `field_type_embed = one_hot(field_id, 8) × Linear(8→512)`
- 3 个 Q-Former Blocks，每个包含：
  - **Cross-Attention**: queries ← visual_tokens + sensor_tokens
    - climate_query 关注植被区域和传感器 token
    - terrain_query 关注地面坡度和天际线
    - vegetation_query 关注绿色区域和纹理
    - 同一视觉特征池，不同 query 各取所需
  - **Self-Attention**: queries ← other queries
    - vegetation 看到 "棕榈树" → climate 自动提升 "tropical" 的 logit
    - terrain 看到 "雪山" → elevation 自动提升海拔估计
    - GeoKB 的隐性约束通过反向传播蒸馏进 attention 权重
  - **FFN**: SwiGLU, hidden_dim=2048

**为什么是 8 个 Query 而非 32 个（BLIP-2）**：
- 8 个 query 与 8 个地理要素字段一一对应，职责在初始化时即明确
- 输出天然结构化：query[0] → climate head, query[1] → terrain head, ...
- 可解释性：可可视化每个 query 的 cross-attention heatmap（"climate 在看植被"）
- 计算量仅 32M FLOPs（vs BLIP-2 的 ~120M FLOPs）

**复杂度分析**：
- Cross-Attention: O(8 × 520 × 512) ≈ 2.1M ops
- Self-Attention: O(8² × 512) ≈ 32K ops（可忽略）
- FFN: O(8 × 512 × 2048) ≈ 8.4M ops
- 3 Blocks 合计：~32M ops —— 比原 VLM 的 LLM 每次 token 生成（1500M 参数 × full forward）差 3 个数量级

### 4.4 Constraint Graph Layer

**设计**：
- 8 个节点的异质图，每个节点是一个地理要素字段
- 预定义约束边（可学习权重）：
  - elevation ↔ climate（海拔物理方程约束）
  - climate ↔ vegetation（生态学约束：热带雨林不在温带）
  - vegetation ↔ terrain（地理学约束：荒漠上不长森林）
  - terrain ↔ elevation（平原海拔 < 500m）
  - urbanization ↔ building_height（城市规划约束）
- 2 轮消息传递（GraphSAGE 风格）：
  - Round 1：从有边邻居收集消息，检测预测矛盾
  - Round 2：基于第一轮结果更新自身表示
- 输出：约束修正后的 8 个 field embeddings + 全局一致性分数

**注意**：边权重可学习。模型在训练中自动学到 "当 elevation 和 vegetation 同时指向 alpine 时，我应该优先考虑 alpine climate"——而非死记规则。

### 4.5 Prediction Heads

8 个独立的分类/回归头：
- 7 个分类头：FC(512→vocab_size) + softmax，输出值 + confidence (entropy)
- 1 个回归头（elevation）：FC(512→2) 输出 [min, max] 范围
- 每头加上 field_type_embed 的 shortcut 连接

### 4.6 Progressive Refinement

**动机**：大部分图像的字段都可以在标准三尺度下高置信度推断，但少数难例（如 mountain_rock_type）需要极高分辨率。对所有图像都做全分辨率会浪费计算。

**设计**：
- 预测阶段输出 per-field confidence（softmax 的熵）
- confidence < 0.5 的字段 → 触发 refinement
- iGPU 裁剪图像中该字段关注的区域 + resize 到 672²
- NPU 对裁剪区域重新 ViT → Q-Former（仅受影响字段）
- 额外开销：~230ms，仅在 ~15-20% 的图像上触发

### 4.7 输出兼容性

GeoVLM 输出 `elements dict` 格式与当前 VLM 输出完全兼容：
- 8 个地理要素字段的枚举值
- `ruled_out_features`（新增，当前引擎已有此字段但 VLM 几乎不输出）
- `confidence`（新增，per-field 置信度，scoring.py 可直接利用）
- `field_consistency`（新增，Constraint Graph 输出的全局一致性分数，用于动态调整 DISPERSE 阈值）

下游 `fuse_elements_v3()` 无感知切换。

---

## 5. 蒸馏训练策略

### 5.1 训练数据

- 1150 张分层采样图片（1000 train + 150 val）
- 双重分层：urbanization × climate_zone，每组 2-3 张不同时段/季节
- 特殊类别（alpine/boreal/tulou 等）强制保留最低样本量

### 5.2 教师标注

Teacher: Qwen3-VL-8B-Instruct（非 Thinking，Instruct 即可）

集成标注流程：
- 3 次独立推理（T=0.6, seed=0/42/99）
- 多数投票 → hard label；平均分布 → soft label（用于 KL 蒸馏）
- 3 票全一致 → 高置信度；2:1 → 中置信度（×0.7 训练权重）；全不一致 → 人工抽检
- 传感器一致性过滤：标注 elevation 与传感器差异 > 2σ → 用传感器值替代

### 5.3 三阶段训练

**Stage 1: 表示对齐**（~30 分钟）
- 冻结 ViT，只训练 Q-Former + Constraint Graph + Heads
- Loss: L_task + L_distill
- 数据：200 张简单图（清晰的 urban/rural/mountain）
- 目标：Q-Former 学会如何从 ViT 特征中读取信息

**Stage 2: 全模型蒸馏**（~3 小时）
- 解冻 ViT 最后 4 层
- 渐进式难度：epochs 1-3 简单 → epochs 4-6 中等 → epochs 7-10 困难
- Loss: L_task + L_distill + L_feature + L_constraint
- 目标：每字段 accuracy 逼近 Teacher

**Stage 3: 传感器对齐**（~1 小时）
- 全模型微调，传感器噪声增强
- Loss: L_task + L_distill + L_sensor（加重权重）
- 目标：elevation_est 对传感器锚点的偏离 < 15%

### 5.4 损失函数

```
L_total = 
    1.0  × L_task          ← CE per field（主导）
  + 0.3  × L_distill        ← KL(softmax_student || softmax_teacher)，传递"暗知识"
  + 0.1  × L_feature        ← MSE(feature_student, feature_teacher)，ViT 特征对齐
  + 0.3  × L_sensor         ← |pred_elev - sensor_elev| / sensor_elev
  + 0.2  × L_constraint     ← constraint violation penalty
  + 0.01 × L_reg            ← L2 权重衰减
```

### 5.5 正则化

- Q-Former: dropout=0.15
- Prediction Heads: dropout=0.1
- ViT fine-tune 层: stochastic depth=0.1
- 权重衰减: 1e-2
- Label Smoothing: 0.1（hard labels only）
- Early Stopping: val loss 3 epochs 不降

### 5.6 数据增强

安全增强（不破坏地理线索）：
- 水平翻转
- 亮度 ±20%、对比度 ±15%、饱和度 ±10%（模拟不同时段/天气）
- 随机裁剪 5-10%

禁止增强（会破坏地理线索）：
- 垂直翻转（天空跑到底部）
- 大幅旋转（地平线倾斜 → 地形判断错误）
- 重度模糊（纹理丧失 → 细粒度字段崩溃）

---

## 6. DK-2500 部署

### 6.1 内存预算

| 组件 | INT8 大小 |
|------|-----------|
| ViT (共享 300M) | 300 MB |
| 多尺度分支 (FPN + Fusion) | 15 MB |
| Q-Former (3 blocks, 110M) | 80 MB |
| Sensor Encoder (10M) | 10 MB |
| Constraint Graph (20M) | 20 MB |
| Prediction Heads (5M) | 5 MB |
| **模型合计** | **430 MB** |
| | |
| 推理激活值 (峰值) | 150 MB |
| OpenVINO Runtime | 150 MB |
| GeoKB + DEM + Climate | 40 MB |
| **运行时合计** | **340 MB** |
| | |
| Windows 11 空闲 | ~3.5 GB |
| Python + 依赖库 | ~0.3 GB |
| **系统合计** | **~3.8 GB** |
| | |
| **内存总计** | **~4.6 GB / 8 GB** |
| **可用余量** | **~3.4 GB** |

### 6.2 异构计算流水线

DK-2500 的核心硬件优势：CPU/NPU/iGPU 共享统一 LPDDR5X 内存，数据搬运零拷贝。

```
NPU  │ ViT(224²)│ ViT(448²)│ ViT(672²)│
     │          │ +ToMe    │ +ToMe    │
─────┼──────────┼──────────┼──────────┼───
CPU  │ img      │ Scale    │ Q-Former │ Sensor   │
     │ 预处理   │ Fusion   │ + GNN    │ + GeoKB  │
─────┴──────────┴──────────┴──────────┴──────────┴───
      |←────── 单张图端到端 < 600ms ──────→|
```

| 阶段 | 硬件 | 耗时 | 说明 |
|------|------|------|------|
| ViT 224² | NPU | ~50ms | 196 tokens |
| ViT 448² + ToMe | NPU | ~120ms | 784→314 tokens |
| ViT 672² + ToMe | NPU | ~200ms | 1764→530 tokens |
| Scale Fusion | CPU | ~30ms | 1040→520 tok, 轻量 SA |
| Q-Former 3 blocks | CPU | ~80ms | 8 queries × 520 tok |
| Constraint GNN | CPU | ~5ms | 8 节点 |
| Prediction Heads | CPU | ~2ms | 8 个 softmax |
| 传感器 + GeoKB 融合 | CPU | ~100ms | 现有引擎 |
| **合计** | | **~590ms** | 不含 Refinement |

Progressive Refinement 额外开销：~230ms（~15-20% 图像上触发）

### 6.3 扬长避短

NPU 擅长：
- 固定 shape 的大矩阵乘法（ViT forward）
- Conv + LayerNorm + GELU
- Attention QKV projection

CPU 擅长：
- if-else 控制流（GNN 消息传递）
- 动态 shape（Token Merging 合并数不固定）
- 小规模稀疏操作（8×8 GNN）
- 文件 IO、JSON 解析

### 6.4 比赛叙事

> "我们针对 DK-2500 的统一内存架构和异构计算特性，设计了 NPU-CPU-iGPU 三级流水线并行方案。传统 GPU 方案受限于 PCIe 数据搬运延迟，而 Meteor Lake 的 NPU 与 CPU 共享 LPDDR5X 物理内存，实现了零拷贝流水线——NPU 产出的视觉特征直接被 CPU 端的 Q-Former 消费。结果: ~600ms/张端到端，NPU 利用率 >80%，内存占用 <5GB，为未来功能扩展留出 >3GB 余量。"

---

## 7. 评测方案

### 7.1 对比矩阵

| | GeoVLM (Ours) | 4B-Instruct (基线) | 8B-Instruct (上界) |
|---|---|---|---|
| 模型参数 | 430M | 4B | 8B |
| 推理时间 | <1s | 76s | 816s |
| DK-2500 可跑 | Y | N | N |

### 7.2 评测流程

**Phase A: Y7000P 离线对比**
- Step 1: 50 张分层测试 → GeoVLM vs 4B-Instruct vs 8B-Instruct
- Step 2: 500 张全集 → GeoVLM vs 4B-Instruct V5 统计显著性检验

**Phase B: DK-2500 部署验证**
- 5 张图在 DK-2500 上跑通
- 验证 OpenVINO INT8 转换成功
- 验证推理延迟 < 2s
- 验证量化精度损失 < 1%

### 7.3 评测指标

- Per-field accuracy vs Teacher（要素提取准确率）
- Per-field Cohen's Kappa（消去随机一致性）
- Fusion 误差 (km)（终极定位指标）
- 搜索空间缩小率（要素贡献度）
- DISPERSE 触发率（多假设场景频率）

### 7.4 精度目标

| 字段 | 当前 4B-Instruct | GeoVLM 目标 | 提升来源 |
|------|:---:|:---:|------|
| climate_zone | 94% | >95% | 传感器直接参与判断 |
| terrain_type | 100% | 100% | 粗尺度就可判断 |
| vegetation_zone | 86% | >90% | 中细尺度 + 传感器约束 |
| urbanization | 100% | 100% | 粗尺度就够 |
| elevation_est | 偏差大 | 偏差<15% | 传感器强制约束 |
| architecture_style | 43% | >65% | 中细尺度纹理 |
| tree_species | 29% | >50% | 细尺度树叶纹理 |
| mountain_rock_type | 4.4% | >35% | 细尺度岩石纹理 |
| **Fusion 均值** | **392km** | **<200km** | 多要素综合提升 |

---

## 8. 实现计划

### 8.1 文件变更

**新增**（6 个文件）：
- `src/geovlm/__init__.py`
- `src/geovlm/vision_encoder.py` — ViT 提取 + 多尺度分支 + ToMe + Scale Fusion
- `src/geovlm/qformer.py` — Field-Aware Q-Former (8 queries, 3 blocks)
- `src/geovlm/constraint_graph.py` — GNN Constraint Layer
- `src/geovlm/sensor_encoder.py` — Sensor Token Encoder
- `src/geovlm/model.py` — GeoVLM 主模型组装
- `scripts/train_geovlm.py` — 三阶段蒸馏训练
- `scripts/convert_geovlm_ov.py` — OpenVINO INT8 转换 + DK-2500 部署

**修改**（1 个文件）：
- `src/Geocot/Geocot.py` — 添加 GeoVLM 推理路径

**不改**：
- `src/regression/` 下所有（融合引擎保持完全兼容）

### 8.2 四周里程碑

| 周 | 里程碑 | 产出 |
|----|--------|------|
| Week 1 | ViT 多尺度 + ToMe | vision_encoder.py 完成，前向验证通过 |
| Week 2 | Q-Former + GNN + Heads | model.py 完成，端到端推理跑通 |
| Week 3 | 教师标注 + 蒸馏训练 | train_geovlm.py 完成，50 张验证集达标 |
| Week 4 | OpenVINO + DK-2500 + E2E | convert_geovlm_ov.py 完成，全面评测报告 |

### 8.3 风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| NPU 不支持自定义算子（ToMe） | 中 | ToMe 在 CPU 上执行，仅 ~50ms 额外开销 |
| 8B-Instruct 细粒度标注有噪声 | 中 | 集成标注（3 次投票）+ 传感器过滤 + 人工抽检 5% |
| 蒸馏精度不到 8B 的 80% | 低 | 渐进式难度训练 + KL 蒸馏 + 多尺度视觉 |
| DK-2500 OpenVINO 转换问题 | 中 | 回退 ONNX Runtime CPU 推理 |

---

## 9. 附录：字段词汇表

| 字段 | 类别数 | 选项 |
|------|--------|------|
| climate_zone | 6 | tropical, subtropical, temperate, arid, alpine, boreal |
| terrain_type | 9 | urban_flat, farmland_plain, rolling_hills, sharp_mountains, karst_peaks, sandstone_pillars, desert_dunes, grassland_steppe, plateau |
| vegetation_zone | 11 | tropical_rainforest, broadleaf_evergreen, broadleaf_deciduous, conifer_forest, mixed_forest, alpine_meadow, desert_scrub, grassland, bamboo_forest, cropland, sparse |
| urbanization | 6 | metropolis, medium_city, small_town, village, rural, wilderness |
| building_height | 6 | super_tall, high_rise, mid_rise, low_rise, mixed, no_buildings |
| pavement_type | 5 | red_brick_tiles, grey_concrete, asphalt, natural, not_visible |
| language_script | 6 | simplified_chinese, traditional_chinese, tibetan, uyghur_arabic, mongolian, bilingual_cn_en, none_visible |
| architecture_style | 12 | modern_glass, modern_residential, old_residential, hui_style, tibetan_stone, courtyard, arcade, stilt_house, tulou, shikumen, traditional_official, soviet_industrial, none_visible |
| tree_species | 10 | banyan, camphor, plane_tree, gingko, poplar, willow, coconut_palm, chinese_red_pine, huangshan_pine, bamboo |
| mountain_rock_type | 7 | granite_spheroidal, quartz_sandstone_pillars, limestone_karst, red_sandstone_danxia, snow_peaks_glaciers, alpine_lakes, null |
| elevation_estimate_m | regr | [min, max] 范围 |
