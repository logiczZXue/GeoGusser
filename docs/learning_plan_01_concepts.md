# 学习计划 1：AI工程核心概念 — 从"图寻"项目理解

## 这个项目属于AI的哪个领域？

"图寻"项目横跨三个AI子领域：

| 领域 | 我们项目中的体现 | 一句话解释 |
|------|------------------|-----------|
| **计算机视觉 (CV)** | VLM的视觉编码器看街景图 | 让AI"看懂"图片 |
| **自然语言处理 (NLP)** | VLM的语言模型生成推理文本 | 让AI"说话和推理" |
| **多模态学习** | 视觉+语言融合的VLM | 让AI同时"看图"和"说话" |

更具体地说，我们做的是**视觉语言模型 (VLM) 的边缘部署**——把一个会看图会推理的大模型，压缩后放到Intel开发板上跑。

---

## 6个核心概念（结合项目代码理解）

### 概念1：模型 (Model)

**是什么：** 一个经过海量数据训练的数学函数，输入图片+文字，输出文字。

**项目中的位置：**
```
Qwen2-VL-2B-Instruct  ← 我们用的模型
├── 视觉编码器 (ViT)    ← 把图片变成数字向量
├── 语言模型 (7B/2B)    ← 把向量变成文字
└── 投影层              ← 连接视觉和语言的桥梁
```

**对应代码：** `Geocot.py:388-406` 的 `load_qwen2vl()`

**动手练习：**
1. 打开 `D:/Geocomp/scripts/test_real_image.py`
2. 找到 `Qwen2VLForConditionalGeneration.from_pretrained()` 这行
3. 试着改 `torch_dtype=torch.bfloat16` 为 `torch.float32`，观察加载速度和内存变化
4. 思考：为什么用bfloat16？答案：省一半内存，精度几乎不变

---

### 概念2：提示词 (Prompt)

**是什么：** 你给模型的"指令"，决定了模型会怎么回答。

**项目中的位置：**
```
GeoCoT的3阶段提示词
├── Stage 1 (宏观)  → "分析气候、植被、地形..."     → 模型回答大洲
├── Stage 2 (区域)  → "看路标语言、建筑风格..."      → 模型回答国家
└── Stage 3 (局部)  → "看人行道砖纹、地标..."        → 模型回答城市
```

**核心思想：** 同一个模型，不同的提示词，效果天差地别。零样本提示（"这是哪？"）准确率低，分阶段推理提示准确率高。

**对应代码：** `Geocot.py:66-98` 的 `_DEFAULT_PROMPTS` 和 `prompts/` 目录

**动手练习：**
1. 读 `prompts/local.txt`，注意最后一行的 `LOCATION:` 格式要求
2. 试着删掉 `LOCATION:` 那行，重新运行，观察 `extract_prediction()` 是否还能正确提取
3. 思考：为什么结构化输出格式对AI工程很重要？答案：让代码能可靠地解析模型输出

---

### 概念3：推理链 (Chain-of-Thought)

**是什么：** 让模型"一步一步想"，而不是直接给答案。就像人类解题要写过程。

**项目中的位置：**
```
GeoCoT = 地理推理链
图片 → [宏观推理] → "植被是热带的，地形平坦" →
      [区域推理] → "路标是中文，建筑是现代风格" →
      [局部推理] → "LOCATION: 深圳, 中国, 亚洲"
```

**为什么有效：** 大模型有"推理能力"，但需要被引导出来。直接问"这是哪"模型只能猜；分步引导，模型会注意到更多细节。

**对应代码：** `Geocot.py:314-333` 的 `GeoCoTPipeline.run()`

**动手练习：**
1. 运行 `run_geocot.py --image test.jpg`，观察3阶段输出
2. 对比：只用 Stage 3 的提示词（跳过1、2），看准确率是否下降
3. 思考：为什么人类玩GeoGuessr也是先看大方向再看细节？

---

### 概念4：部署 (Deployment)

**是什么：** 把训练好的模型放到目标设备上运行。

**项目中的挑战：**
```
开发环境 (你的笔记本)          目标环境 (DK-2500)
├── NVIDIA RTX 4060 (8GB)     ├── Intel Arc GPU (集成)
├── CUDA 加速                 ├── OpenVINO 加速
├── PyTorch 原生推理           ├── 需要格式转换
└── 内存充足                   └── 16GB 统一内存 (紧张)
```

**对应代码：** `src/inference/openvino_engine.py` 和 `device_manager.py`

**关键概念：** 模型格式转换（PyTorch → ONNX → OpenVINO IR），就像把Word文档转成PDF才能在另一台电脑打开。

---

### 概念5：量化 (Quantization)

**是什么：** 用更少的数字位数来存模型权重，牺牲微小精度换大幅内存节省。

**项目中的应用：**
```
原始模型 (bfloat16):  每个权重用16位存 → 2B模型 ≈ 4.4GB
INT8 量化:           每个权重用8位存  → 2B模型 ≈ 2.2GB
INT4 量化:           每个权重用4位存  → 2B模型 ≈ 1.1GB
```

**类比：** 就像图片压缩——JPEG比PNG小很多，但肉眼几乎看不出区别。

**对应代码：** `scripts/quantize_model.py`

**动手练习：**
1. 打开 `scripts/test_real_image.py`，看 `torch_dtype=torch.bfloat16` 这行
2. 思考：如果DK-2500只有16GB内存，OS占4GB，模型能多大？答案：~12GB，所以7B模型需要INT4量化

---

### 概念6：评估 (Evaluation)

**是什么：** 用客观指标衡量模型好不好，而不是"感觉还行"。

**项目中的指标：**
| 指标 | 含义 | 代码位置 |
|------|------|---------|
| 分类准确率 | 城市/国家/大洲猜对的比例 | `GeoClassificationMetrics.py` |
| 距离误差 | 预测位置和真实位置差多少公里 | `GeoDistanceChecker.py` |
| 语义相似度 | 推理过程和参考答案有多像 | `SemanticSimilarityEvaluator.py` |
| 幻觉率 | 模型编造了不存在的事实 | 人工标注 |

**对应代码：** `src/Geoeval/` 目录

---

## 学习路线

```
第1周：理解项目流程
  └─ 读懂 test_real_image.py → run_geocot.py → Geocot.py
  └─ 能跑通推理，理解每行代码的作用

第2周：补充基础知识
  └─ 学习 PyTorch 基础（tensor, model, forward）
  └─ 学习 Transformers 库的基础用法
  └─ 理解 VLM 的架构（ViT + LLM + 投影层）

第3-4周：按挑战深入
  └─ 量化：学 NNCF/OpenVINO 量化原理
  └─ 部署：学 ONNX/OpenVINO 模型转换
  └─ 提示词：学 prompt engineering 技术
```

---

## 推荐学习资源

| 资源 | 用途 | 时间 |
|------|------|------|
| [PyTorch官方教程](https://pytorch.org/tutorials/) 前3章 | Python深度学习基础 | 2天 |
| [HuggingFace Transformers教程](https://huggingface.co/learn/nlp-course) 第1-3章 | 理解模型加载和推理 | 1天 |
| [Qwen2-VL技术报告](https://arxiv.org/abs/2409.12191) | 理解我们用的模型 | 半天 |
| [OpenVINO文档](https://docs.openvino.ai/) | DK-2500部署必读 | 持续参考 |
| 本项目代码 | 最好的教材就是我们的代码 | 每天 |
