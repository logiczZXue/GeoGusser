# 学习计划 2：代码精读路线 — 读懂"图寻"每一行代码

## 目标

从零开始，逐步读懂项目中每一行关键代码。不是走马观花，而是理解**为什么这样写**。

---

## 第1站：入口脚本 (1小时)

### 文件：`D:/Geocomp/src/Geocot/run_geocot.py`

这是整个系统的入口，最简单也最重要。

**阅读顺序：**
```
第22行 main()            ← 程序从哪开始？
  ↓
第33行 load_qwen2vl()    ← 模型怎么加载的？
  ↓
第40行 GeoCoTPipeline()  ← 推理流水线怎么建？
  ↓
第62行 pipeline.run()    ← 怎么跑推理？
  ↓
第66-68行 打印+保存      ← 结果怎么输出？
```

**关键问题（读完后试着回答）：**
1. `prompts_dir` 变量指向哪个目录？为什么要把提示词放在单独的文件里？
2. `os.makedirs(..., exist_ok=True)` 中 `exist_ok=True` 是什么意思？去掉会怎样？
3. 为什么用 `Image.open(img_path).convert("RGB")` 而不是直接 `Image.open(img_path)`？

---

## 第2站：核心流水线 (2小时)

### 文件：`D:/Geocomp/src/Geocot/Geocot.py`

这是最核心的文件，分为4个区域。

### 区域A：数据结构 (第28-56行)

```python
GeoCoTStage   ← 枚举，3个阶段的名字
GeoPrediction ← 预测结果，有city/country/continent三个字段
GeoCoTResult  ← 完整结果，包含3阶段输出和最终预测
```

**为什么用 dataclass？** 看第35-41行，`@dataclass` 让我们不用手写 `__init__`，Python自动生成。对比手写：
```python
# 不用dataclass要写这么多：
class GeoPrediction:
    def __init__(self, city="Unknown", country="Unknown", continent="Unknown"):
        self.city = city
        self.country = country
        self.continent = continent
    def __str__(self): ...

# 用dataclass一行搞定：
@dataclass
class GeoPrediction:
    city: str = "Unknown"
    country: str = "Unknown"
    continent: str = "Unknown"
```

### 区域B：提示词管理 (第63-145行)

**阅读重点：** `get_prompt()` 方法（第123-136行）

```python
def get_prompt(self, stage, prev_outputs):
    template = self._prompts[stage]       # 拿到该阶段的提示词模板
    fmt = {}                               # 准备填充的变量
    if "{prev_output}" in template:        # 如果模板有占位符
        fmt["prev_output"] = prev_outputs[...]  # 用上阶段输出填充
    return template.format(**fmt)          # 字符串格式化
```

**关键概念：** 模板字符串。`"{name}你好".format(name="张三")` → `"张三你好"`。

**动手练习：** 打开 `prompts/regional.txt`，找到 `{prev_output}`，理解它会被什么内容替换。

### 区域C：预测提取 (第199-281行)

这是最复杂的部分，因为模型输出不可控。

**阅读策略：从外到内**

```python
extract_prediction(text)
  ↓
第1步：找 "LOCATION:" 标记     ← 最可靠，我们要求模型输出这个
  ↓ 找不到？
第2步：用正则匹配 "taken in City, Country, Continent"
  ↓ 找不到？
第3步：在结论中找地标名       ← "埃菲尔铁塔" → Paris, France
  ↓ 找不到？
第4步：在结论中找城市名       ← "tokyo" → Tokyo, Japan
  ↓ 找不到？
第5步：在全文找地标名         ← 最后的兜底
  ↓ 都找不到？
返回 Unknown, Unknown, Unknown
```

**为什么这么复杂？** 因为大模型输出不固定——有时候按要求写LOCATION，有时候写成"我认为这是深圳"，有时候英文有时候中文。代码必须处理各种情况。

**关键问题：**
1. 第243-246行为什么只扫结论（最后一句）而不是全文？（提示：避免匹配到推理过程中提到的城市）
2. 第266行为什么要 `sorted(..., key=lambda x: -len(x[0]))`？（提示："new york"比"new"长，先匹配长的避免误匹配）

### 区域D：VLM后端 (第340-406行)

**阅读重点：** `model_fn()` 函数（第353-383行）

这是模型实际推理的地方，每一行都有原因：

```python
# 第354-357行：构造对话消息
messages = [{"role": "user", "content": [图片, 文字]}]

# 第359行：套用聊天模板（加<|im_start|>等特殊标记）
text = processor.apply_chat_template(messages, ...)

# 第360行：提取图片数据
image_inputs, video_inputs = process_vision_info(messages)

# 第361-365行：把文字+图片一起编码成模型输入
inputs = processor(text=..., images=..., ...)

# 第369行：清GPU缓存，防止OOM
torch.cuda.empty_cache()

# 第371-377行：生成回答
output_ids = model.generate(**inputs, temperature=0.7, ...)

# 第380-382行：只解码新生成的部分（跳过输入）
result = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], ...)
```

**关键问题：**
1. 第369行 `torch.cuda.empty_cache()` 为什么要在 generate 前调用？
2. 第380行 `output_ids[:, inputs.input_ids.shape[1]:]` 为什么要切片？不切片会怎样？
3. `temperature=0.7` 改成 `0.01` 会怎样？改成 `2.0` 呢？

---

## 第3站：提示词文件 (30分钟)

### 目录：`D:/Geocomp/src/Geocot/prompts/`

按顺序读：`macro.txt` → `regional.txt` → `local.txt`

**关注点：**
1. 每个阶段关注什么地理线索？为什么这样分层？
2. `regional.txt` 中的 `{prev_output}` 和 `local.txt` 中的 `{macro_output}` `{regional_output}` 分别会被什么替换？
3. `local.txt` 末尾的 `LOCATION:` 格式要求为什么至关重要？

---

## 第4站：评估代码 (1小时)

### 目录：`D:/Geocomp/src/Geoeval/`

按这个顺序读：
1. `GeoClassificationMetrics.py` — 最简单，数对/错
2. `GeoDistanceChecker.py` — 算经纬度距离
3. `SemanticSimilarityEvaluator.py` — 用向量相似度评估推理质量

---

## 第5站：部署代码 (1小时)

### 文件：`src/inference/openvino_engine.py`, `device_manager.py`

这些代码目前还没在DK-2500上测试过，但结构清晰：
- `device_manager.py`：检测硬件，分配组件到CPU/GPU/NPU
- `openvino_engine.py`：用OpenVINO替代PyTorch做推理

**类比理解：** PyTorch是Windows，OpenVINO是Linux——同样跑程序，但系统不同，需要适配。

---

## 总时间：约6小时

建议分2-3天完成，每天2小时。每读完一个文件，试着用自己的话复述它做了什么。
