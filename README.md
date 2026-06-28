# 图寻 — 离线视觉地理定位系统

Intel Cup Embedded AI Competition 参赛项目。面向 DK-2500 嵌入式平台的户外离线定位系统，包含**传感器轨迹定位**（主体/落地）与**视觉语义推理**（创新/潜力）两条互补流水线。

## 项目结构

```
图寻/
├── sensor_localization/        # 【主体】传感器定位流水线 — 落地可行
│   ├── hiking_route_localization/  # ROS 路径匹配引擎 (Python)
│   ├── VINS-Fusion/                # VIO 视觉-惯性里程计 (C++)
│   ├── BME280_Code/                # BME280 气压计 STM32 固件 (C)
│   └── Mobile-GVIO-calib/         # 相机标定参数
│
├── vision_reasoning/           # 【创新】视觉推理定位流水线 — 潜力展望
│   ├── src/
│   │   ├── geovlm/                 # GeoVLM 430M 轻量视觉语言模型
│   │   ├── Geocot/                 # GeoCoT 地理思维链推理框架
│   │   ├── regression/             # Fusion Engine v2.2 元素融合引擎
│   │   └── utils/
│   ├── scripts/                    # 标注/蒸馏/训练/测试脚本
│   ├── config/                     # 模型配置
│   └── requirements.txt
│
├── docs/                        # 项目文档
│   └── 项目书.txt
├── models/                      # 模型权重
├── output/                      # 输出结果
├── data/                        # 数据集
└── README.md
```

## 一、传感器定位流水线 (主体)

**思路**：传感器数据采集 → VIO 轨迹重建 → 与 GPX 路径库匹配 → 定位

这是实际落地可行的方案，已具备完整的硬件-软件链路：

| 组件 | 功能 | 技术栈 |
|------|------|--------|
| `hiking_route_localization/` | ROS 路径匹配引擎 | Python, ROS, GPX |
| `VINS-Fusion/` | 视觉-惯性里程计 | C++, ROS, Ceres |
| `BME280_Code/` | 气压高度计嵌入式固件 | C, STM32F103, I2C |
| `Mobile-GVIO-calib/` | 相机-IMU 标定参数 | YAML |

**ROS 路径匹配节点** (hiking_route_localization/scripts/)：
- `nmea_gps_node.py` — NMEA GPS 串口读取
- `serial_altitude_node.py` / `bme280_altitude_node.py` — 气压高度采集
- `route_matcher_node.py` — 实时轨迹与 GPX 路径匹配
- `route_visualizer_node.py` — RViz 可视化
- `gpx_to_route.py` — GPX → route 格式转换
- `demo_sensor_player.py` — 离线传感器数据回放

**运行方式**：
```bash
# 完整传感器定位流程
roslaunch hiking_route_localization hiking_full.launch
# 或演示模式（使用预录数据）
roslaunch hiking_route_localization demo_route_matcher.launch
```

## 二、视觉推理定位流水线 (创新)

**思路**：单张图像 → GeoCoT 多尺度地理推理 → 坐标预测

这是具有学术创新性的方案，探索用纯视觉语义推理实现零先验知识的定位：

| 组件 | 功能 | 技术栈 |
|------|------|--------|
| `GeoCoT` | 地理思维链推理 (宏观→中观→微观) | Python, VLMs |
| `GeoVLM 430M` | 自研轻量地理视觉模型 | PyTorch, Q-Former, GNN |
| `Fusion Engine v2.2` | 多源特征融合 + 坐标回归 | Python, GeoKB |

**特点**：无需预先知道目标区域的路径/地图，仅凭图像中的地理线索推理位置。模型轻量化后可部署于 DK-2500。

**当前状态**：框架代码完成，模型蒸馏训练中（~88% 标注质量合格），尚未达到可部署水平。

## 双流水线关系

```
传感器定位 (主体)                    视觉推理 (创新)
─────────────────                   ────────────────
输入: GPS + IMU + 气压 + 相机       输入: 单张图像
输出: 轨迹 → 路径匹配 → 精确定位     输出: 语义推理 → 区域预测
依赖: 需要目标区域预存 GPX 路径      依赖: 无（零先验知识）
精度: 高（亚米级）                   精度: 中（区域级）
成熟度: 落地可用                     成熟度: 研究阶段
```

两条流水线**互补融合**：视觉推理先给出大致地理区域 → 传感器流水线在区域内精确匹配路径 → 快速精确定位。

## 代码统计

| 流水线 | 语言 | 代码量 | 定位 |
|--------|------|--------|------|
| sensor_localization | C++/Python/C | ~63,000 行 | **主体** |
| vision_reasoning | Python | ~25,000 行 | 创新 |
| **总计** | | **~88,000 行** | |

## 目标硬件

DK-2500 嵌入式 AI 平台。传感器流水线通过串口/I2C/USB 接入 BME280、GPS、IMU、相机等外设；视觉流水线 GeoVLM 430M 经 INT8 量化后部署于 NPU。

## License

MIT
