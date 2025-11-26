# CHIEF Light MoE - Cancer Cell Detection System

<div align="center">

**基于深度学习的癌细胞自动检测系统**  
*Automated Cancer Cell Detection System Based on Deep Learning*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)

</div>

---

## 📖 项目简介 | Project Overview

CHIEF Light MoE 是一个先进的癌细胞检测系统，专门用于病理切片图像（WSI）中的癌细胞识别。基于多实例学习（MIL）和注意力机制，提供标准版和轻量化版两种配置，支持不同硬件环境。

CHIEF Light MoE is an advanced cancer cell detection system specifically designed for identifying cancer cells in whole slide images (WSI). Based on Multiple Instance Learning (MIL) and attention mechanisms, it offers both standard and lightweight versions to support different hardware environments.

## 🎯 主要特性 | Key Features

### ✨ 双版本支持 | Dual Version Support
- **标准版**: 追求最佳检测性能（需要8GB+显存）
- **轻量化版**: 6GB显存可运行，参数减少80%，推理速度提升20%

- **Standard Version**: Best detection performance (requires 8GB+ VRAM)
- **Lightweight Version**: Runs on 6GB VRAM, 80% fewer parameters, 20% faster inference

### 🔬 支持的癌症类型 | Supported Cancer Types
- ✅ 结肠癌 | Colon Cancer
- ✅ 乳腺癌 | Breast Cancer
- 🔧 可扩展其他类型 | Extensible to other types

### 🚀 核心技术 | Core Technologies
- **多实例学习** (MIL): 处理包级别标签
- **注意力机制**: 自动学习重要区域
- **迁移学习**: 使用CHIEF预训练模型
- **混合精度训练**: 节省显存，加速训练

- **Multiple Instance Learning** (MIL): Handle bag-level labels
- **Attention Mechanism**: Automatically learn important regions
- **Transfer Learning**: Using CHIEF pre-trained model
- **Mixed Precision Training**: Save VRAM and accelerate training

## 📁 项目结构 | Project Structure

```
chief_light_moe/
├── 📄 README.md                        # 本文件
├── 📊 项目分析报告.md                   # 详细中文分析报告
├── 📊 PROJECT_ANALYSIS.md              # Detailed English analysis report
├── 📜 LICENSE                          # Apache 2.0
│
└── 📁 癌细胞检测代码整理/
    ├── 🚀 quick_start.py               # 快速启动脚本
    ├── 📋 requirements.txt             # Python依赖
    ├── 📝 README.md                    # 功能说明
    ├── 📘 癌细胞检测详细说明.md          # 技术文档
    ├── 📗 轻量化说明.md                  # 轻量化版本说明
    │
    └── 📁 Cancer_Cell_Detection/
        ├── 🔬 classification_eval.py       # 标准版评估
        ├── ⚡ classification_eval_light.py  # 轻量化评估
        ├── 📁 configs/                     # 配置文件
        ├── 📁 models/                      # 模型定义
        ├── 📁 datasets/                    # 数据处理
        ├── 📁 training_methods/            # 训练方法
        └── 📁 utils/                       # 工具函数
```

## 🚀 快速开始 | Quick Start

### 1️⃣ 环境安装 | Environment Setup

```bash
# 克隆仓库 | Clone repository
git clone <repository-url>
cd chief_light_moe/癌细胞检测代码整理

# 安装依赖 | Install dependencies
pip install -r requirements.txt

# 检查环境 | Check environment
python quick_start.py --check-env
```

### 2️⃣ 运行检测 | Run Detection

#### 方法1: 使用快速启动脚本 | Using Quick Start Script
```bash
# 结肠癌检测 | Colon cancer detection
python quick_start.py --run colon

# 乳腺癌检测 | Breast cancer detection
python quick_start.py --run breast
```

#### 方法2: Windows批处理 | Windows Batch Script
```bash
# 进入项目目录 | Enter project directory
cd 癌细胞检测代码整理

# 标准版 | Standard version
run_detection.bat

# 轻量化版（6GB显存推荐） | Lightweight version (6GB VRAM recommended)
run_light_detection.bat
```

#### 方法3: 直接运行 | Direct Execution
```bash
cd Cancer_Cell_Detection

# 轻量化版本（推荐） | Lightweight version (recommended)
python classification_eval_light.py \
    --config_path configs/colon_light.yaml \
    --dataset_name Dataset_PT \
    --use_mixed_precision
```

## 📊 性能对比 | Performance Comparison

| 指标 Metric | 标准版 Standard | 轻量化版 Lightweight | 优化 Optimization |
|------------|----------------|---------------------|------------------|
| 模型参数 Parameters | ~50M | ~10M | **-80%** |
| 显存占用 VRAM | 8-12GB | 4-6GB | **-50%** |
| 推理速度 Speed | 基准 Baseline | +20% | **Faster** |
| 准确率 Accuracy | 基准 Baseline | -2~5% | Slight decrease |

## 💻 硬件要求 | Hardware Requirements

### 轻量化版本 | Lightweight Version (推荐 Recommended)
- **GPU**: RTX 4060/3060 (6GB VRAM)
- **RAM**: 8GB
- **存储 Storage**: 5GB+

### 标准版本 | Standard Version
- **GPU**: RTX 3080/4070+ (8GB+ VRAM)
- **RAM**: 16GB+
- **存储 Storage**: 10GB+

## 📚 文档 | Documentation

- 📊 **[项目分析报告.md](项目分析报告.md)** - 完整的项目技术分析（中文）
- 📊 **[PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md)** - Complete technical analysis (English)
- 📝 **[癌细胞检测详细说明.md](癌细胞检测代码整理/癌细胞检测详细说明.md)** - 详细技术文档
- 📗 **[轻量化说明.md](癌细胞检测代码整理/轻量化说明.md)** - 轻量化版本优化说明

## 🎓 技术原理 | Technical Principles

### 多实例学习 (MIL)
```
WSI 全切片图像
    ↓
Patches 切割成小块
    ↓
特征提取 (CHIEF模型)
    ↓
注意力聚合
    ↓
分类预测 (正常/癌症)
```

### 关键技术 | Key Technologies
- **门控注意力机制**: 自适应学习重要patch权重
- **特征聚合**: 将patch级特征聚合为WSI级特征
- **迁移学习**: 利用CHIEF预训练权重

## 🔧 配置示例 | Configuration Example

### 轻量化配置 | Lightweight Configuration
```yaml
Model:
    size_arg: tiny              # 使用最小模型
    dropout: 0.3               # 增强正则化

Train:
    batch_size: 8              # 减小批量大小
    mixed_precision: true      # 混合精度训练
    max_bag_size: 512         # 限制patch数量
    gradient_accumulation_steps: 4  # 梯度累积
```

## 📈 评估指标 | Evaluation Metrics

系统输出以下评估指标：
The system outputs the following evaluation metrics:

- **AUC**: ROC曲线下面积 | Area Under ROC Curve
- **F1-Score**: F1分数 | F1 Score
- **Precision**: 精确率 | Precision
- **Recall**: 召回率 | Recall
- **Accuracy**: 准确率 | Accuracy

## 🐛 常见问题 | Common Issues

<details>
<summary><b>Q: CUDA内存不足怎么办？ | CUDA Out of Memory?</b></summary>

**解决方案 | Solutions**:
1. 使用轻量化版本 | Use lightweight version
2. 减小batch_size | Reduce batch_size
3. 启用混合精度训练 | Enable mixed precision training
4. 限制max_bag_size | Limit max_bag_size

```yaml
Train:
    batch_size: 4              # 进一步减小
    max_bag_size: 256         # 进一步限制
```
</details>

<details>
<summary><b>Q: 如何扩展新的癌症类型？ | How to extend to new cancer types?</b></summary>

1. 准备数据和标签 | Prepare data and labels
2. 创建配置文件 | Create configuration file
3. 运行训练/评估 | Run training/evaluation

详见分析报告 | See analysis report for details
</details>

## 🤝 贡献 | Contributing

欢迎贡献！请查看贡献指南。
Contributions welcome! Please check the contribution guidelines.

### 贡献方式 | How to Contribute
- 🐛 报告Bug | Report bugs
- ✨ 提出新功能 | Propose new features
- 📝 改进文档 | Improve documentation
- 🎨 优化代码 | Optimize code

## 📄 许可证 | License

本项目采用 [Apache License 2.0](LICENSE) 许可证。
This project is licensed under [Apache License 2.0](LICENSE).

## 📞 联系方式 | Contact

- **GitHub**: [@HeYiShengXiaoMo-vrkim](https://github.com/HeYiShengXiaoMo-vrkim)
- **项目地址 | Repository**: [chief_light_moe](https://github.com/HeYiShengXiaoMo-vrkim/chief_light_moe)

## 🌟 致谢 | Acknowledgments

感谢所有为病理学AI研究做出贡献的研究者和开发者。
Thanks to all researchers and developers contributing to pathology AI research.

---

<div align="center">

**CHIEF Light MoE - 让癌症检测更智能、更高效**  
*Making Cancer Detection Smarter and More Efficient*

⭐ 如果这个项目对您有帮助，请给我们一个星标！  
⭐ If this project helps you, please give us a star!

</div>