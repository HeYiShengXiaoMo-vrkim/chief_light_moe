# 癌细胞检测 (Cancer Cell Detection)

## 🎯 功能简介
基于CHIEF深度学习模型的癌细胞自动检测系统，支持在病理切片图像中准确识别癌细胞。

## 🚀 支持的癌症类型
- **结肠癌** (Colon Cancer)
- **乳腺癌** (Breast Cancer)  
- 可扩展支持其他癌症类型

## 📁 文件结构
```
癌细胞检测代码整理/
├── Cancer_Cell_Detection/        # 核心检测模块
│   ├── classification_eval.py    # 主评估脚本
│   ├── configs/                  # 配置文件
│   ├── datasets/                 # 数据处理
│   ├── models/                   # CHIEF模型
│   └── utils/                    # 工具函数
├── quick_start.py               # Python快速入门脚本
├── run_detection.bat            # Windows批处理脚本
├── 癌细胞检测详细说明.md         # 详细技术文档
└── requirements.txt             # 依赖包列表
```

## ⚡ 快速开始

### 🔥 轻量化版本 (6GB显存推荐)
```bash
# 一键启动轻量化检测
run_light_detection.bat

# 或手动运行
cd Cancer_Cell_Detection
python classification_eval_light.py --config_path configs/colon_light.yaml --use_mixed_precision
```

### 方法1: 使用批处理脚本 (Windows推荐)
```bash
# 双击运行或在命令行执行
run_detection.bat          # 标准版本
run_light_detection.bat    # 轻量化版本 (6GB显存)
```

### 方法2: 使用Python脚本
```bash
# 检查环境
python quick_start.py --check-env

# 运行结肠癌检测
python quick_start.py --run colon

# 运行乳腺癌检测  
python quick_start.py --run breast
```

### 方法3: 直接运行
```bash
cd Cancer_Cell_Detection

# 标准版本
python classification_eval.py --config_path configs/colon.yaml --dataset_name Dataset_PT

# 轻量化版本 (推荐6GB显存用户)
python classification_eval_light.py --config_path configs/colon_light.yaml --use_mixed_precision
```

## 📋 环境要求

### 💻 硬件要求

#### 轻量化版本 (推荐配置)
- **GPU**: RTX 4060/3060 (6GB显存)
- **内存**: 8GB
- **存储**: 5GB+ 可用空间

#### 标准版本
- **GPU**: RTX 3080/4070+ (8GB+显存)
- **内存**: 16GB+
- **存储**: 10GB+ 可用空间

### 🛠️ 软件依赖
- Python 3.8+
- PyTorch (CUDA支持推荐)
- pandas, numpy, scikit-learn
- PyYAML

### 📦 安装依赖
```bash
pip install -r requirements.txt
```

## 📊 输出结果

### 评估指标
- **AUC**: ROC曲线下面积
- **精确率**: Precision  
- **召回率**: Recall
- **F1分数**: F1-score
- **准确率**: Accuracy

### 结果文件
- `results/{dataset}/evaluation/{dataset_name}/preds_0.csv`
  - 包含每个样本的预测概率和真实标签

## 🔧 配置说明

### 结肠癌配置 (configs/colon.yaml)
```yaml
Data:
    n_classes: 2                    # 二分类
    data_dir: ./feature/Dataset_PT/  # 特征数据
    anatomic: 13                    # 结肠部位编码
```

### 乳腺癌配置 (configs/breast.yaml)
```yaml
Data:
    n_classes: 2
    data_dir: ./feature/DROID_breast/
    anatomic: 1                     # 乳腺部位编码
```

## 📈 扩展新癌症类型

### 1. 准备数据
- 病理图像特征文件
- CSV格式标签文件

### 2. 创建配置
```yaml
General:
    result_dir: ./results/new_cancer
Data:
    data_dir: ./feature/new_cancer/
    external_dir: ./csv/new_cancer.csv
    anatomic: X  # 新部位编码
```

### 3. 运行检测
```bash
python classification_eval.py --config_path configs/new_cancer.yaml
```

## ⚠️ 注意事项

1. **模型权重**: 需要下载CHIEF预训练权重文件
2. **数据格式**: 确保数据按指定格式组织
3. **GPU内存**: 如遇内存不足，可减小batch_size
4. **路径设置**: 配置文件中的路径需要正确设置

## 🐛 常见问题

### Q: CUDA内存不足
**A**: 在配置文件中减小 `batch_size`

### Q: 找不到模型权重
**A**: 确保 `CHIEF_pretraining.pth` 在正确路径

### Q: 数据加载失败  
**A**: 检查数据路径和CSV文件格式

## 🔬 模型可靠性说明

### 准确性保障
- **AUC值**: 0.92-0.97 (不同癌症类型)
- **准确率**: 90-95%
- **敏感性**: 85-92%
- **特异性**: 88-94%

### 训练数据规模
- **WSI数量**: 60,000+ 全切片图像
- **patch数量**: 1.5亿+ 图像块
- **癌症类型**: 32种常见癌症
- **数据来源**: TCGA、CPTAC等权威数据库 + 多家三甲医院

### 可解释性支持
- **注意力机制生成可解释性热力图**: 直观显示模型关注的关键区域
- **辅助医生诊断**: 快速定位可疑区域，减少遗漏
- **决策透明**: 可视化模型判断依据，增强可信度

详细说明请参考: `癌细胞检测详细说明.md`

## 📞 技术支持

完整技术文档:
- `癌细胞检测详细说明.md` - 深入的技术原理和使用说明
- `轻量化说明.md` - 6GB显存优化版本说明

---
**CHIEF Cancer Cell Detection System**  
*Powered by Deep Learning & Pathology AI*
