#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
癌细胞检测快速入门脚本
演示如何使用CHIEF进行癌细胞检测
"""

import os
import argparse
import subprocess
import sys

def print_banner():
    """打印项目横幅"""
    print("="*60)
    print("    CHIEF 癌细胞检测 (Cancer Cell Detection)")
    print("="*60)
    print("    检测病理图像中的癌细胞")
    print("    支持结肠癌、乳腺癌等多种癌症类型")
    print("="*60)
    print()

def check_environment():
    """检查环境依赖"""
    print("🔍 检查环境...")
    
    requirements = []
    
    try:
        import torch
        print(f"✓ PyTorch 版本: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA 可用: {cuda_available}")
        if cuda_available:
            print(f"✓ GPU 数量: {torch.cuda.device_count()}")
            print(f"✓ 当前GPU: {torch.cuda.current_device()}")
        else:
            print("⚠️  警告: CUDA不可用，将使用CPU (速度较慢)")
    except ImportError:
        print("❌ PyTorch 未安装")
        requirements.append("torch")
    
    try:
        import pandas as pd
        print(f"✓ Pandas 版本: {pd.__version__}")
    except ImportError:
        print("❌ Pandas 未安装")
        requirements.append("pandas")
    
    try:
        import numpy as np
        print(f"✓ NumPy 版本: {np.__version__}")
    except ImportError:
        print("❌ NumPy 未安装")
        requirements.append("numpy")
        
    try:
        from sklearn.metrics import roc_auc_score
        print(f"✓ Scikit-learn 可用")
    except ImportError:
        print("❌ Scikit-learn 未安装")
        requirements.append("scikit-learn")
    
    if requirements:
        print(f"\n📦 需要安装的包: {', '.join(requirements)}")
        print("运行: pip install " + " ".join(requirements))
    
    print()

def check_model_weights():
    """检查模型权重文件"""
    print("🔍 检查模型权重...")
    
    weight_path = "../../model_weight/CHIEF_pretraining.pth"
    if os.path.exists(weight_path):
        size = os.path.getsize(weight_path) / (1024*1024)  # MB
        print(f"✓ 找到预训练权重: {weight_path} ({size:.1f} MB)")
    else:
        print(f"❌ 未找到预训练权重: {weight_path}")
        print("请下载CHIEF预训练模型权重文件")
    
    print()

def check_data_structure():
    """检查数据结构"""
    print("🔍 检查数据结构...")
    
    data_dirs = [
        "./feature/Dataset_PT/",      # 结肠癌特征
        "./feature/DROID_breast/",    # 乳腺癌特征
        "./csv/Dataset_PT.csv",       # 结肠癌标签
        "./csv/DROID_breast.csv"      # 乳腺癌标签
    ]
    
    for data_path in data_dirs:
        if os.path.exists(data_path):
            print(f"✓ 找到数据: {data_path}")
        else:
            print(f"❌ 未找到数据: {data_path}")
    
    print()

def show_available_configs():
    """显示可用的配置文件"""
    print("📋 可用配置:")
    
    config_dir = "./Cancer_Cell_Detection/configs/"
    if os.path.exists(config_dir):
        configs = [f for f in os.listdir(config_dir) if f.endswith('.yaml')]
        for config in configs:
            print(f"  - {config}")
            
            # 读取配置文件显示简要信息
            config_path = os.path.join(config_dir, config)
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'colon' in config.lower():
                        print(f"    用途: 结肠癌检测")
                    elif 'breast' in config.lower():
                        print(f"    用途: 乳腺癌检测")
            except:
                pass
    else:
        print(f"❌ 配置目录不存在: {config_dir}")
    
    print()

def run_detection(cancer_type):
    """运行癌细胞检测"""
    print(f"🚀 开始运行 {cancer_type} 检测...")
    
    if cancer_type == "colon":
        config_path = "./Cancer_Cell_Detection/configs/colon.yaml"
        dataset_name = "Dataset_PT"
    elif cancer_type == "breast":
        config_path = "./Cancer_Cell_Detection/configs/breast.yaml"  
        dataset_name = "DROID_breast"
    else:
        print(f"❌ 不支持的癌症类型: {cancer_type}")
        return
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 构建命令
    cmd = [
        sys.executable,  # python
        "./Cancer_Cell_Detection/classification_eval.py",
        "--config_path", config_path,
        "--dataset_name", dataset_name
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # 运行检测
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ 检测完成!")
            print("输出:")
            print(result.stdout)
        else:
            print("❌ 检测失败!")
            print("错误信息:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ 运行错误: {e}")

def main():
    parser = argparse.ArgumentParser(description="CHIEF 癌细胞检测快速入门")
    parser.add_argument("--check-env", action="store_true", help="检查环境依赖")
    parser.add_argument("--check-weights", action="store_true", help="检查模型权重")
    parser.add_argument("--check-data", action="store_true", help="检查数据结构")
    parser.add_argument("--show-configs", action="store_true", help="显示可用配置")
    parser.add_argument("--run", choices=["colon", "breast"], help="运行指定类型的检测")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.check_env:
        check_environment()
    
    if args.check_weights:
        check_model_weights()
    
    if args.check_data:
        check_data_structure()
    
    if args.show_configs:
        show_available_configs()
    
    if args.run:
        run_detection(args.run)
    
    if not any([args.check_env, args.check_weights, args.check_data, 
                args.show_configs, args.run]):
        print("🎯 癌细胞检测功能:")
        print("  1. 结肠癌检测 - 在结肠病理图像中检测癌细胞")
        print("  2. 乳腺癌检测 - 在乳腺病理图像中检测癌细胞")
        print()
        print("📚 使用方法:")
        print("  python quick_start.py --check-env      # 检查环境")
        print("  python quick_start.py --check-weights  # 检查模型权重")
        print("  python quick_start.py --check-data     # 检查数据")
        print("  python quick_start.py --show-configs   # 显示配置")
        print("  python quick_start.py --run colon      # 运行结肠癌检测")
        print("  python quick_start.py --run breast     # 运行乳腺癌检测")
        print()
        print("📖 详细说明请查看: 癌细胞检测详细说明.md")

if __name__ == "__main__":
    main()
