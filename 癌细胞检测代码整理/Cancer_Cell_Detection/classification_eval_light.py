import argparse
import os
import torch
import pandas as pd
import numpy as np
import gc
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from utils.utils import read_yaml
from datasets.dataloader_factory import create_dataloader
from training_methods.embedding_general import evaluation
from models.CHIEF_Light import CHIEF_Light

def load_light_model(cfg):
    """加载轻量化模型"""
    # 使用轻量化模型
    model = CHIEF_Light(n_classes=cfg.Data.n_classes, size_arg="tiny", dropout=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式以节省内存
    return model

def optimize_memory():
    """内存优化"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def light_evaluation(fold, model, dataloader, result_dir, cfg):
    """轻量化评估函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    test_error = 0.
    test_loss = 0.
    all_probs = np.zeros((len(dataloader), cfg.Data.n_classes))
    all_labels = np.zeros(len(dataloader))
    
    slide_ids = dataloader.dataset.slide_data['slide_id']
    patient_results = {}
    
    with torch.no_grad():  # 禁用梯度计算节省内存
        for batch_idx, (data, label, slide_id) in enumerate(dataloader):
            # 内存优化：定期清理
            if batch_idx % 10 == 0:
                optimize_memory()
                
            data = data.to(device)
            label = label.to(device)
            
            # 限制输入大小
            if hasattr(cfg.Train, 'max_bag_size') and data.size(0) > cfg.Train.max_bag_size:
                indices = torch.randperm(data.size(0))[:cfg.Train.max_bag_size]
                data = data[indices]
            
            logits, Y_prob, Y_hat, A_raw = model(data)
            
            all_probs[batch_idx] = Y_prob.cpu().numpy()
            all_labels[batch_idx] = label.item()
            
            patient_results.update({slide_id[0]: {'slide_id': np.array(slide_id), 
                                                 'prob': Y_prob.cpu().numpy(), 
                                                 'label': label.item()}})
            
            error = calculate_error(Y_hat, label)
            test_error += error
            
            # 清理中间变量
            del data, logits, Y_prob, Y_hat, A_raw
            
    # 保存结果
    test_error /= len(dataloader)
    
    if len(np.unique(all_labels)) == 1:
        auc = -1
    else:
        if cfg.Data.n_classes == 2:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    # 保存结果到CSV
    df = pd.DataFrame({'slide_id': slide_ids, 'label': all_labels})
    for i in range(cfg.Data.n_classes):
        df[f'prob_{i}'] = all_probs[:, i]
    
    df.to_csv(os.path.join(result_dir, f'preds_{fold}.csv'), index=False)
    
    print(f'Test error: {test_error:.4f}, AUC: {auc:.4f}')
    return test_error, auc, df

def calculate_error(Y_hat, Y):
    """计算错误率"""
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
    return error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/colon_light.yaml')
    parser.add_argument('--dataset_name', type=str, default='test_set')
    parser.add_argument('--decimals', type=int, default=4)
    parser.add_argument('--use_mixed_precision', action='store_true', help='使用混合精度')
    args = parser.parse_args()
    
    decimals = args.decimals
    
    # 设置内存优化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False  # 禁用以节省内存
        torch.backends.cudnn.deterministic = True
    
    print("🚀 开始轻量化癌细胞检测...")
    print(f"配置文件: {args.config_path}")
    print(f"数据集: {args.dataset_name}")
    
    cfg = read_yaml(args.config_path)
    model = load_light_model(cfg)
    
    result_dir = os.path.join(cfg.General.result_dir, 'evaluation', args.dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"结果保存目录: {result_dir}")
    
    # 检查显存使用
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    fold = 0
    
    # 创建数据加载器
    dataloader = create_dataloader(fold, args.dataset_name, cfg, result_dir)
    
    # 加载预训练权重
    weight_path = '../../model_weight/CHIEF_pretraining.pth'
    if os.path.exists(weight_path):
        print(f"加载预训练权重: {weight_path}")
        try:
            # 尝试加载权重，如果不兼容则跳过某些层
            state_dict = torch.load(weight_path, map_location='cpu')
            model_dict = model.state_dict()
            
            # 过滤不匹配的键
            filtered_dict = {}
            for k, v in state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"跳过不匹配的层: {k}")
            
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)
            print("✓ 权重加载完成")
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")
            print("使用随机初始化权重")
    else:
        print(f"⚠️ 未找到预训练权重: {weight_path}")
        print("使用随机初始化权重")
    
    # 运行轻量化评估
    print("\n🔍 开始评估...")
    optimize_memory()
    
    try:
        if hasattr(cfg.Train, 'mixed_precision') and cfg.Train.mixed_precision and args.use_mixed_precision:
            print("使用混合精度评估")
            with torch.cuda.amp.autocast():
                test_error, auc, df = light_evaluation(fold, model, dataloader, result_dir, cfg)
        else:
            test_error, auc, df = light_evaluation(fold, model, dataloader, result_dir, cfg)
            
        # 计算其他指标
        if len(np.unique(df['label'])) > 1:
            if cfg.Data.n_classes == 2:
                y_true = df['label'].values
                y_pred = (df['prob_1'].values > 0.5).astype(int)
                y_score = df['prob_1'].values
                
                f1 = f1_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                accuracy = accuracy_score(y_true, y_pred)
                
                print(f"\n📊 评估结果:")
                print(f"AUC: {auc:.{decimals}f}")
                print(f"F1-Score: {f1:.{decimals}f}")
                print(f"Precision: {precision:.{decimals}f}")
                print(f"Recall: {recall:.{decimals}f}")
                print(f"Accuracy: {accuracy:.{decimals}f}")
                print(f"Test Error: {test_error:.{decimals}f}")
                
                # 保存指标
                metrics = {
                    'auc': auc,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'test_error': test_error
                }
                
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(os.path.join(result_dir, 'metrics.csv'), index=False)
                print(f"\n✓ 结果已保存到: {result_dir}")
                
        else:
            print("⚠️ 标签只有一个类别，无法计算某些指标")
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("❌ GPU内存不足，尝试以下解决方案:")
            print("1. 减小配置文件中的 batch_size")
            print("2. 减小配置文件中的 max_bag_size")
            print("3. 使用 --use_mixed_precision 参数")
            print("4. 关闭其他占用GPU的程序")
        else:
            print(f"❌ 运行错误: {e}")
    
    finally:
        optimize_memory()
        print("\n🏁 评估完成")

if __name__ == '__main__':
    main()
