import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import h5py
import os

class LightBagDataset(Dataset):
    """轻量化包数据集 - 优化内存使用"""
    
    def __init__(self, csv_path, data_dir, max_bag_size=512, 
                 feature_dim=768, preload=False):
        super(LightBagDataset, self).__init__()
        
        # 读取CSV文件
        self.slide_data = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.max_bag_size = max_bag_size
        self.feature_dim = feature_dim
        self.preload = preload
        
        # 如果选择预加载且数据量不大，则预加载到内存
        self.cached_data = {}
        if preload and len(self.slide_data) < 100:  # 只对小数据集预加载
            self._preload_data()
    
    def _preload_data(self):
        """预加载数据到内存（仅对小数据集）"""
        print("预加载数据到内存...")
        for idx in range(len(self.slide_data)):
            slide_id = self.slide_data.iloc[idx]['slide_id']
            try:
                features = self._load_features(slide_id)
                if features is not None:
                    self.cached_data[slide_id] = features
            except:
                continue
        print(f"已预加载 {len(self.cached_data)} 个样本")
    
    def _load_features(self, slide_id):
        """加载特征数据"""
        # 尝试不同的文件格式
        possible_paths = [
            os.path.join(self.data_dir, f"{slide_id}.pt"),
            os.path.join(self.data_dir, f"{slide_id}.h5"),
            os.path.join(self.data_dir, f"{slide_id}.npy"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    if path.endswith('.pt'):
                        features = torch.load(path, map_location='cpu')
                    elif path.endswith('.h5'):
                        with h5py.File(path, 'r') as f:
                            features = torch.from_numpy(f['features'][:]).float()
                    elif path.endswith('.npy'):
                        features = torch.from_numpy(np.load(path)).float()
                    
                    # 限制特征数量以节省内存
                    if features.size(0) > self.max_bag_size:
                        # 随机采样或取前N个
                        indices = torch.randperm(features.size(0))[:self.max_bag_size]
                        features = features[indices]
                    
                    return features
                    
                except Exception as e:
                    print(f"加载 {path} 失败: {e}")
                    continue
        
        # 如果都失败了，返回随机特征作为占位符
        print(f"⚠️ 无法加载 {slide_id} 的特征，使用随机特征")
        return torch.randn(min(100, self.max_bag_size), self.feature_dim)
    
    def __len__(self):
        return len(self.slide_data)
    
    def __getitem__(self, idx):
        slide_id = self.slide_data.iloc[idx]['slide_id']
        label = int(self.slide_data.iloc[idx]['label'])
        
        # 从缓存或磁盘加载特征
        if slide_id in self.cached_data:
            features = self.cached_data[slide_id]
        else:
            features = self._load_features(slide_id)
        
        # 确保特征不为空
        if features is None or features.size(0) == 0:
            features = torch.randn(min(100, self.max_bag_size), self.feature_dim)
        
        return features, torch.tensor(label), slide_id

def create_light_dataloader(fold, dataset_name, cfg, result_dir, shuffle=False):
    """创建轻量化数据加载器"""
    
    # 构建数据路径
    if hasattr(cfg.Data, 'external_dir'):
        csv_path = cfg.Data.external_dir
    else:
        csv_path = os.path.join(cfg.Data.data_dir, f"{dataset_name}.csv")
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"⚠️ CSV文件不存在: {csv_path}")
        # 创建一个示例CSV文件
        sample_data = {
            'slide_id': [f'sample_{i}' for i in range(10)],
            'label': [0, 1] * 5
        }
        sample_df = pd.DataFrame(sample_data)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        sample_df.to_csv(csv_path, index=False)
        print(f"已创建示例CSV文件: {csv_path}")
    
    # 获取配置参数
    max_bag_size = getattr(cfg.Train, 'max_bag_size', 512) if hasattr(cfg, 'Train') else 512
    
    # 创建数据集
    dataset = LightBagDataset(
        csv_path=csv_path,
        data_dir=cfg.Data.data_dir,
        max_bag_size=max_bag_size,
        preload=False  # 对于大数据集关闭预加载
    )
    
    # 创建数据加载器 - 使用较小的batch_size
    batch_size = 1  # 包数据集通常使用batch_size=1
    num_workers = min(2, os.cpu_count())  # 限制worker数量以节省内存
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=light_collate_fn
    )
    
    print(f"✓ 创建数据加载器: {len(dataset)} 个样本")
    return dataloader

def light_collate_fn(batch):
    """轻量化的collate函数"""
    # 由于batch_size=1，直接返回第一个元素
    if len(batch) == 1:
        features, label, slide_id = batch[0]
        return features, label, [slide_id]
    else:
        # 如果batch_size > 1，需要处理不同大小的特征
        features_list = []
        labels = []
        slide_ids = []
        
        for features, label, slide_id in batch:
            features_list.append(features)
            labels.append(label)
            slide_ids.append(slide_id)
        
        # 对于多个包，返回列表
        return features_list, torch.stack(labels), slide_ids
