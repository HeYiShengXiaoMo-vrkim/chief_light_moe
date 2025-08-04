import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np

class Att_Head_Light(nn.Module):
    """轻量化注意力头"""
    def __init__(self, FEATURE_DIM, ATT_IM_DIM):
        super(Att_Head_Light, self).__init__()
        # 减少中间层维度
        mid_dim = min(ATT_IM_DIM // 2, 128)
        
        self.fc1 = nn.Linear(FEATURE_DIM, mid_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 添加dropout
        self.fc2 = nn.Linear(mid_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class Attn_Net_Light(nn.Module):
    """轻量化注意力网络"""
    def __init__(self, L=1024, D=128, dropout=True, n_classes=1):  # 减少D从256到128
        super(Attn_Net_Light, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.3))  # 增加dropout率

        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x

class Attn_Net_Gated_Light(nn.Module):
    """轻量化门控注意力网络"""
    def __init__(self, L=1024, D=128, dropout=True, n_classes=1):  # 减少维度
        super(Attn_Net_Gated_Light, self).__init__()
        
        # 注意力分支a
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        # 注意力分支b  
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        
        if dropout:
            self.attention_a.append(nn.Dropout(0.3))
            self.attention_b.append(nn.Dropout(0.3))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class CHIEF_Light(nn.Module):
    """轻量化CHIEF模型 - 适用于6GB显存"""
    def __init__(self, gate=True, size_arg="tiny", dropout=True, n_classes=2, 
                 anatomic=None, logits_field='bag_logits'):
        super(CHIEF_Light, self).__init__()
        
        # 轻量化的size配置
        self.size_dict_path = {
            "tiny": [768, 64, 1],      # 大幅减少参数 [input, hidden, output]
            "small": [768, 128, 1],    # 中等减少参数
            "big": [768, 256, 1]       # 原始大小
        }
        
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.3)]
        
        if dropout:
            fc.append(nn.Dropout(0.3))
            
        # 使用轻量化的注意力网络
        if gate:
            attention_net = Attn_Net_Gated_Light(L=size[1], D=size[1]//2, 
                                               dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net_Light(L=size[1], D=size[1]//2, 
                                         dropout=dropout, n_classes=1)
            
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        # 分类器
        bag_classifiers = [nn.Linear(size[1], 1) for _ in range(n_classes)]
        self.classifiers = nn.ModuleList(bag_classifiers)
        
        # 实例级分类器（轻量化）
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        
        self.k_sample = 8  # 减少采样数量从8到4
        self.n_classes = n_classes
        self.logits_field = logits_field
        
        # 初始化权重
        initialize_weights(self)

    def relocate(self):
        """将模型移动到设备"""
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    def forward(self, h, label=None, instance_eval=False, return_features=False, 
                attention_only=False):
        device = h.device
        
        # 限制输入大小以节省内存
        if h.size(0) > 1000:  # 如果patch数量太多，随机采样
            indices = torch.randperm(h.size(0))[:1000]
            h = h[indices]
        
        A, h = self.attention_net(h)  # A: N x n_classes, h: N x size[1]
        A = torch.transpose(A, 1, 0)  # A: n_classes x N
        
        if attention_only:
            return A
            
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N
        
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            for i in range(len(self.instance_classifiers)):
                inst_label = F.one_hot(label, num_classes=self.n_classes).squeeze()
                classifier = self.instance_classifiers[i]
                preds = classifier(h)
                all_preds.append(preds)
                instance_loss = F.cross_entropy(preds, inst_label)
                total_inst_loss += instance_loss

            if return_features:
                results_dict = {'instance_loss': total_inst_loss, 'inst_labels': label, 
                              'inst_preds': all_preds}
                return results_dict, h, A
            else:
                results_dict = {'instance_loss': total_inst_loss, 'inst_labels': label, 
                              'inst_preds': all_preds}
                return results_dict

        M = torch.mm(A, h)  # A: n_classes x N, h: N x size[1] -> M: n_classes x size[1]
        
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
            
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)
        
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': label, 
                          'inst_preds': all_preds}
            if return_features:
                return logits, Y_prob, Y_hat, A_raw, results_dict, M
            else:
                return logits, Y_prob, Y_hat, A_raw, results_dict
        else:
            if return_features:
                return logits, Y_prob, Y_hat, A_raw, M
            else:
                return logits, Y_prob, Y_hat, A_raw

    # 计算目标函数  
    def calculate_objective(self, X, Y):
        device = Y.device
        Y = Y.float()
        Y_hat, Y_prob, _, A_raw = self.forward(X)
        Y_hat = Y_hat.float()
        
        # 减少采样数量以节省内存
        sample_size = min(self.k_sample, A_raw.shape[1])
        if sample_size > 0:
            top_indices = torch.topk(A_raw[Y.long()], sample_size, dim=1)[1]
            top_features = X[top_indices]
        else:
            top_features = X
            
        return Y_hat, Y_prob

# 创建轻量化模型的工厂函数
def create_light_model(n_classes=2, size_arg="tiny", **kwargs):
    """创建轻量化CHIEF模型"""
    return CHIEF_Light(
        gate=True,
        size_arg=size_arg,
        dropout=True,
        n_classes=n_classes,
        **kwargs
    )
