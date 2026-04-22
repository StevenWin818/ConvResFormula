"""
SIGReg-like 损失函数实现
基于 LeWorldModel 的机制，使用随机投影 + 正态性匹配来正则化 embedding 分布
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union, Any


class SIGRegLikeLoss(nn.Module):
    """
    SIGReg-like 正则化损失
    
    原理：
    1. 对输入 embedding 进行 M 次随机方向投影，得到 M 个 1D 标量序列
    2. 对每个 1D 序列计算 Epps-Pulley 正态性统计量（0~1，1表示完全正态）
    3. 平均统计量，目标是最大化正态性（使其接近 1），从而让 embedding 分布接近 N(0, I)
    4. 损失 = 1 - 平均正态性统计量（最小化损失 = 最大化正态性）
    """
    
    def __init__(
        self,
        num_projections: int = 1024,
        collapse_threshold: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Args:
            num_projections: 随机投影次数（多次投影可提高统计稳定性）
            collapse_threshold: embedding 标准差低于此值时判定为坍塌，仅用于监控
            seed: 随机数生成器种子（用于复现性）
        """
        super().__init__()
        self.num_projections = num_projections
        self.collapse_threshold = collapse_threshold
        self.seed = seed
        
        # 使用属性而不是 buffer 来存储投影矩阵
        self._projection_matrix_cache = None
        self._d_model_cache = None
    
    def _init_projections(self, d_model: int, device: torch.device):
        """延迟初始化投影矩阵（需要知道 d_model）"""
        if self._projection_matrix_cache is not None:
            return
        
        # 生成 [num_projections, d_model] 的高斯随机矩阵
        # 每行是一个随机投影方向
        proj_matrix = torch.randn(self.num_projections, d_model, device=device)
        
        # 标准化每个投影方向为单位向量（可选但有助于数值稳定性）
        proj_matrix = F.normalize(proj_matrix, p=2, dim=1)
        
        # 缓存到属性中
        self._projection_matrix_cache = proj_matrix
        self._d_model_cache = d_model
    
    @staticmethod
    def epps_pulley_statistic(x: torch.Tensor) -> torch.Tensor:
        """
        计算 Epps-Pulley 正态性统计量
        
        E_P = (1/n) * sum_i[(Q_i - i/(n+1))^2] / (1/12 * n)
        其中 Q_i 是第 i 个样本的理论标准正态分布分位数
        简化实现：使用样本 CDF 与理论 CDF 的 L2 距离
        
        Args:
            x: [N,] 一维样本序列
        
        Returns:
            统计量值（0~1，1表示完全正态）
        """
        n = x.size(0)
        if n < 2:
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
        
        # 标准化为均值0、方差1
        x_norm = (x - x.mean()) / (x.std() + 1e-8)
        
        # 计算样本的经验分位数
        # sorted_indices: 排序后的索引
        sorted_x, sorted_indices = torch.sort(x_norm)
        
        # 理论分位数（假设标准正态）
        # F(x) = Phi(x) 的反函数，即 x_i 处的理论分位数
        # 简化：使用 i/(n+1) 作为经验分位数，计算与理论值的差
        theoretical_quantiles = torch.erfinv(2 * torch.arange(1, n + 1, dtype=x.dtype, device=x.device) / (n + 1) - 1) * (2 ** 0.5)
        
        # L2 距离作为"非正态性"度量
        distance = torch.sqrt(((sorted_x - theoretical_quantiles) ** 2).mean() + 1e-8)
        
        # 转换为"正态性"分数（距离越小，正态性越高）
        # 使用指数衰减：exp(-distance)
        normality_score = torch.exp(-distance)
        
        return normality_score
    
    def forward(
        self,
        embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_monitoring: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        计算 SIGReg-like 损失
        
        Args:
            embedding: [B, Seq_Len, d_model] embedding 张量
            mask: [B, Seq_Len] 布尔掩码（True 表示有效位置），可选
            return_monitoring: 是否返回监控指标（collapse_flag, embed_std）
        
        Returns:
            当 return_monitoring=False: loss (标量张量)
            当 return_monitoring=True: (loss, monitoring_dict)
        """
        b, seq_len, d_model = embedding.shape
        device = embedding.device
        
        # 初始化投影矩阵
        if self._projection_matrix_cache is None:
            self._init_projections(d_model, device)
        
        # 应用掩码过滤有效位置
        if mask is not None:
            # mask: [B, Seq_Len]，True 表示有效，False 表示 padding
            valid_mask = mask.view(b, seq_len, 1).expand_as(embedding)
            embedding_valid = embedding[valid_mask].view(-1, d_model)
        else:
            embedding_valid = embedding.view(b * seq_len, d_model)
        
        if embedding_valid.size(0) < 2:
            # 如果没有足够的有效样本，返回零损失
            loss = torch.tensor(0.0, device=device, dtype=embedding.dtype)
            if return_monitoring:
                return loss, {"collapse_flag": False, "embed_std": 0.0}
            return loss
        
        # 计算 embedding 的标准差（用于监控坍塌）
        embed_std = embedding_valid.std(dim=0).mean().item()
        collapse_flag = embed_std < self.collapse_threshold
        
        # 进行 M 次随机投影
        # [num_projections, d_model] @ [d_model, N] -> [num_projections, N]
        # 确保投影矩阵不为 None
        assert self._projection_matrix_cache is not None, "投影矩阵应该已初始化"
        projections = torch.mm(self._projection_matrix_cache, embedding_valid.t())  # [M, N]
        
        # 计算每个投影的正态性统计量
        normality_scores = []
        for i in range(self.num_projections):
            proj_1d = projections[i]  # [N]
            score = self.epps_pulley_statistic(proj_1d)
            normality_scores.append(score)
        
        # 平均正态性分数
        mean_normality = torch.stack(normality_scores).mean()
        
        # 损失 = 1 - 平均正态性（最小化损失 = 最大化正态性）
        loss = 1.0 - mean_normality
        
        if return_monitoring:
            monitoring_dict: Dict[str, Any] = {
                "collapse_flag": collapse_flag,
                "embed_std": embed_std,
                "mean_normality": mean_normality.item(),
            }
            return loss, monitoring_dict
        
        return loss


# 使用示例
if __name__ == "__main__":
    # 测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = SIGRegLikeLoss(num_projections=128, seed=42)
    
    # 随机 embedding
    embedding = torch.randn(4, 50, 256, device=device)
    
    # 计算损失
    loss, monitoring = loss_fn(embedding, return_monitoring=True)
    print(f"Loss: {loss.item():.6f}")
    print(f"Monitoring: {monitoring}")
