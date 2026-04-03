import torch
from models.base import ModelBase
from dataclasses import dataclass


class MF(ModelBase):
    """矩阵分解模型 - 非训练版本（基于SVD）
    
    将用户-物品交互矩阵进行SVD分解，得到用户嵌入和物品嵌入
    """
    
    @dataclass
    class ModelConfig(ModelBase.ModelConfig):
        embedding_dim: int = 50  # SVD保留的维度
    
    @classmethod
    def get_best_model_config(cls, interaction_data: str, device: str) -> 'ModelConfig':
        """根据数据集返回最优配置"""
        match interaction_data:
            case "games":
                return cls.ModelConfig(embedding_dim=100, device=device)
            case "books":
                return cls.ModelConfig(embedding_dim=50, device=device)
            case "toys":
                return cls.ModelConfig(embedding_dim=50, device=device)
            case _:
                return cls.ModelConfig(embedding_dim=50, device=device)
    
    def __init__(
        self,
        rate_matrix: torch.Tensor,
        model_config: ModelConfig,
    ):
        super(MF, self).__init__(model_config=model_config)
        self.device = self.model_config.device
        self.rate_matrix = rate_matrix.to(self.device)
        self.embedding_dim = self.model_config.embedding_dim
        
        # 进行SVD分解
        self._decompose_matrix()
    
    def _decompose_matrix(self):
        """使用SVD分解rate_matrix"""
        # 转移到CPU进行SVD计算（某些情况下更稳定）
        rate_matrix_cpu = self.rate_matrix.cpu()
        
        # 进行SVD分解
        U, S, Vh = torch.linalg.svd(rate_matrix_cpu, full_matrices=False)
        
        # 只保留前embedding_dim个奇异值及对应的向量
        k = min(self.embedding_dim, S.shape[0])
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
        
        # 用户嵌入 = U @ sqrt(Sigma)
        self.user_embedding = U @ torch.diag(torch.sqrt(S))
        # 物品嵌入 = sqrt(Sigma) @ Vh
        self.item_embedding = torch.diag(torch.sqrt(S)) @ Vh
        
        # 转移到目标设备
        self.user_embedding = self.user_embedding.to(self.device)
        self.item_embedding = self.item_embedding.to(self.device)
    
    def predict(self) -> torch.Tensor:
        """生成预测评分矩阵"""
        pred_matrix = self.user_embedding @ self.item_embedding
        return pred_matrix


# 测试代码
from utils import DataLoader, Metric
import matplotlib.pyplot as plt
import tqdm


def test_mf_by_embedding_dim(device: str = 'cpu'):
    """测试不同嵌入维度对性能的影响"""
    dataloader = DataLoader(
        interaction_data="games",
        semantic_data=None,
        device=device,
    )
    metric = Metric(k_list=[10, 20], device=device)
    fig, ax = plt.subplots()
    embedding_dims = range(10, 201, 10)
    results = None
    
    for embedding_dim in tqdm.tqdm(embedding_dims):
        mf_model = MF(
            rate_matrix=dataloader.rate_matrix,
            model_config=MF.ModelConfig(embedding_dim=embedding_dim, device=device)
        )
        
        test_rate_matrix = dataloader.test_rate_matrix
        score = mf_model.test(metric, test_rate_matrix)
        
        if results is None:
            results = {k: [] for k in score.keys()}
        for k, v in score.items():
            results[k].append(v)
    
    for k in results.keys():
        ax.plot(embedding_dims, results[k], label=k)
        best_idx = results[k].index(max(results[k]))
        print(f"Max {k}: {max(results[k])} at embedding_dim={list(embedding_dims)[best_idx]}")
    
    ax.legend()
    ax.set_xlabel('Embedding Dimension')
    ax.set_ylabel('Score')
    ax.set_title('MF Performance vs Embedding Dimension')
    plt.show()


if __name__ == "__main__":
    '''终端输入: python -m models.MF'''
    test_mf_by_embedding_dim(device='cuda')
