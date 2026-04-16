import torch
from models.base import ModelBase
from dataclasses import dataclass
from utils.grid_search import MatrixNormalization


class UniSpecRec(ModelBase):
    """将CF预测矩阵与语义嵌入融合的推荐模型"""

    @dataclass
    class ModelConfig(ModelBase.ModelConfig):
        power: float = 0.0
        gamma: float = 0.0
        normalization_method: str = 'zscore'

    def __init__(
        self,
        cf_pred_matrix: torch.Tensor,
        user_semantic_embeddings: torch.Tensor,
        item_semantic_embeddings: torch.Tensor,
        rate_matrix: torch.Tensor,
        model_config: ModelConfig,
    ):
        super().__init__(model_config=model_config)
        self.rate_matrix = rate_matrix

        # CF预测矩阵（不需要梯度）
        self.register_buffer('cf_pred_matrix', cf_pred_matrix)

        # 预计算SVD分解
        combined = torch.cat([user_semantic_embeddings, item_semantic_embeddings], dim=0)
        U, S, V_T = torch.linalg.svd(combined)
        self.register_buffer('svd_U', U)
        self.register_buffer('svd_S', S)

        self.num_users = rate_matrix.shape[0]

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        normalize_fn = MatrixNormalization.get_method(self.model_config.normalization_method)

        # 语义预测矩阵
        se_pred_matrix = (
            self.svd_U[:self.num_users, :len(self.svd_S)] * torch.pow(self.svd_S, self.model_config.power)
            @ (self.svd_U[self.num_users:, :len(self.svd_S)] * torch.pow(self.svd_S, self.model_config.power)).T
        )

        # 归一化并融合
        norm_cf = normalize_fn(self.cf_pred_matrix)
        norm_se = normalize_fn(se_pred_matrix)

        gamma = self.model_config.gamma
        return (1 - gamma) * norm_cf + gamma * norm_se
