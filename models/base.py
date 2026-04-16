import torch
from torch import nn
from utils import Metric, cal_bpr_loss, cal_infonce_loss
from typing import Literal
from dataclasses import dataclass

class ModelBase(nn.Module):
    @dataclass
    class ModelConfig():
        device: str = 'cpu'
    def __init__(self, model_config: ModelConfig):
        super(ModelBase, self).__init__()
        self.model_config = model_config

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        """生成预测评分矩阵，形状是(num_users, num_items)"""
        raise NotImplementedError("Subclasses should implement this!")

    def test(self, metric: Metric, test_rate_matrix: torch.Tensor):
        return metric.eval(train_matrix=self.rate_matrix, test_matrix=test_rate_matrix, pred_matrix=self.predict())
    

class TrainableModelBase(ModelBase):
    @dataclass
    class BPRLossConfig():
        type: Literal["bpr"] = "bpr"
        similarity: Literal["cos", "dot"] = "dot"
        num_neg_item: int = 1
    @dataclass
    class InfoNCELossConfig():
        type: Literal["infonce"] = "infonce"
        similarity: Literal["cos", "dot"] = "dot"
        temperature: float = 0.5
        is_inbatch: bool = False
        num_neg_item: int = 1

    LossConfig = BPRLossConfig | InfoNCELossConfig
    def __init__(
        self,
        model_config: ModelBase.ModelConfig,
        loss_config: LossConfig,
    ):
        super(TrainableModelBase, self).__init__(model_config=model_config)
        self.loss_config = loss_config
        
    def get_loss(
        self,
        user_index: torch.Tensor, # shape: (batch_size,)
        pos_item_index: torch.Tensor, # shape: (batch_size,)
        neg_item_indices: torch.Tensor, # shape: (batch_size, num_neg_item)
    ) -> torch.Tensor:
        batch_size = user_index.shape[0]

        combined_user_index = torch.cat([user_index, user_index], dim=0)
        combined_item_index = torch.cat([pos_item_index, neg_item_indices.flatten()], dim=0)
        
        combined_user_embedding, combined_item_embedding = self.forward(
            user_index=combined_user_index,
            item_index=combined_item_index
        )
        
        user_embedding = combined_user_embedding[:batch_size]
        pos_item_embedding = combined_item_embedding[:batch_size]
        neg_item_embeddings = combined_item_embedding[batch_size:].view(batch_size, -1, combined_item_embedding.shape[1])

        match self.loss_config.type:
            case "bpr":
                loss = cal_bpr_loss(
                    user_embedding=user_embedding,
                    pos_item_embedding=pos_item_embedding,
                    neg_item_embeddings=neg_item_embeddings,
                    similarity=self.loss_config.similarity,
                )
            case "infonce":
                loss = cal_infonce_loss(
                    user_embedding=user_embedding,
                    pos_item_embedding=pos_item_embedding,
                    neg_item_embeddings=neg_item_embeddings,
                    similarity=self.loss_config.similarity,
                    temperature=self.loss_config.temperature,
                    is_inbatch=self.loss_config.is_inbatch,
                    pos_item_index=pos_item_index if self.loss_config.is_inbatch else None,
                )
            case _:
                raise ValueError(f"Unsupported loss type: {self.loss_config.type}")

        return loss
    
    def forward(self, user_index: torch.Tensor, item_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据用户索引和物品索引，生成对应的用户嵌入和物品嵌入，形状都是(batch_size, embedding_dim)"""
        raise NotImplementedError("Subclasses should implement this!")
        user_embedding = None
        item_embedding = None
        return user_embedding, item_embedding


