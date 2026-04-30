import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal
from models.base import TrainableModelBase


class MF(TrainableModelBase):
    @dataclass
    class ModelConfig(TrainableModelBase.ModelConfig):
        embedding_dim: int = 64
        similarity: Literal['cos', 'dot'] = 'dot'

    def __init__(
        self,
        rate_matrix: torch.Tensor,
        model_config: ModelConfig,
        loss_config: TrainableModelBase.LossConfig,
    ):
        super(MF, self).__init__(model_config=model_config, loss_config=loss_config)

        self.rate_matrix = rate_matrix
        self.device = self.model_config.device
        self.embedding_dim = self.model_config.embedding_dim
        self.num_users = rate_matrix.shape[0]
        self.num_items = rate_matrix.shape[1]

        self.user_embedding = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(self.num_users, self.embedding_dim)
            ).to(self.device),
            requires_grad=True,
        )
        self.item_embedding = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(self.num_items, self.embedding_dim)
            ).to(self.device),
            requires_grad=True,
        )

    def forward(
        self,
        user_index: torch.Tensor,
        item_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_embedding = self.user_embedding[user_index]
        item_embedding = self.item_embedding[item_index]
        return user_embedding, item_embedding

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        match self.model_config.similarity:
            case 'cos':
                user_embedding = F.normalize(self.user_embedding, p=2, dim=1)
                item_embedding = F.normalize(self.item_embedding, p=2, dim=1)
                pred_matrix = torch.mm(user_embedding, item_embedding.t())
            case 'dot':
                pred_matrix = torch.mm(self.user_embedding, self.item_embedding.t())
            case _:
                raise ValueError(f"Unsupported similarity type: {self.model_config.similarity}")
        return pred_matrix
