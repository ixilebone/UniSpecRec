import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal
from models.base import TrainableModelBase


class AlignMF(TrainableModelBase):
    @dataclass
    class ModelConfig(TrainableModelBase.ModelConfig):
        embedding_dim: int = 64
        similarity: Literal['cos', 'dot'] = 'dot'

    def __init__(
        self,
        rate_matrix: torch.Tensor,
        user_semantic_embeddings: torch.Tensor,
        item_semantic_embeddings: torch.Tensor,
        model_config: ModelConfig,
        loss_config: TrainableModelBase.LossConfig,
    ):
        super(AlignMF, self).__init__(model_config=model_config, loss_config=loss_config)

        self.rate_matrix = rate_matrix
        self.device = self.model_config.device
        self.embedding_dim = self.model_config.embedding_dim
        self.num_users = rate_matrix.shape[0]
        self.num_items = rate_matrix.shape[1]

        self.register_buffer('user_semantic_embeddings', user_semantic_embeddings.float().to(self.device))
        self.register_buffer('item_semantic_embeddings', item_semantic_embeddings.float().to(self.device))

        self.adapter = self._build_adapter()
        self._init_weights()

    def _build_adapter(self) -> nn.Module:
        semantic_dim = self.user_semantic_embeddings.shape[1]
        hidden_dim = max(1, semantic_dim // 2)
        return nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.embedding_dim, bias=True),
        ).to(self.device)

    def _init_weights(self):
        for module in self.adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_all_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_embedding = self.adapter(self.user_semantic_embeddings)
        item_embedding = self.adapter(self.item_semantic_embeddings)
        return user_embedding, item_embedding

    def forward(
        self,
        user_index: torch.Tensor,
        item_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_user_embedding, all_item_embedding = self.get_all_embeddings()
        user_embedding = all_user_embedding[user_index]
        item_embedding = all_item_embedding[item_index]
        return user_embedding, item_embedding

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        user_embedding, item_embedding = self.get_all_embeddings()
        match self.model_config.similarity:
            case 'cos':
                user_embedding = F.normalize(user_embedding, p=2, dim=1)
                item_embedding = F.normalize(item_embedding, p=2, dim=1)
                pred_matrix = torch.mm(user_embedding, item_embedding.t())
            case 'dot':
                pred_matrix = torch.mm(user_embedding, item_embedding.t())
            case _:
                raise ValueError(f"Unsupported similarity type: {self.model_config.similarity}")
        return pred_matrix
