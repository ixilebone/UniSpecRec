import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import TrainableModelBase
from dataclasses import dataclass
from typing import Literal


class AlphaRec(TrainableModelBase):
    @dataclass
    class ModelConfig(TrainableModelBase.ModelConfig):
        latent_dim: int = 64
        num_layers: int = 3
        similarity: Literal['cos', 'dot'] = 'cos'

    def __init__(
        self,
        rate_matrix: torch.Tensor,
        user_semantic_embeddings: torch.Tensor,
        item_semantic_embeddings: torch.Tensor,
        model_config: ModelConfig,
        loss_config: TrainableModelBase.LossConfig,
    ):
        super(AlphaRec, self).__init__(model_config=model_config, loss_config=loss_config)
        self.rate_matrix = rate_matrix
        self.user_semantic_embeddings = user_semantic_embeddings
        self.item_semantic_embeddings = item_semantic_embeddings
        self.latent_dim = self.model_config.latent_dim
        self.num_layers = self.model_config.num_layers
        self.device = self.model_config.device

        self.user_size = rate_matrix.shape[0]
        self.item_size = rate_matrix.shape[1]
        self.norm_adj_matrix = self._create_norm_adj_matrix(self.rate_matrix).to(self.device)

        self.adapter = self._build_adapter()
        self._init_weights()

    def _build_adapter(self):
        embed_size = self.user_semantic_embeddings.shape[1]
        return nn.Sequential(
            nn.Linear(embed_size, embed_size//2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(embed_size//2, self.latent_dim, bias=True)
        ).to(self.device)


    def _init_weights(self):
        for module in self.adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _propagate(
        self,
        adj_matrix: torch.Tensor,
        embedding: torch.Tensor
    ) -> torch.Tensor:
        # torch.sparse.mm 不支持 float16，禁用 autocast 用 float32 计算
        with torch.amp.autocast('cuda', enabled=False):
            return torch.sparse.mm(adj_matrix.float(), embedding.float())

    def _create_norm_adj_matrix(self, rate_matrix: torch.Tensor) -> torch.Tensor:
        num_users = self.user_size
        num_items = self.item_size

        user_degree = rate_matrix.sum(dim=1)
        item_degree = rate_matrix.sum(dim=0)

        user_degree_inv_sqrt = torch.where(
            user_degree > 0,
            1.0 / torch.sqrt(user_degree),
            torch.zeros_like(user_degree)
        )
        item_degree_inv_sqrt = torch.where(
            item_degree > 0,
            1.0 / torch.sqrt(item_degree),
            torch.zeros_like(item_degree)
        )

        norm_R = user_degree_inv_sqrt.unsqueeze(1) * rate_matrix * item_degree_inv_sqrt.unsqueeze(0)

        indices_R = norm_R.nonzero(as_tuple=True)
        values_R = norm_R[indices_R]

        indices_ui_row = indices_R[0]
        indices_ui_col = indices_R[1] + num_users

        indices_iu_row = indices_R[1] + num_users
        indices_iu_col = indices_R[0]

        all_indices = torch.cat([
            torch.stack([indices_ui_row, indices_ui_col], dim=0),
            torch.stack([indices_iu_row, indices_iu_col], dim=0)
        ], dim=1)
        all_values = torch.cat([values_R, values_R])

        total_size = num_users + num_items
        norm_adj = torch.sparse_coo_tensor(
            all_indices,
            all_values,
            (total_size, total_size),
            device=rate_matrix.device,
            dtype=rate_matrix.dtype
        )

        return norm_adj.coalesce()

    def forward(
        self,
        user_index: torch.Tensor, # shape: (batch_size,)
        item_index: torch.Tensor, # shape: (batch_size,)
    ) -> torch.Tensor:

        user_embedding = self.adapter(self.user_semantic_embeddings.to(self.device))
        item_embedding = self.adapter(self.item_semantic_embeddings.to(self.device))
        embedding = torch.cat([user_embedding, item_embedding], dim=0)
        embedding_list = [embedding]

        for _ in range(self.num_layers):
            embedding = self._propagate(self.norm_adj_matrix, embedding)
            embedding_list.append(embedding)

        final_embedding = torch.stack(embedding_list, dim=0).mean(dim=0)
        final_user_embedding = final_embedding[:self.user_size]
        final_item_embedding = final_embedding[self.user_size:]

        user_embedding = final_user_embedding[user_index]
        item_embedding = final_item_embedding[item_index]

        return user_embedding, item_embedding

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        user_embedding, item_embedding = self.forward(
            user_index=torch.arange(self.user_size, device=self.device),
            item_index=torch.arange(self.item_size, device=self.device),
        )
        # 计算余弦相似度
        match self.model_config.similarity:
            case "cos":
                user_embedding_norm = F.normalize(user_embedding, p=2, dim=1)
                item_embedding_norm = F.normalize(item_embedding, p=2, dim=1)
                pred_matrix = torch.matmul(user_embedding_norm, item_embedding_norm.t())
            case "dot":
                pred_matrix = torch.matmul(user_embedding, item_embedding.t())
        return pred_matrix
