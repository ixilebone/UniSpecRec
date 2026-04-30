import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import TrainableModelBase
from utils import cal_bpr_loss, cal_infonce_loss
from dataclasses import dataclass
from typing import Literal


class SimGCL(TrainableModelBase):
    @dataclass
    class ModelConfig(TrainableModelBase.ModelConfig):
        latent_dim: int = 32
        num_layers: int = 2
        similarity: Literal['cos', 'dot'] = 'dot'
        reg_weight: float = 0.0
        eps: float = 0.1
        cl_weight: float = 0.02
        cl_tau: float = 0.2

    def __init__(
        self,
        rate_matrix: torch.Tensor,
        model_config: ModelConfig,
        loss_config: TrainableModelBase.LossConfig
    ):
        super(SimGCL, self).__init__(model_config=model_config, loss_config=loss_config)

        self.rate_matrix = rate_matrix
        self.embedding_dim = self.model_config.latent_dim
        self.num_layers = self.model_config.num_layers
        self.num_users = rate_matrix.shape[0]
        self.num_items = rate_matrix.shape[1]
        self.device = self.model_config.device

        self.user_embedding = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(self.num_users, self.embedding_dim)
            ).to(self.device),
            requires_grad=True
        )
        self.item_embedding = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(self.num_items, self.embedding_dim)
            ).to(self.device),
            requires_grad=True
        )

        self.norm_adj_matrix = self._create_norm_adj_matrix(self.rate_matrix).to(self.device)

    def _create_norm_adj_matrix(self, rate_matrix: torch.Tensor) -> torch.sparse_coo_tensor:
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
        indices_ui_col = indices_R[1] + self.num_users

        indices_iu_row = indices_R[1] + self.num_users
        indices_iu_col = indices_R[0]

        all_indices = torch.cat([
            torch.stack([indices_ui_row, indices_ui_col], dim=0),
            torch.stack([indices_iu_row, indices_iu_col], dim=0)
        ], dim=1)
        all_values = torch.cat([values_R, values_R])

        total_size = self.num_users + self.num_items
        norm_adj = torch.sparse_coo_tensor(
            all_indices,
            all_values,
            (total_size, total_size),
            device=rate_matrix.device,
            dtype=rate_matrix.dtype
        )
        return norm_adj.coalesce()

    def _propagate(
        self,
        adj_matrix: torch.sparse_coo_tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            return torch.sparse.mm(adj_matrix.float(), embedding.float())

    def _perturb_embedding(self, embeddings: torch.Tensor) -> torch.Tensor:
        noise = F.normalize(torch.rand_like(embeddings), p=2, dim=1)
        noise = noise * torch.sign(embeddings) * self.model_config.eps
        return embeddings + noise

    def _get_propagated_embeddings(self, perturbed: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.cat([self.user_embedding, self.item_embedding], dim=0)

        scale = 1.0 / (self.num_layers + 1)
        embeddings_sum = embeddings * scale
        for _ in range(self.num_layers):
            embeddings = self._propagate(self.norm_adj_matrix, embeddings)
            if perturbed:
                embeddings = self._perturb_embedding(embeddings)
            embeddings_sum.add_(embeddings, alpha=scale)

        final_user_embedding = embeddings_sum[:self.num_users]
        final_item_embedding = embeddings_sum[self.num_users:]
        return final_user_embedding, final_item_embedding

    def forward(
        self,
        user_index: torch.Tensor,
        item_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_user_embedding, final_item_embedding = self._get_propagated_embeddings(perturbed=False)
        user_embedding = final_user_embedding[user_index]
        item_embedding = final_item_embedding[item_index]
        return user_embedding, item_embedding

    def get_loss(
        self,
        user_index: torch.Tensor,
        pos_item_index: torch.Tensor,
        neg_item_indices: torch.Tensor,
    ) -> torch.Tensor:
        clean_user_embedding, clean_item_embedding = self._get_propagated_embeddings(perturbed=False)
        perturbed_user_embedding_1, perturbed_item_embedding_1 = self._get_propagated_embeddings(perturbed=True)
        perturbed_user_embedding_2, perturbed_item_embedding_2 = self._get_propagated_embeddings(perturbed=True)

        batch_user_embedding = clean_user_embedding[user_index]
        batch_pos_item_embedding = clean_item_embedding[pos_item_index]
        batch_neg_item_embeddings = clean_item_embedding[neg_item_indices]

        bpr_loss = cal_bpr_loss(
            user_embedding=batch_user_embedding,
            pos_item_embedding=batch_pos_item_embedding,
            neg_item_embeddings=batch_neg_item_embeddings,
            similarity=self.loss_config.similarity,
        )

        user_cl_loss = cal_infonce_loss(
            user_embedding=perturbed_user_embedding_1[user_index],
            pos_item_embedding=perturbed_user_embedding_2[user_index],
            similarity='cos',
            temperature=self.model_config.cl_tau,
            is_inbatch=True,
        )
        item_cl_loss = cal_infonce_loss(
            user_embedding=perturbed_item_embedding_1[pos_item_index],
            pos_item_embedding=perturbed_item_embedding_2[pos_item_index],
            similarity='cos',
            temperature=self.model_config.cl_tau,
            is_inbatch=True,
        )
        cl_loss = 0.5 * (user_cl_loss + item_cl_loss)

        reg_loss = self.model_config.reg_weight * 0.5 * (
            self.user_embedding.pow(2).sum() + self.item_embedding.pow(2).sum()
        )

        return bpr_loss + self.model_config.cl_weight * cl_loss + reg_loss

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        final_user_embedding, final_item_embedding = self._get_propagated_embeddings(perturbed=False)
        match self.model_config.similarity:
            case 'cos':
                user_embedding = F.normalize(final_user_embedding, p=2, dim=1)
                item_embedding = F.normalize(final_item_embedding, p=2, dim=1)
                pred_matrix = torch.mm(user_embedding, item_embedding.t())
            case 'dot':
                pred_matrix = torch.mm(final_user_embedding, final_item_embedding.t())
            case _:
                raise ValueError(f"Unsupported similarity type: {self.model_config.similarity}")
        return pred_matrix
