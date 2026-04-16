import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base import TrainableModelBase
from dataclasses import dataclass
from typing import Literal


class LightGCN(TrainableModelBase):
	@dataclass
	class ModelConfig(TrainableModelBase.ModelConfig):
		latent_dim: int = 32
		num_layers: int = 2
		similarity: Literal['cos', 'dot'] = 'dot'
	def __init__(
		self,
		rate_matrix: torch.Tensor,
		model_config: ModelConfig,
		loss_config: TrainableModelBase.LossConfig
	):

		super(LightGCN, self).__init__(model_config=model_config, loss_config=loss_config)

		self.rate_matrix = rate_matrix
		self.embedding_dim = self.model_config.latent_dim
		self.num_layers = self.model_config.num_layers
		self.num_users = rate_matrix.shape[0]
		self.num_items = rate_matrix.shape[1]
		self.device = self.model_config.device

		# 初始化用户和物品嵌入
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

		# 预计算归一化邻接矩阵
		self.norm_adj_matrix = self._create_norm_adj_matrix(self.rate_matrix).to(self.device)

	def _create_norm_adj_matrix(self, rate_matrix: torch.Tensor) -> torch.sparse_coo_tensor:
		"""
		Create normalized adjacency matrix in sparse COO format.
		**Optimization**: Use sparse format to save memory (10-100x for large graphs).
		"""
		
		user_degree = rate_matrix.sum(dim=1)  # [num_users]
		item_degree = rate_matrix.sum(dim=0)  # [num_items]
		
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
		
		# 归一化交互矩阵: D_u^{-1/2} R D_i^{-1/2}
		norm_R = user_degree_inv_sqrt.unsqueeze(1) * rate_matrix * item_degree_inv_sqrt.unsqueeze(0)
		
		# Build sparse adjacency matrix
		# Structure:  [  0     R  ]
		#            [ R^T    0  ]
		# Get non-zero indices from norm_R
		indices_R = norm_R.nonzero(as_tuple=True)
		values_R = norm_R[indices_R]
		
		# User-item part indices (top-right)
		indices_ui_row = indices_R[0]
		indices_ui_col = indices_R[1] + self.num_users
		
		# Item-user part indices (bottom-left)
		indices_iu_row = indices_R[1] + self.num_users
		indices_iu_col = indices_R[0]
		
		# Concatenate both parts
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
		
		# Convert to CSR for more efficient operations
		return norm_adj.coalesce()

	def _propagate(
		self,
		adj_matrix: torch.sparse_coo_tensor,  # Now sparse format
		embedding: torch.Tensor,
	) -> torch.Tensor:
		# Use sparse matrix multiplication (float32 only - sparse.mm doesn't support half)
		with torch.amp.autocast('cuda', enabled=False):
			return torch.sparse.mm(adj_matrix.float(), embedding.float())
	
	def _get_propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
		"""
		Extract shared propagation logic to avoid code duplication.
		Performs graph propagation and returns final user/item embeddings.
		Optimization: accumulate embeddings in-place to save memory.
		"""
		embeddings = torch.cat([self.user_embedding, self.item_embedding], dim=0)
		
		# Accumulate embeddings across layers (more efficient than list + stack + mean)
		# Use mul_ to copy with scaling factor already applied, avoiding clone()
		scale = 1.0 / (self.num_layers + 1)
		embeddings_sum = embeddings * scale
		for _ in range(self.num_layers):
			embeddings = self._propagate(self.norm_adj_matrix, embeddings)
			embeddings_sum.add_(embeddings, alpha=scale)  # Fused multiply-add
		
		final_embeddings = embeddings_sum
		
		final_user_embedding = final_embeddings[:self.num_users]
		final_item_embedding = final_embeddings[self.num_users:]
		
		return final_user_embedding, final_item_embedding

	def forward(
		self,
		user_index: torch.Tensor,  # shape: (batch_size,)
		item_index: torch.Tensor,  # shape: (batch_size,)
	) -> tuple[torch.Tensor, torch.Tensor]:
		final_user_embedding, final_item_embedding = self._get_propagated_embeddings()
		
		user_embedding = final_user_embedding[user_index]
		item_embedding = final_item_embedding[item_index]
		
		return user_embedding, item_embedding
	
	@torch.no_grad()
	def predict(self) -> torch.Tensor:
		final_user_embedding, final_item_embedding = self._get_propagated_embeddings()
		match self.model_config.similarity:
			case "cos":
				user_embedding = F.normalize(final_user_embedding, p=2, dim=1)
				item_embedding = F.normalize(final_item_embedding, p=2, dim=1)
				pred_matrix = torch.mm(user_embedding, item_embedding.t())
			case "dot":
				pred_matrix = torch.mm(final_user_embedding, final_item_embedding.t())
			case _:
				raise ValueError(f"Unsupported similarity type: {self.model_config.similarity}")
		return pred_matrix