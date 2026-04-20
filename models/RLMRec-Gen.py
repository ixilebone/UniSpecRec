import torch
import torch.nn as nn
from dataclasses import dataclass
from models.base import TrainableModelBase
from utils import cal_infonce_loss


class RLMRecGen(TrainableModelBase):
	@dataclass
	class ModelConfig(TrainableModelBase.ModelConfig):
		mask_ratio: float = 0.2
		recon_weight: float = 1.0
		recon_temperature: float = 0.1
		hidden_dim: int = 0

	def __init__(
		self,
		base_model: TrainableModelBase,
		user_semantic_embeddings: torch.Tensor,
		item_semantic_embeddings: torch.Tensor,
		model_config: ModelConfig | None = None,
	):
		if model_config is None:
			model_config = self.ModelConfig(device=base_model.model_config.device)

		super(RLMRecGen, self).__init__(
			model_config=model_config,
			loss_config=base_model.loss_config,
		)

		self.base_model = base_model
		self.rate_matrix = base_model.rate_matrix
		self.device = self.model_config.device
		self.num_users, self.num_items = self.rate_matrix.shape

		self.register_buffer("user_semantic_embeddings", user_semantic_embeddings.float().to(self.device))
		self.register_buffer("item_semantic_embeddings", item_semantic_embeddings.float().to(self.device))

		self.embedding_dim = getattr(base_model, "embedding_dim", None)
		if self.embedding_dim is None:
			raise ValueError("base_model must expose attribute 'embedding_dim'.")

		sem_dim = self.user_semantic_embeddings.shape[1]
		hidden_dim = self.model_config.hidden_dim if self.model_config.hidden_dim > 0 else (self.embedding_dim + sem_dim) // 2
		self.decoder = nn.Sequential(
			nn.Linear(self.embedding_dim, hidden_dim, bias=False),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim, sem_dim, bias=False),
		).to(self.device)

		self._init_weights()

	def _init_weights(self):
		for module in self.decoder.modules():
			if isinstance(module, nn.Linear):
				nn.init.xavier_uniform_(module.weight)
				if module.bias is not None:
					nn.init.zeros_(module.bias)

	def forward(
		self,
		user_index: torch.Tensor,
		item_index: torch.Tensor,
	) -> tuple[torch.Tensor, torch.Tensor]:
		return self.base_model.forward(user_index, item_index)

	def _sample_nodes(self) -> tuple[torch.Tensor, torch.Tensor]:
		num_nodes = self.num_users + self.num_items
		num_seeds = max(1, int(num_nodes * self.model_config.mask_ratio))
		seeds = torch.randperm(num_nodes, device=self.device)[:num_seeds]

		user_seeds = seeds[seeds < self.num_users]
		item_seeds = seeds[seeds >= self.num_users] - self.num_users
		return user_seeds, item_seeds

	def _reconstruction_loss(self) -> torch.Tensor:
		user_seeds, item_seeds = self._sample_nodes()
		all_cf_embeddings = []
		all_sem_embeddings = []

		if user_seeds.numel() > 0:
			dummy_item_index = torch.zeros(user_seeds.shape[0], dtype=torch.long, device=self.device)
			user_cf_embedding, _ = self.base_model.forward(user_seeds, dummy_item_index)
			all_cf_embeddings.append(user_cf_embedding)
			all_sem_embeddings.append(self.user_semantic_embeddings[user_seeds])

		if item_seeds.numel() > 0:
			dummy_user_index = torch.zeros(item_seeds.shape[0], dtype=torch.long, device=self.device)
			_, item_cf_embedding = self.base_model.forward(dummy_user_index, item_seeds)
			all_cf_embeddings.append(item_cf_embedding)
			all_sem_embeddings.append(self.item_semantic_embeddings[item_seeds])

		cf_embedding = torch.cat(all_cf_embeddings, dim=0)
		sem_embedding = torch.cat(all_sem_embeddings, dim=0)
		reconstructed_semantic = self.decoder(cf_embedding)

		return cal_infonce_loss(
			user_embedding=reconstructed_semantic,
			pos_item_embedding=sem_embedding,
			similarity="cos",
			temperature=self.model_config.recon_temperature,
			is_inbatch=True,
		)

	def get_loss(
		self,
		user_index: torch.Tensor,
		pos_item_index: torch.Tensor,
		neg_item_indices: torch.Tensor,
	) -> torch.Tensor:
		base_loss = self.base_model.get_loss(
			user_index=user_index,
			pos_item_index=pos_item_index,
			neg_item_indices=neg_item_indices,
		)
		recon_loss = self._reconstruction_loss()
		return base_loss + self.model_config.recon_weight * recon_loss

	@torch.no_grad()
	def predict(self) -> torch.Tensor:
		return self.base_model.predict()
