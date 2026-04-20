import torch
import torch.nn as nn
from dataclasses import dataclass
from models.base import TrainableModelBase
from utils import cal_infonce_loss


class RLMRecCon(TrainableModelBase):
	@dataclass
	class ModelConfig(TrainableModelBase.ModelConfig):
		kd_weight: float = 1.0
		kd_temperature: float = 0.1
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

		super(RLMRecCon, self).__init__(
			model_config=model_config,
			loss_config=base_model.loss_config,
		)

		self.base_model = base_model
		self.rate_matrix = base_model.rate_matrix
		self.device = self.model_config.device

		self.register_buffer("user_semantic_embeddings", user_semantic_embeddings.float().to(self.device))
		self.register_buffer("item_semantic_embeddings", item_semantic_embeddings.float().to(self.device))

		self.embedding_dim = getattr(base_model, "embedding_dim", None)
		if self.embedding_dim is None:
			raise ValueError("base_model must expose attribute 'embedding_dim'.")

		sem_dim = self.user_semantic_embeddings.shape[1]
		hidden_dim = self.model_config.hidden_dim if self.model_config.hidden_dim > 0 else (sem_dim + self.embedding_dim) // 2
		self.adapter = nn.Sequential(
			nn.Linear(sem_dim, hidden_dim, bias=False),
			nn.LeakyReLU(),
			nn.Linear(hidden_dim, self.embedding_dim, bias=False),
		).to(self.device)

		self._init_weights()

	def _init_weights(self):
		for module in self.adapter.modules():
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

		user_cf_embedding, item_cf_embedding = self.base_model.forward(user_index, pos_item_index)
		user_sem_embedding = self.adapter(self.user_semantic_embeddings[user_index])
		item_sem_embedding = self.adapter(self.item_semantic_embeddings[pos_item_index])

		user_align_loss = cal_infonce_loss(
			user_embedding=user_cf_embedding,
			pos_item_embedding=user_sem_embedding,
			similarity="cos",
			temperature=self.model_config.kd_temperature,
			is_inbatch=True,
		)
		item_align_loss = cal_infonce_loss(
			user_embedding=item_cf_embedding,
			pos_item_embedding=item_sem_embedding,
			similarity="cos",
			temperature=self.model_config.kd_temperature,
			is_inbatch=True,
		)
		kd_loss = 0.5 * (user_align_loss + item_align_loss)

		return base_loss + self.model_config.kd_weight * kd_loss

	@torch.no_grad()
	def predict(self) -> torch.Tensor:
		return self.base_model.predict()
