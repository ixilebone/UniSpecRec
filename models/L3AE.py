import torch
from dataclasses import dataclass
from models.base import ModelBase


class L3AE(ModelBase):
	@dataclass
	class ModelConfig(ModelBase.ModelConfig):
		reg_X: float = 1.0
		reg_F: float = 1.0
		reg_E: float = 1.0

	def __init__(
		self,
		rate_matrix: torch.Tensor,
		item_semantic_embeddings: torch.Tensor,
		model_config: ModelConfig,
	):
		super(L3AE, self).__init__(model_config=model_config)

		self.device = self.model_config.device
		self.rate_matrix = rate_matrix.float().to(self.device)

		num_items = self.rate_matrix.shape[1]
		semantic_embeddings = item_semantic_embeddings.float().to(self.device)

		# Ensure F has shape (semantic_dim, num_items)
		if semantic_embeddings.shape[0] == num_items:
			F = semantic_embeddings.t()
		elif semantic_embeddings.shape[1] == num_items:
			F = semantic_embeddings
		else:
			raise ValueError(
				"item_semantic_embeddings has incompatible shape. "
				f"Expected one dimension to equal num_items={num_items}, got {tuple(semantic_embeddings.shape)}"
			)

		self.item_similarity = self._build_item_similarity(F)

	def _safe_inverse(self, matrix: torch.Tensor) -> torch.Tensor:
		try:
			return torch.linalg.inv(matrix)
		except RuntimeError:
			return torch.linalg.pinv(matrix)

	def _build_item_similarity(self, F: torch.Tensor) -> torch.Tensor:
		reg_X = self.model_config.reg_X
		reg_F = self.model_config.reg_F
		reg_E = self.model_config.reg_E

		dtype = self.rate_matrix.dtype
		device = self.rate_matrix.device
		num_items = self.rate_matrix.shape[1]
		identity = torch.eye(num_items, dtype=dtype, device=device)

		# Step 1: build semantic item-item matrix C
		GF = F.t().mm(F)
		P1 = self._safe_inverse(GF + reg_F * identity)

		diag_P1 = torch.diag(P1)
		denom_p1 = -diag_P1
		denom_p1 = torch.where(
			denom_p1.abs() < 1e-12,
			torch.full_like(denom_p1, 1e-12),
			denom_p1,
		)
		C = P1 / denom_p1.unsqueeze(1)
		C.fill_diagonal_(0)

		# Step 2: ensemble with interaction co-occurrence
		GX = self.rate_matrix.t().mm(self.rate_matrix)
		P2 = self._safe_inverse(GX + (reg_X + reg_E) * identity)
		P2C = P2.mm(C)

		diag_P2 = torch.diag(P2)
		diag_P2 = torch.where(
			diag_P2.abs() < 1e-12,
			torch.full_like(diag_P2, 1e-12),
			diag_P2,
		)
		diag_term = (1 + reg_E * torch.diag(P2C)) / diag_P2

		B = reg_E * P2C - P2 * diag_term.unsqueeze(1)
		B.fill_diagonal_(0)
		return B

	@torch.no_grad()
	def predict(self) -> torch.Tensor:
		return self.rate_matrix.mm(self.item_similarity)
