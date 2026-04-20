import torch
from models.base import ModelBase
from dataclasses import dataclass


class SGFCF(ModelBase):
	@dataclass
	class ModelConfig(ModelBase.ModelConfig):
		k: int = 100
		beta_1: float = 1.0
		beta_2: float = 1.0
		alpha: float = 0.0
		eps: float = 0.5
		gamma: float = 1.0

	def __init__(
		self,
		rate_matrix: torch.Tensor,
		model_config: ModelConfig,
	):
		super(SGFCF, self).__init__(model_config=model_config)
		self.device = self.model_config.device
		self.rate_matrix = (rate_matrix > 0).float().to(self.device)

		self.k = self.model_config.k
		self.beta_1 = self.model_config.beta_1
		self.beta_2 = self.model_config.beta_2
		self.alpha = self.model_config.alpha
		self.eps = self.model_config.eps
		self.gamma = self.model_config.gamma
		self.svd_oversample = 200
		self.svd_niter = 30

		self.pred_matrix = self._build_pred_matrix()

	def _individual_weight(self, singular_values: torch.Tensor, homo_ratio: torch.Tensor) -> torch.Tensor:
		y_min, y_max = self.beta_1, self.beta_2
		x_min = homo_ratio.min()
		x_max = homo_ratio.max()

		if torch.isclose(x_max, x_min):
			homo_weight = torch.full_like(homo_ratio, fill_value=(y_min + y_max) / 2.0)
		else:
			homo_weight = (
				(y_max - y_min) / (x_max - x_min) * homo_ratio
				+ (x_max * y_min - y_max * x_min) / (x_max - x_min)
			)

		return singular_values.pow(homo_weight.unsqueeze(1))

	def _compute_homophily(self, freq_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		num_users, num_items = freq_matrix.shape

		train_data = [torch.where(freq_matrix[u] > 0)[0] for u in range(num_users)]
		train_data_item = [torch.where(freq_matrix[:, i] > 0)[0] for i in range(num_items)]

		homo_ratio_user = []
		for u in range(num_users):
			interacted_items = train_data[u]
			if interacted_items.numel() > 1:
				inter_items = freq_matrix[:, interacted_items].t().clone()
				inter_items[:, u] = 0
				connect_matrix = inter_items.mm(inter_items.t())

				size = inter_items.shape[0]
				ratio_u = (
					(connect_matrix != 0).sum().item()
					- (connect_matrix.diag() != 0).sum().item()
				) / (size * (size - 1))
				homo_ratio_user.append(ratio_u)
			else:
				homo_ratio_user.append(0.0)

		homo_ratio_item = []
		for i in range(num_items):
			interacted_users = train_data_item[i]
			if interacted_users.numel() > 1:
				inter_users = freq_matrix[interacted_users].clone()
				inter_users[:, i] = 0
				connect_matrix = inter_users.mm(inter_users.t())

				size = inter_users.shape[0]
				ratio_i = (
					(connect_matrix != 0).sum().item()
					- (connect_matrix.diag() != 0).sum().item()
				) / (size * (size - 1))
				homo_ratio_item.append(ratio_i)
			else:
				homo_ratio_item.append(0.0)

		return (
			torch.tensor(homo_ratio_user, dtype=freq_matrix.dtype, device=freq_matrix.device),
			torch.tensor(homo_ratio_item, dtype=freq_matrix.dtype, device=freq_matrix.device),
		)

	def _row_normalize(self, matrix: torch.Tensor) -> torch.Tensor:
		row_sum = matrix.sum(dim=1, keepdim=True)
		row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
		return matrix / row_sum

	def _build_pred_matrix(self) -> torch.Tensor:
		freq_matrix = self.rate_matrix
		num_users, num_items = freq_matrix.shape

		homo_ratio_user, homo_ratio_item = self._compute_homophily(freq_matrix)

		user_degree = freq_matrix.sum(dim=1)
		item_degree = freq_matrix.sum(dim=0)

		D_u = 1.0 / (user_degree + self.alpha).pow(self.eps)
		D_i = 1.0 / (item_degree + self.alpha).pow(self.eps)
		D_u[torch.isinf(D_u)] = 0
		D_i[torch.isinf(D_i)] = 0

		norm_freq_matrix = D_u.unsqueeze(1) * freq_matrix * D_i.unsqueeze(0)

		q = min(self.k + self.svd_oversample, min(num_users, num_items))
		U, value, V = torch.svd_lowrank(norm_freq_matrix, q=q, niter=self.svd_niter)
		value = value / value.max().clamp_min(1e-12)

		k = min(self.k, value.shape[0])
		U_k = U[:, :k]
		V_k = V[:, :k]
		value_k = value[:k]

		user_weight = self._individual_weight(value_k, homo_ratio_user)
		item_weight = self._individual_weight(value_k, homo_ratio_item)
		low_freq_score = (U_k * user_weight).mm((V_k * item_weight).t())
		low_freq_score = self._row_normalize(low_freq_score)

		high_freq_score = norm_freq_matrix.mm(norm_freq_matrix.t()).mm(norm_freq_matrix)
		high_freq_score = self._row_normalize(high_freq_score)

		rate_matrix = (low_freq_score + self.gamma * high_freq_score).sigmoid()
		return rate_matrix

	@torch.no_grad()
	def predict(self) -> torch.Tensor:
		return self.pred_matrix
