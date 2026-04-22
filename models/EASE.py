import torch
from models.base import ModelBase
from dataclasses import dataclass

class EASE(ModelBase):
    @dataclass
    class ModelConfig(ModelBase.ModelConfig):
        regu_lambda: float = 100.0
    def __init__(
        self,
        rate_matrix: torch.Tensor,
        model_config: ModelConfig,
    ):
        super(EASE, self).__init__(model_config=model_config)
        self.device = self.model_config.device
        self.rate_matrix = rate_matrix.to(self.device)
        self.regu_lambda = self.model_config.regu_lambda
        self.item_weights = self._compute_item_weights()

    def _compute_item_weights(self):
        G = self.rate_matrix.T @ self.rate_matrix
        
        diag_indices = torch.arange(G.shape[0], device=self.device)
        G[diag_indices, diag_indices] += self.regu_lambda

        P = torch.linalg.inv(G)
        
        self.item_weights = P / (-P.diag())
        self.item_weights[diag_indices, diag_indices] = 0.0

        return self.item_weights

    def predict(self) -> torch.Tensor:
        pred_matrix = self.rate_matrix @ self.item_weights
        return pred_matrix

from utils import DataHandler, Metric
import matplotlib.pyplot as plt
import tqdm

def test_ease_by_regu_lambda(device: str = 'cpu'):
    datahandler = DataHandler(
        interaction_data="games",
        semantic_data=None,
        device=device,
    )
    metric = Metric(k_list=[10, 20], device=device)
    fig, ax = plt.subplots()
    regu_lambdas = range(1, 401, 1)
    results = None
    for regu_lambda in tqdm.tqdm(regu_lambdas):
        ease_model = EASE(rate_matrix=datahandler.rate_matrix, model_config=EASE.ModelConfig(regu_lambda=regu_lambda, device=device))

        
        test_rate_matrix = datahandler.test_rate_matrix

        score = ease_model.test(metric, test_rate_matrix)
        if results is None:
            results = {k: [] for k in score.keys()}
        for k, v in score.items():
            results[k].append(v)

    for k in results.keys():
        ax.plot(regu_lambdas, results[k], label=k)
        print(f"Max {k}: {max(results[k])} at lambda={regu_lambdas[results[k].index(max(results[k]))]}")
    ax.legend()
    ax.set_xlabel('Regularization Lambda')
    ax.set_ylabel('Score')
    ax.set_title('EASE Performance vs Regularization Lambda')
    plt.show()

if __name__ == "__main__":
    '''终端输入python -m models.EASE'''
    test_ease_by_regu_lambda(device='cuda')
