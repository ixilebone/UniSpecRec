import sys
sys.path.append("..")
from utils import DataHandler, Metric, find_best_result_from_results_list
import torch
import tqdm


# 矩阵处理方法集合
class MatrixNormalization:
    """矩阵行级别的归一化方法"""
    
    @staticmethod
    def direct_combine(matrix):
        """直接使用，不做处理"""
        return matrix
    
    @staticmethod
    def max_normalize(matrix):
        """按行除以最大值"""
        max_val = matrix.abs().max(dim=1, keepdim=True)[0]
        max_val = torch.where(max_val == 0, torch.ones_like(max_val), max_val)  # 避免除以0
        return matrix / max_val
    
    @staticmethod
    def min_max_normalize(matrix):
        """按行进行min-max归一化到[0,1]"""
        min_val = matrix.min(dim=1, keepdim=True)[0]
        max_val = matrix.max(dim=1, keepdim=True)[0]
        range_val = max_val - min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
        return (matrix - min_val) / range_val
    
    @staticmethod
    def z_score_normalize(matrix):
        """按行进行z-score标准化"""
        mean = matrix.mean(dim=1, keepdim=True)
        std = matrix.std(dim=1, keepdim=True)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return (matrix - mean) / std
    
    @staticmethod
    def l2_normalize(matrix):
        """按行进行L2归一化"""
        norm = torch.norm(matrix, p=2, dim=1, keepdim=True)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        return matrix / norm
    
    @staticmethod
    def softmax_normalize(matrix):
        """按行进行softmax归一化"""
        # 减去每行最大值以提高数值稳定性
        matrix_shifted = matrix - matrix.max(dim=1, keepdim=True)[0]
        exp_matrix = torch.exp(matrix_shifted)
        softmax_matrix = exp_matrix / exp_matrix.sum(dim=1, keepdim=True)
        return softmax_matrix

    @staticmethod
    def get_method(method_name):
        """根据名称获取方法"""
        methods = {
            'direct': MatrixNormalization.direct_combine,
            'max': MatrixNormalization.max_normalize,
            'minmax': MatrixNormalization.min_max_normalize,
            'zscore': MatrixNormalization.z_score_normalize,
            'l2': MatrixNormalization.l2_normalize,
            'softmax': MatrixNormalization.softmax_normalize,
        }
        return methods.get(method_name, MatrixNormalization.direct_combine)





def grid_search_unispecrec_hyperparamters(
    cf_pred_matrix: torch.Tensor,
    datahandler: DataHandler,
    metric: Metric,
    normalization_method='zscore',
    eval_rate_matrix: torch.Tensor = None,
    show_progress: bool = True,
) -> tuple[float, float, dict[str, float]]:
    def _search_best_gamma(
        pred_matrix_1: torch.Tensor,
        pred_matrix_2: torch.Tensor,
        rate_matrix: torch.Tensor,
        test_rate_matrix: torch.Tensor,
        normalization_method: str,
        metric: Metric,
    ) -> tuple[float, dict[str, float]]:
        normalize_fn = MatrixNormalization.get_method(normalization_method)
        
        # 对两个矩阵进行归一化处理
        norm_pred_matrix_1 = normalize_fn(pred_matrix_1)
        norm_pred_matrix_2 = normalize_fn(pred_matrix_2)
        
        gamma_list = [i*0.01 for i in range(101)]
        results = []
        for gamma in gamma_list:
            pred_matrix = (1 - gamma) * norm_pred_matrix_1 + gamma * norm_pred_matrix_2
            results.append(metric.eval(train_matrix=rate_matrix, test_matrix=test_rate_matrix, pred_matrix=pred_matrix))
        
        index, best_result = find_best_result_from_results_list(results)
        best_gamma = gamma_list[index]
        return best_gamma, best_result
    
    if eval_rate_matrix is None:
        eval_rate_matrix = datahandler.test_rate_matrix

    (U, S, V_T) = torch.linalg.svd(torch.cat([datahandler.user_semantic_embeddings, datahandler.item_semantic_embeddings], dim=0))

    power_list = [i*0.01 for i in range(0, 101, 5)]

    results = []
    best_gammas = []
    powers = []
    power_iter = tqdm.tqdm(power_list) if show_progress else power_list
    for power in power_iter:
        se_pred_matrix = U[:datahandler.num_users, :len(S)] * torch.pow(S, power) @ (U[datahandler.num_users:, :len(S)] * torch.pow(S, power)).T
        best_gamma, best_result = _search_best_gamma(
            pred_matrix_1=cf_pred_matrix,
            pred_matrix_2=se_pred_matrix,
            rate_matrix=datahandler.rate_matrix,
            test_rate_matrix=eval_rate_matrix,
            normalization_method=normalization_method,
            metric=metric,
        )
        results.append(best_result)
        best_gammas.append(best_gamma)
        powers.append(power)
        del se_pred_matrix
    index, best_result = find_best_result_from_results_list(results)
    best_power = power_list[index]
    best_gamma = best_gammas[index]
    print(f"Method: {normalization_method} | Best power: {best_power}, Best gamma: {best_gamma}, Best result: {best_result}")
    return best_power, best_gamma, best_result