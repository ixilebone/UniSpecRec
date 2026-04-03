import sys
sys.path.append("..")
from utils import DataLoader, Metric, find_best_result_from_results_list
import torch
import tqdm
from itertools import product


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


def search_best_gamma(
    pred_matrix_1,
    pred_matrix_2,
    rate_matrix,
    test_rate_matrix,
    normalization_method,
    metric = Metric(device="cuda"),
):
    """
    搜索最优的gamma参数来加权两个预测矩阵
    
    Args:
        pred_matrix_1: 第一个预测矩阵
        pred_matrix_2: 第二个预测矩阵
        rate_matrix: 训练集评分矩阵
        test_rate_matrix: 测试集评分矩阵
        normalization_method: 矩阵处理方法，选项为 'direct', 'max', 'minmax', 'zscore', 'l2', 'softmax'
    
    Returns:
        best_gamma: 最优的gamma值
        best_result: 最优的评估结果
    """
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


def search_best_hyperparamters(cf_pred_matrix, dataloader, normalization_method='zscore'):
    """
    搜索最优超参数
    
    Args:
        normalization_method: 矩阵归一化方法 ('direct', 'max', 'minmax', 'zscore', 'l2', 'softmax')
    """

    (U, S, V_T) = torch.linalg.svd(torch.cat([dataloader.user_semantic_embeddings, dataloader.item_semantic_embeddings], dim=0))
    
    power_list = [i*0.01 for i in range(0, 101, 5)]

    results = []
    best_gammas = []
    powers = []
    for power in tqdm.tqdm(power_list):
        se_pred_matrix = U[:dataloader.num_users, :len(S)] * torch.pow(S, power) @ (U[dataloader.num_users:, :len(S)] * torch.pow(S, power)).T
        best_gamma, best_result = search_best_gamma(
            pred_matrix_1=cf_pred_matrix,
            pred_matrix_2=se_pred_matrix,
            rate_matrix=dataloader.rate_matrix,
            test_rate_matrix=dataloader.test_rate_matrix,
            normalization_method=normalization_method,
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


if __name__ == "__main__":
    """输入python -m scripts.uni_grid_search"""
    from models import *
 
    method_list = ['zscore']  # ['minmax', 'zscore', 'softmax']
    interaction_data_list = ["games"]  # ['nvidia', 'qwen', 'games']
    semantic_data_list = ['nvidia', 'llama', 'qwen']
    
    print("\n=== 搜索最优超参数 ===")
    for method, interaction_data, semantic_data in product(method_list, interaction_data_list, semantic_data_list):
        dataloader = DataLoader(
            interaction_data=interaction_data,
            semantic_data=semantic_data,
            device="cuda",
        )

        ease_model = EASE(
            rate_matrix=dataloader.rate_matrix,
            model_config=EASE.get_best_model_config(interaction_data=interaction_data, device="cuda")
        )

        lightgcn_model = LightGCN(
            rate_matrix=dataloader.rate_matrix,
            model_config=LightGCN.ModelConfig(latent_dim=32, num_layers=3, device="cuda"),
        )

        cf_pred_matrix = lightgcn_model.predict()
        search_best_hyperparamters(
            cf_pred_matrix=cf_pred_matrix,
            dataloader=dataloader,
            normalization_method=method
        )
