import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_best_result_from_results_list(results_list: list[dict[str, float]], metric_name: str="NDCG@20",) -> tuple[int, dict[str, float]]:
    """从结果列表中找到指定指标最好的结果的索引"""
    best_index = 0
    best_value = float('-inf')
    best_metrics = {}
    for i, result in enumerate(results_list):
        if result[metric_name] > best_value:
            best_value = result[metric_name]
            best_index = i
    best_metrics = results_list[best_index]
    best_metrics = {k: round(v, 4) for k, v in best_metrics.items()}
    return best_index, best_metrics

def find_best_result_from_results_dict(results_dict: dict[str, list[float]], metric_name: str="NDCG@20",) -> tuple[int, dict[str, float]]:
    """从结果字典中找到指定指标最好的结果的索引"""
    best_index = 0
    best_value = float('-inf')
    best_metrics = {}
    for i, value in enumerate(results_dict[metric_name]):
        if value > best_value:
            best_value = value
            best_index = i
    best_metrics = {metric: round(values[best_index], 4) for metric, values in results_dict.items()}
    return best_index, best_metrics

def results_list_to_results_dict(results_list: list[dict[str, float]]) -> dict[str, list[float]]:
    """将结果列表转换为指标名称到数值列表的字典"""
    results_dict = {}
    for result in results_list:
        for metric_name, value in result.items():
            if metric_name not in results_dict:
                results_dict[metric_name] = []
            results_dict[metric_name].append(round(value, 4))
    return results_dict

def results_dict_to_results_list(results_dict: dict[str, list[float]]) -> list[dict[str, float]]:
    """将结果字典转换为结果列表"""
    num_results = len(next(iter(results_dict.values())))  # 获取结果数量
    results_list = []
    for i in range(num_results):
        result = {metric_name: values[i] for metric_name, values in results_dict.items()}
        results_list.append(result)
    return results_list

