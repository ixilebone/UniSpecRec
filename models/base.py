import torch
from torch import nn
from utils import Metric, cal_bpr_loss, cal_infonce_loss
from typing import Literal
from dataclasses import dataclass

class ModelBase(nn.Module):
    @dataclass
    class ModelConfig():
        device: str = 'cpu'
    def __init__(self, model_config: ModelConfig):
        super(ModelBase, self).__init__()
        self.model_config = model_config

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        """生成预测评分矩阵，形状是(num_users, num_items)"""
        raise NotImplementedError("Subclasses should implement this!")

    def test(self, metric: Metric, test_rate_matrix: torch.Tensor):
        return metric.eval(train_matrix=self.rate_matrix, test_matrix=test_rate_matrix, pred_matrix=self.predict())
    

class TrainableModelBase(ModelBase):
    @dataclass
    class BPRLossConfig():
        type: Literal["bpr"] = "bpr"
        similarity: Literal["cos", "dot"] = "dot"
        num_neg_item: int = 1
    @dataclass
    class InfoNCELossConfig():
        type: Literal["infonce"] = "infonce"
        similarity: Literal["cos", "dot"] = "dot"
        temperature: float = 0.5
        is_inbatch: bool = False
        num_neg_item: int = 1

    LossConfig = BPRLossConfig | InfoNCELossConfig
    def __init__(
        self,
        model_config: ModelBase.ModelConfig,
        loss_config: LossConfig,
    ):
        super(TrainableModelBase, self).__init__(model_config=model_config)
        self.loss_config = loss_config
        
    def get_loss(
        self,
        user_index: torch.Tensor, # shape: (batch_size,)
        pos_item_index: torch.Tensor, # shape: (batch_size,)
        neg_item_indices: torch.Tensor, # shape: (batch_size, num_neg_item)
    ) -> torch.Tensor:
        batch_size = user_index.shape[0]

        combined_user_index = torch.cat([user_index, user_index], dim=0)
        combined_item_index = torch.cat([pos_item_index, neg_item_indices.flatten()], dim=0)
        
        combined_user_embedding, combined_item_embedding = self.forward(
            user_index=combined_user_index,
            item_index=combined_item_index
        )
        
        user_embedding = combined_user_embedding[:batch_size]
        pos_item_embedding = combined_item_embedding[:batch_size]
        neg_item_embeddings = combined_item_embedding[batch_size:].view(batch_size, -1, combined_item_embedding.shape[1])

        match self.loss_config.type:
            case "bpr":
                loss = cal_bpr_loss(
                    user_embedding=user_embedding,
                    pos_item_embedding=pos_item_embedding,
                    neg_item_embeddings=neg_item_embeddings,
                    similarity=self.loss_config.similarity,
                )
            case "infonce":
                loss = cal_infonce_loss(
                    user_embedding=user_embedding,
                    pos_item_embedding=pos_item_embedding,
                    neg_item_embeddings=neg_item_embeddings,
                    similarity=self.loss_config.similarity,
                    temperature=self.loss_config.temperature,
                    is_inbatch=self.loss_config.is_inbatch,
                    pos_item_index=pos_item_index if self.loss_config.is_inbatch else None,
                )
            case _:
                raise ValueError(f"Unsupported loss type: {self.loss_config.type}")

        return loss
    
    def forward(self, user_index: torch.Tensor, item_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """根据用户索引和物品索引，生成对应的用户嵌入和物品嵌入，形状都是(batch_size, embedding_dim)"""
        raise NotImplementedError("Subclasses should implement this!")
        user_embedding = None
        item_embedding = None
        return user_embedding, item_embedding


import time
from utils import DataLoader, grid_search_hyperparamters

def train_model(
    model: TrainableModelBase,
    dataloader: DataLoader,
    metric: Metric,
    epoch: int,
    batch_size: int,
) -> tuple[list[float], dict[str, list[float]]]:
    
    num_reports = 20
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    
    # 在训练前先做一次test（初始化模型的性能）
    results = {k: [v] for k, v in model.test(metric=metric, test_rate_matrix=dataloader.test_rate_matrix).items()}
    losses = []

    time1 = time.time()
    get_batch_time = 0.0
    zero_grad_time = 0.0
    get_loss_time = 0.0
    backward_time = 0.0
    step_time = 0.0

    for i in range(epoch):
        # 直接用整数索引获取当前 step 对应的预采样批次
        get_batch_start_time = time.time()
        user_index, pos_item_index, neg_item_indices = dataloader.sample(
            num=batch_size,
            num_neg_item=model.loss_config.num_neg_item,
        )
        get_batch_time += time.time() - get_batch_start_time

        zero_grad_start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        zero_grad_time += time.time() - zero_grad_start_time

        get_loss_start_time = time.time()
        loss = model.get_loss(
            user_index=user_index,
            pos_item_index=pos_item_index,
            neg_item_indices=neg_item_indices,
        )
        get_loss_time += time.time() - get_loss_start_time

        backward_start_time = time.time()
        loss.backward()
        backward_time += time.time() - backward_start_time

        step_start_time = time.time()
        optimizer.step()
        step_time += time.time() - step_start_time

        # 打印固定的 num_reports 次，并防止除零错误
        report_interval = max(1, epoch // num_reports)
        if (i + 1) % report_interval == 0 or i == epoch - 1:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # 及时清理梯度和激活缓存
            time2 = time.time()
            losses.append(loss.item())
            print(f"Epoch {i+1}/{epoch}, Loss: {loss.item():.4f}, Time: {time2-time1:.2f}s")
            print(f"Get Batch Time: {get_batch_time:.2f}s, Zero Grad Time: {zero_grad_time:.2f}s, Get Loss Time: {get_loss_time:.2f}s, Backward Time: {backward_time:.2f}s, Step Time: {step_time:.2f}s")
            get_batch_time = 0.0
            zero_grad_time = 0.0
            get_loss_time = 0.0
            backward_time = 0.0
            step_time = 0.0


            for k, v in model.test(metric=metric, test_rate_matrix=dataloader.test_rate_matrix).items():
                results[k].append(v)
            
            # 重置 CUDA 缓存分配器，避免 predict() 中逐实体循环造成的内存碎片
            # 累积影响后续训练的 attention 大块内存分配效率
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            time1 = time.time()
            print(f"Test Metrics: " + ", ".join([f"{k}: {results[k][-1]:.4f}" for k in results.keys()]) + f", Time: {time1 - time2:.2f}s")
    return losses, results

def train_base_model(
    model: TrainableModelBase,
    dataloader: DataLoader,
    metric: Metric,
    epoch: int,
    batch_size: int,
    normalization_method: str = 'zscore',
) -> tuple[list[float], dict[str, list[float]]]:
    
    num_reports = 20
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    
    # 在训练前先做一次test（初始化模型的性能）
    base_results = {k: [v] for k, v in model.test(metric=metric, test_rate_matrix=dataloader.test_rate_matrix).items()}
    plus_results = {k: [v] for k, v in grid_search_hyperparamters(cf_pred_matrix=model.predict(), dataloader=dataloader, normalization_method=normalization_method)[2].items()}

    losses = []

    time1 = time.time()
    get_batch_time = 0.0
    zero_grad_time = 0.0
    get_loss_time = 0.0
    backward_time = 0.0
    step_time = 0.0

    for i in range(epoch):
        # 直接用整数索引获取当前 step 对应的预采样批次
        get_batch_start_time = time.time()
        user_index, pos_item_index, neg_item_indices = dataloader.sample(
            num=batch_size,
            num_neg_item=model.loss_config.num_neg_item,
        )
        get_batch_time += time.time() - get_batch_start_time

        zero_grad_start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        zero_grad_time += time.time() - zero_grad_start_time

        get_loss_start_time = time.time()
        loss = model.get_loss(
            user_index=user_index,
            pos_item_index=pos_item_index,
            neg_item_indices=neg_item_indices,
        )
        get_loss_time += time.time() - get_loss_start_time

        backward_start_time = time.time()
        loss.backward()
        backward_time += time.time() - backward_start_time

        step_start_time = time.time()
        optimizer.step()
        step_time += time.time() - step_start_time

        # 打印固定的 num_reports 次，并防止除零错误
        report_interval = max(1, epoch // num_reports)
        if (i + 1) % report_interval == 0 or i == epoch - 1:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # 及时清理梯度和激活缓存
            time2 = time.time()
            losses.append(loss.item())
            print(f"Epoch {i+1}/{epoch}, Loss: {loss.item():.4f}, Time: {time2-time1:.2f}s")
            print(f"Get Batch Time: {get_batch_time:.2f}s, Zero Grad Time: {zero_grad_time:.2f}s, Get Loss Time: {get_loss_time:.2f}s, Backward Time: {backward_time:.2f}s, Step Time: {step_time:.2f}s")
            get_batch_time = 0.0
            zero_grad_time = 0.0
            get_loss_time = 0.0
            backward_time = 0.0
            step_time = 0.0

            for k, v in model.test(metric=metric, test_rate_matrix=dataloader.test_rate_matrix).items():
                base_results[k].append(v)
            
            time1 = time.time()
            print(f"Test Metrics: " + ", ".join([f"{k}: {base_results[k][-1]:.4f}" for k in base_results.keys()]) + f", Time: {time1 - time2:.2f}s")

            best_power, best_gamma, best_result = grid_search_hyperparamters(cf_pred_matrix=model.predict(), dataloader=dataloader, normalization_method=normalization_method)
            for k, v in best_result.items():
                plus_results[k].append(v)

            # 重置 CUDA 缓存分配器，避免 predict() 中逐实体循环造成的内存碎片
            # 累积影响后续训练的 attention 大块内存分配效率
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            
    return losses, plus_results, base_results

