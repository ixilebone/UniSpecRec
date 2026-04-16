from utils import *
from models import *
from copy import deepcopy
from dataclasses import dataclass
import torch
import tqdm
import time
from .grid_search import grid_search_unispecrec_hyperparamters

@dataclass
class TrainingDebugInfo:
    """单个报告周期的详细调试信息"""
    data_wait_time: float = 0.0
    zero_grad_time: float = 0.0
    get_loss_time: float = 0.0
    backward_time: float = 0.0
    step_time: float = 0.0
    eval_time: float = 0.0
    other_time: float = 0.0

def train_epoch(
    datahandler: DataHandler,
    optimizer: torch.optim.Optimizer,
    model: TrainableModelBase,
    device: str = 'cuda',
    use_amp: bool = True,
    verbose: bool = False
) -> tuple[float, TrainingDebugInfo]:
    model.train()
    total_loss = 0.0
    num_batches = 0
    debug_info = TrainingDebugInfo()
    epoch_start = time.time()

    use_amp = use_amp and device != 'cpu'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def _sync_cuda():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    dataloader = datahandler.get_train_dataloader(shuffle=True)
    data_iter = iter(dataloader)
    while True:
        fetch_start = time.time()
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        # CPU -> GPU 异步传输
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        debug_info.data_wait_time += time.time() - fetch_start

        if verbose:
            _sync_cuda()
            zero_grad_start = time.time()
            optimizer.zero_grad(set_to_none=True)
            _sync_cuda()
            debug_info.zero_grad_time += time.time() - zero_grad_start

            _sync_cuda()
            get_loss_start = time.time()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = model.get_loss(
                    user_index=batch['user_index'],
                    pos_item_index=batch['pos_item_index'],
                    neg_item_indices=batch['neg_item_indices']
                )
            _sync_cuda()
            debug_info.get_loss_time += time.time() - get_loss_start

            _sync_cuda()
            backward_start = time.time()
            scaler.scale(loss).backward()
            _sync_cuda()
            debug_info.backward_time += time.time() - backward_start

            _sync_cuda()
            step_start = time.time()
            scaler.step(optimizer)
            scaler.update()
            _sync_cuda()
            debug_info.step_time += time.time() - step_start
        else:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = model.get_loss(
                    user_index=batch['user_index'],
                    pos_item_index=batch['pos_item_index'],
                    neg_item_indices=batch['neg_item_indices']
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    epoch_total = time.time() - epoch_start
    measured = (
        debug_info.data_wait_time
        + debug_info.zero_grad_time
        + debug_info.get_loss_time
        + debug_info.backward_time
        + debug_info.step_time
    )
    debug_info.other_time = max(epoch_total - measured, 0.0)

    return avg_loss, debug_info

def train_model(
    datahandler: DataHandler,
    metric: Metric,
    model: TrainableModelBase,
    num_epochs: int = 3000,
    num_steps: int = 20,
    patience_steps: int = 5,
    primary_metric: str = 'NDCG@20',
    verbose: bool = True,
    use_amp: bool = True,
) -> dict:
    """
    训练推荐模型

    Args:
        datahandler: 数据处理器
        metric: 评估指标
        model: 训练模型
        num_epochs: 训练轮数
        num_steps: 报告间隔（每 num_epochs//num_steps 轮报告一次）
        patience_steps: 早停耐心值
        primary_metric: 主要优化指标
        verbose: 是否打印详细的调试信息
        use_amp: 是否使用混合精度训练
        
    Returns:
        最终测试结果
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    best_state_dict = None
    now_patience = 0
    device = model.model_config.device
    
    # 初始化：打印初始模型性能
    model.eval()
    print(f"Initial Model Performance (Before Training):")
    initial_validation_result = model.test(metric=metric, test_rate_matrix=datahandler.valid_rate_matrix)
    best_validation_result = initial_validation_result
    print("  " + ", ".join([f"{k}: {v:.4f}" for k, v in initial_validation_result.items()]))
    
    report_interval = max(1, num_epochs // num_steps)
    epoch_start_time = time.time()
    accumulated_debug_info = TrainingDebugInfo()  # 累积report周期内的时间

    for epoch in range(num_epochs):
        avg_loss, debug_info = train_epoch(
            datahandler,
            optimizer,
            model,
            device=device,
            use_amp=use_amp,
            verbose=verbose
        )
        
        # 累积debug信息
        accumulated_debug_info.data_wait_time += debug_info.data_wait_time
        accumulated_debug_info.zero_grad_time += debug_info.zero_grad_time
        accumulated_debug_info.get_loss_time += debug_info.get_loss_time
        accumulated_debug_info.backward_time += debug_info.backward_time
        accumulated_debug_info.step_time += debug_info.step_time
        accumulated_debug_info.other_time += debug_info.other_time
        
        # 定期验证和报告
        if (epoch + 1) % report_interval == 0 or epoch == num_epochs - 1:
            model.eval()
            eval_start = time.time()
            validation_result = model.test(metric=metric, test_rate_matrix=datahandler.valid_rate_matrix)
            accumulated_debug_info.eval_time += time.time() - eval_start
            epoch_elapsed = time.time() - epoch_start_time
            
            # 单行报告：epoch、loss、时间、指标
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in validation_result.items()])
            known = (
                accumulated_debug_info.data_wait_time
                + accumulated_debug_info.zero_grad_time
                + accumulated_debug_info.get_loss_time
                + accumulated_debug_info.backward_time
                + accumulated_debug_info.step_time
                + accumulated_debug_info.eval_time
            )
            residual_other = max(epoch_elapsed - known, 0.0)
            time_breakdown = (
                f"Data:{accumulated_debug_info.data_wait_time:.1f}s "
                f"ZG:{accumulated_debug_info.zero_grad_time:.1f}s "
                f"Loss:{accumulated_debug_info.get_loss_time:.1f}s "
                f"BW:{accumulated_debug_info.backward_time:.1f}s "
                f"Step:{accumulated_debug_info.step_time:.1f}s "
                f"Eval:{accumulated_debug_info.eval_time:.1f}s "
                f"Other:{residual_other:.1f}s"
            ) if verbose else ""
            time_str = f"Time: {epoch_elapsed:.1f}s {time_breakdown}"
            print(f"Epoch [{epoch+1:>5d}/{num_epochs}]  Loss: {avg_loss:.4f}  {time_str}  {metrics_str}")

            # 早停逻辑
            if validation_result[primary_metric] > best_validation_result[primary_metric]:
                best_validation_result = deepcopy(validation_result)
                best_state_dict = deepcopy(model.state_dict())
                now_patience = 0
            else:
                now_patience += 1
                if now_patience > patience_steps:
                    print(f"\n{'='*80}")
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Best validation results:")
                    print(f"  " + ", ".join([f"{k}: {v:.4f}" for k, v in best_validation_result.items()]))
                    print(f"{'='*80}")
                    break
            
            # 重置epoch时钟和累积信息
            epoch_start_time = time.time()
            accumulated_debug_info = TrainingDebugInfo()

    # 加载最优模型并测试
    print(f"\nLoading best model and evaluating on test set...")
    model.load_state_dict(best_state_dict)
    test_result = model.test(metric=metric, test_rate_matrix=datahandler.test_rate_matrix)
    print(f"{'='*80}")
    print(f"Final test results:")
    print(f"  " + ", ".join([f"{k}: {v:.4f}" for k, v in test_result.items()]))
    print(f"{'='*80}")
    return test_result


def train_unispecrec(
    datahandler: DataHandler,
    metric: Metric,
    model: TrainableModelBase,
    num_epochs: int = 3000,
    num_steps: int = 20,
    patience_steps: int = 5,
    primary_metric: str = 'NDCG@20',
    verbose: bool = True,
    use_amp: bool = True,
    normalization_method: str = 'zscore',
) -> dict:
    """
    训练推荐模型，每个验证步都用UniSpecRec网格搜索在验证集上评估，最后在测试集上测试

    Args:
        datahandler: 数据处理器
        metric: 评估指标
        model: 训练模型
        num_epochs: 训练轮数
        num_steps: 报告间隔（每 num_epochs//num_steps 轮报告一次）
        patience_steps: 早停耐心值
        primary_metric: 主要优化指标
        verbose: 是否打印详细的调试信息
        use_amp: 是否使用混合精度训练
        normalization_method: 矩阵归一化方法

    Returns:
        最终测试结果
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    best_state_dict = None
    best_power = 0.0
    best_gamma = 0.0
    now_patience = 0
    device = model.model_config.device

    # 初始化：用UniSpecRec评估初始模型
    model.eval()
    print(f"Initial Model Performance (Before Training):")
    cf_pred_matrix = model.predict()
    init_power, init_gamma, initial_validation_result = grid_search_unispecrec_hyperparamters(
        cf_pred_matrix=cf_pred_matrix,
        datahandler=datahandler,
        metric=metric,
        normalization_method=normalization_method,
        eval_rate_matrix=datahandler.valid_rate_matrix,
        show_progress=False,
    )
    best_validation_result = initial_validation_result
    best_power = init_power
    best_gamma = init_gamma
    best_state_dict = deepcopy(model.state_dict())
    metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in initial_validation_result.items()])
    print(f"  {metrics_str}  UniSpecRec: power={init_power}, gamma={init_gamma}")

    report_interval = max(1, num_epochs // num_steps)
    epoch_start_time = time.time()
    accumulated_debug_info = TrainingDebugInfo()

    for epoch in range(num_epochs):
        avg_loss, debug_info = train_epoch(
            datahandler,
            optimizer,
            model,
            device=device,
            use_amp=use_amp,
            verbose=verbose
        )

        # 累积debug信息
        accumulated_debug_info.data_wait_time += debug_info.data_wait_time
        accumulated_debug_info.zero_grad_time += debug_info.zero_grad_time
        accumulated_debug_info.get_loss_time += debug_info.get_loss_time
        accumulated_debug_info.backward_time += debug_info.backward_time
        accumulated_debug_info.step_time += debug_info.step_time
        accumulated_debug_info.other_time += debug_info.other_time

        # 定期验证和报告
        if (epoch + 1) % report_interval == 0 or epoch == num_epochs - 1:
            model.eval()
            eval_start = time.time()

            # 用UniSpecRec在验证集上做网格搜索
            cf_pred_matrix = model.predict()
            step_power, step_gamma, validation_result = grid_search_unispecrec_hyperparamters(
                cf_pred_matrix=cf_pred_matrix,
                datahandler=datahandler,
                metric=metric,
                normalization_method=normalization_method,
                eval_rate_matrix=datahandler.valid_rate_matrix,
                show_progress=True,
            )
            accumulated_debug_info.eval_time += time.time() - eval_start
            epoch_elapsed = time.time() - epoch_start_time

            # 单行报告
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in validation_result.items()])
            known = (
                accumulated_debug_info.data_wait_time
                + accumulated_debug_info.zero_grad_time
                + accumulated_debug_info.get_loss_time
                + accumulated_debug_info.backward_time
                + accumulated_debug_info.step_time
                + accumulated_debug_info.eval_time
            )
            residual_other = max(epoch_elapsed - known, 0.0)
            time_breakdown = (
                f"Data:{accumulated_debug_info.data_wait_time:.1f}s "
                f"ZG:{accumulated_debug_info.zero_grad_time:.1f}s "
                f"Loss:{accumulated_debug_info.get_loss_time:.1f}s "
                f"BW:{accumulated_debug_info.backward_time:.1f}s "
                f"Step:{accumulated_debug_info.step_time:.1f}s "
                f"Eval:{accumulated_debug_info.eval_time:.1f}s "
                f"Other:{residual_other:.1f}s"
            ) if verbose else ""
            time_str = f"Time: {epoch_elapsed:.1f}s {time_breakdown}"
            print(f"Epoch [{epoch+1:>5d}/{num_epochs}]  Loss: {avg_loss:.4f}  {time_str}  {metrics_str}  UniSpecRec: power={step_power}, gamma={step_gamma}")

            # 早停逻辑（基于UniSpecRec在验证集上的结果）
            if validation_result[primary_metric] > best_validation_result[primary_metric]:
                best_validation_result = deepcopy(validation_result)
                best_state_dict = deepcopy(model.state_dict())
                best_power = step_power
                best_gamma = step_gamma
                now_patience = 0
            else:
                now_patience += 1
                if now_patience > patience_steps:
                    print(f"\n{'='*80}")
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Best validation results (UniSpecRec):")
                    print(f"  Power: {best_power}, Gamma: {best_gamma}")
                    print(f"  " + ", ".join([f"{k}: {v:.4f}" for k, v in best_validation_result.items()]))
                    print(f"{'='*80}")
                    break

            # 重置epoch时钟和累积信息
            epoch_start_time = time.time()
            accumulated_debug_info = TrainingDebugInfo()

    # 加载最优模型，用最佳超参数在测试集上评估
    print(f"\nLoading best model and evaluating on test set...")
    model.load_state_dict(best_state_dict)
    model.eval()
    cf_pred_matrix = model.predict()

    # 用验证集上搜索到的最优(power, gamma)构造UniSpecRec模型，在测试集上评估
    from models.UniSpecRec import UniSpecRec
    unispecrec = UniSpecRec(
        cf_pred_matrix=cf_pred_matrix,
        user_semantic_embeddings=datahandler.user_semantic_embeddings,
        item_semantic_embeddings=datahandler.item_semantic_embeddings,
        rate_matrix=datahandler.rate_matrix,
        model_config=UniSpecRec.ModelConfig(
            power=best_power,
            gamma=best_gamma,
            normalization_method=normalization_method,
            device=device,
        ),
    )
    test_result = unispecrec.test(metric=metric, test_rate_matrix=datahandler.test_rate_matrix)

    print(f"\n{'='*80}")
    print(f"Final test results (UniSpecRec):")
    print(f"  Power: {best_power}, Gamma: {best_gamma}")
    print(f"  " + ", ".join([f"{k}: {v:.4f}" for k, v in test_result.items()]))
    print(f"{'='*80}")
    return test_result