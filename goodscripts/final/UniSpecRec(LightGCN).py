import sys
sys.path.append('..')
import torch
from utils import *
from models import *

# 3种最优参数: {interaction_data: {param: value}}
BEST_PARAMS = {
    'books': {'similarity': 'cos', 'loss_temperature': 0.1},
    'games': {'similarity': 'dot', 'loss_temperature': 0.25},
    'toys':  {'similarity': 'dot', 'loss_temperature': 0.15},
}

def main(
    interaction_data,
    semantic_data,
    normalization_method='zscore',
    num_epochs=200,
    device='cuda'
):
    if interaction_data not in BEST_PARAMS:
        raise ValueError(f"Unknown dataset: {interaction_data}. Available: {list(BEST_PARAMS.keys())}")

    params = BEST_PARAMS[interaction_data]
    similarity = params['similarity']
    loss_temperature = params['loss_temperature']

    datahandler = DataHandler(
        interaction_data=interaction_data,
        semantic_data=semantic_data,
        num_neg_item=1,
        batch_size=4096,
        device=device,
        seed=42,
    )
    metric = Metric(device=device)
    model = LightGCN(
        rate_matrix=datahandler.rate_matrix,
        model_config=LightGCN.ModelConfig(
            num_layers=2,
            latent_dim=32,
            device=device,
            similarity=similarity
        ),
        loss_config=LightGCN.InfoNCELossConfig(
            type='infonce',
            similarity='cos',
            temperature=loss_temperature,
            is_inbatch=True,
        )
    )
    train_unispecrec(
        datahandler=datahandler,
        metric=metric,
        model=model,
        num_epochs=num_epochs,
        verbose=True,
        use_amp=True,
        normalization_method=normalization_method,
    )

"""输入python -m goodscripts.final.UniSpecRec(LightGCN)"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UniSpecRec(LightGCN) Training Script")
    parser.add_argument('--interaction_data', type=str, required=True, choices=['books', 'games', 'toys'])
    parser.add_argument('--semantic_data', type=str, required=True, choices=['nvidia', 'llama', 'qwen'])
    parser.add_argument('--normalization_method', type=str, default='zscore', choices=['direct', 'max', 'minmax', 'zscore', 'l2', 'softmax'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(
        interaction_data=args.interaction_data,
        semantic_data=args.semantic_data,
        normalization_method=args.normalization_method,
        num_epochs=args.num_epochs,
        device=args.device,
    )
