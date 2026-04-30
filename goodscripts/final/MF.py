import sys
sys.path.append('..')
from utils import *
from models import *

# 3种数据集共用同一组最优参数
BEST_PARAMS = {
    'books': {'embedding_dim': 32, 'model_similarity': 'dot', 'loss_similarity': 'cos', 'loss_temperature': 0.3},
    'games': {'embedding_dim': 32, 'model_similarity': 'dot', 'loss_similarity': 'cos', 'loss_temperature': 0.3},
    'toys':  {'embedding_dim': 32, 'model_similarity': 'dot', 'loss_similarity': 'cos', 'loss_temperature': 0.3},
}


def main(
    interaction_data,
    num_epochs=200,
    device='cuda'
):
    if interaction_data not in BEST_PARAMS:
        raise ValueError(f"Unknown dataset: {interaction_data}. Available: {list(BEST_PARAMS.keys())}")

    params = BEST_PARAMS[interaction_data]

    datahandler = DataHandler(
        interaction_data=interaction_data,
        semantic_data=None,
        num_neg_item=64,
        batch_size=4096,
        device=device,
        seed=42,
    )
    metric = Metric(device=device)
    model = MF(
        rate_matrix=datahandler.rate_matrix,
        model_config=MF.ModelConfig(
            embedding_dim=params['embedding_dim'],
            device=device,
            similarity=params['model_similarity'],
        ),
        loss_config=MF.InfoNCELossConfig(
            type='infonce',
            similarity=params['loss_similarity'],
            temperature=params['loss_temperature'],
            is_inbatch=True,
            num_neg_item=64,
        )
    )
    train_model(
        datahandler=datahandler,
        metric=metric,
        model=model,
        num_epochs=num_epochs,
        verbose=True,
        use_amp=True
    )


"""输入python -m goodscripts.final.MF --interaction_data books --num_epochs 200 --device cuda"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MF Training Script")
    parser.add_argument('--interaction_data', type=str, required=True, choices=['books', 'games', 'toys'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(
        interaction_data=args.interaction_data,
        num_epochs=args.num_epochs,
        device=args.device,
    )
