import sys
sys.path.append('..')
from utils import *
from models import *
from goodscripts.final.MF import BEST_PARAMS as BEST_MF_PARAMS


def main(
    interaction_data,
    semantic_data,
    normalization_method='zscore',
    num_epochs=200,
    device='cuda'
):
    if interaction_data not in BEST_MF_PARAMS:
        raise ValueError(f"Unknown dataset: {interaction_data}. Available: {list(BEST_MF_PARAMS.keys())}")

    mf_params = BEST_MF_PARAMS[interaction_data]

    datahandler = DataHandler(
        interaction_data=interaction_data,
        semantic_data=semantic_data,
        num_neg_item=64,
        batch_size=4096,
        device=device,
        seed=42,
    )
    metric = Metric(device=device)

    model = MF(
        rate_matrix=datahandler.rate_matrix,
        model_config=MF.ModelConfig(
            embedding_dim=mf_params['embedding_dim'],
            device=device,
            similarity=mf_params['model_similarity'],
        ),
        loss_config=MF.InfoNCELossConfig(
            type='infonce',
            similarity=mf_params['loss_similarity'],
            temperature=mf_params['loss_temperature'],
            is_inbatch=True,
            num_neg_item=64,
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


"""输入python -m goodscripts.final.UniSpecRec(MF)"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UniSpecRec(MF) Training Script")
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
