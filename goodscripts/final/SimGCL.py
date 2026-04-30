import sys
sys.path.append('..')
from utils import *
from models import *

# 3种最优参数: {interaction_data: {param: value}}
BEST_PARAMS = {
    'books': {'similarity': 'dot', 'reg_weight': 0.0, 'cl_weight': 0.02, 'cl_tau': 0.15, 'eps': 0.01},
    'games': {'similarity': 'dot', 'reg_weight': 1e-6, 'cl_weight': 0.02, 'cl_tau': 0.2, 'eps': 0.01},
    'toys':  {'similarity': 'dot', 'reg_weight': 1e-8, 'cl_weight': 0.02, 'cl_tau': 0.15, 'eps': 0.1},
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
        num_neg_item=1,
        batch_size=4096,
        device=device,
        seed=42,
    )
    metric = Metric(device=device)
    model = SimGCL(
        rate_matrix=datahandler.rate_matrix,
        model_config=SimGCL.ModelConfig(
            num_layers=2,
            latent_dim=32,
            device=device,
            similarity=params['similarity'],
            reg_weight=params['reg_weight'],
            cl_weight=params['cl_weight'],
            cl_tau=params['cl_tau'],
            eps=params['eps'],
        ),
        loss_config=SimGCL.BPRLossConfig(
            type='bpr',
            similarity=params['similarity'],
            num_neg_item=1,
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


"""输入python -m goodscripts.final.SimGCL --interaction_data books --num_epochs 200 --device cuda"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SimGCL Training Script")
    parser.add_argument('--interaction_data', type=str, required=True, choices=['books', 'games', 'toys'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(
        interaction_data=args.interaction_data,
        num_epochs=args.num_epochs,
        device=args.device,
    )
