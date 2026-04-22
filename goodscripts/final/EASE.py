import sys
sys.path.append('..')
from utils import *
from models import *

# 3种最优参数: {interaction_data: {param: value}}
BEST_PARAMS = {
    'books': {'regu_lambda': 100.0},
    'games': {'regu_lambda': 83},
    'toys': {'regu_lambda': 100.0},
}


def main(
    interaction_data,
    device='cuda'
):
    if interaction_data not in BEST_PARAMS:
        raise ValueError(f"Unknown dataset: {interaction_data}. Available: {list(BEST_PARAMS.keys())}")

    params = BEST_PARAMS[interaction_data]

    datahandler = DataHandler(
        interaction_data=interaction_data,
        semantic_data=None,
        device=device,
        seed=42,
    )
    metric = Metric(device=device)

    model = EASE(
        rate_matrix=datahandler.rate_matrix,
        model_config=EASE.ModelConfig(
            regu_lambda=params['regu_lambda'],
            device=device,
        ),
    )

    valid_result = model.test(metric=metric, test_rate_matrix=datahandler.valid_rate_matrix)
    test_result = model.test(metric=metric, test_rate_matrix=datahandler.test_rate_matrix)

    print(f"EASE Params ({interaction_data}): {params}")
    print("Validation:", ", ".join([f"{k}: {v:.4f}" for k, v in valid_result.items()]))
    print("Test:", ", ".join([f"{k}: {v:.4f}" for k, v in test_result.items()]))


"""输入python -m goodscripts.final.EASE"""
"""输入python -m goodscripts.final.EASE --interaction_data games --device cuda"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EASE Evaluation Script")
    parser.add_argument('--interaction_data', type=str, required=True, choices=['books', 'games', 'toys'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(
        interaction_data=args.interaction_data,
        device=args.device,
    )
