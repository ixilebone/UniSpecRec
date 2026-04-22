import sys
sys.path.append('..')
from utils import *
from models import *

# 9种最优参数: {(interaction_data, semantic_data): {param: value}}
BEST_PARAMS = {
    ('books', 'nvidia'): {'reg_X': -100, 'reg_F': 10, 'reg_E': 150},
    ('games', 'nvidia'): {'reg_X': -50, 'reg_F': 5, 'reg_E': 150},
    ('toys', 'nvidia'): {'reg_X': -100, 'reg_F': 10, 'reg_E': 200},
    ('books', 'llama'): {'reg_X': -50, 'reg_F': 0.05, 'reg_E': 100},
    ('games', 'llama'): {'reg_X': 20, 'reg_F': 0.1, 'reg_E': 80},
    ('toys', 'llama'): {'reg_X': 20, 'reg_F': 0.05, 'reg_E': 80},
    ('books', 'qwen'): {'reg_X': 0, 'reg_F': 1, 'reg_E': 50},
    ('games', 'qwen'): {'reg_X': 0, 'reg_F': 5, 'reg_E': 100},
    ('toys', 'qwen'): {'reg_X': -50, 'reg_F': 10, 'reg_E': 150},
}


def main(
    interaction_data,
    semantic_data,
    device='cuda'
):
    key = (interaction_data, semantic_data)
    if key not in BEST_PARAMS:
        raise ValueError(f"Unknown combination: {key}. Available: {list(BEST_PARAMS.keys())}")

    params = BEST_PARAMS[key]

    datahandler = DataHandler(
        interaction_data=interaction_data,
        semantic_data=semantic_data,
        device=device,
        seed=42,
    )
    metric = Metric(device=device)

    model = L3AE(
        rate_matrix=datahandler.rate_matrix,
        item_semantic_embeddings=datahandler.item_semantic_embeddings,
        model_config=L3AE.ModelConfig(
            reg_X=params['reg_X'],
            reg_F=params['reg_F'],
            reg_E=params['reg_E'],
            device=device,
        ),
    )

    valid_result = model.test(metric=metric, test_rate_matrix=datahandler.valid_rate_matrix)
    test_result = model.test(metric=metric, test_rate_matrix=datahandler.test_rate_matrix)

    print(f"L3AE Params ({interaction_data}, {semantic_data}): {params}")
    print("Validation:", ", ".join([f"{k}: {v:.4f}" for k, v in valid_result.items()]))
    print("Test:", ", ".join([f"{k}: {v:.4f}" for k, v in test_result.items()]))


"""输入python -m goodscripts.final.L3AE"""
"""输入python -m goodscripts.final.L3AE --interaction_data games --semantic_data qwen --device cuda"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="L3AE Evaluation Script")
    parser.add_argument('--interaction_data', type=str, required=True, choices=['books', 'games', 'toys'])
    parser.add_argument('--semantic_data', type=str, required=True, choices=['nvidia', 'llama', 'qwen'])
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(
        interaction_data=args.interaction_data,
        semantic_data=args.semantic_data,
        device=args.device,
    )
