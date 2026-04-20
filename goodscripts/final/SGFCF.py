import sys
sys.path.append('..')
from utils import *
from models import *

# 3种最优参数: {interaction_data: {param: value}}
BEST_PARAMS = {
	'books': {'k': 2500, 'beta_1': 1.0, 'beta_2': 1.0, 'alpha': 0.0, 'eps': 0.26, 'gamma': 2.5},
	'games': {'k': 130, 'beta_1': 1.0, 'beta_2': 1.2, 'alpha': 19.0, 'eps': 0.5, 'gamma': 0.5},
	'toys': {'k': 440, 'beta_1': 0.5, 'beta_2': 1.7, 'alpha': 9.0, 'eps': 0.5, 'gamma': 2.5},
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

	model = SGFCF(
		rate_matrix=datahandler.rate_matrix,
		model_config=SGFCF.ModelConfig(
			k=params['k'],
			beta_1=params['beta_1'],
			beta_2=params['beta_2'],
			alpha=params['alpha'],
			eps=params['eps'],
			gamma=params['gamma'],
			device=device,
		),
	)

	valid_result = model.test(metric=metric, test_rate_matrix=datahandler.valid_rate_matrix)
	test_result = model.test(metric=metric, test_rate_matrix=datahandler.test_rate_matrix)

	print(f"SGFCF Params ({interaction_data}): {params}")
	print("Validation:", ", ".join([f"{k}: {v:.4f}" for k, v in valid_result.items()]))
	print("Test:", ", ".join([f"{k}: {v:.4f}" for k, v in test_result.items()]))

"""激活环境，输入source .venv/bin/activate"""
"""输入python -m goodscripts.final.SGFCF"""
"""输入python -m goodscripts.final.SGFCF --interaction_data games --device cuda"""
if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="SGFCF Evaluation Script")
	parser.add_argument('--interaction_data', type=str, required=True, choices=['books', 'games', 'toys'])
	parser.add_argument('--device', type=str, default='cuda')
	args = parser.parse_args()

	main(
		interaction_data=args.interaction_data,
		device=args.device,
	)
