import sys
sys.path.append('..')
from utils import *
from models import *


# SGFCF 在三个数据集上的最优参数
BEST_SGFCF_PARAMS = {
	'books': {'k': 2500, 'beta_1': 1.0, 'beta_2': 1.0, 'alpha': 0.0, 'eps': 0.26, 'gamma': 2.5},
	'games': {'k': 130, 'beta_1': 1.0, 'beta_2': 1.2, 'alpha': 19.0, 'eps': 0.5, 'gamma': 0.5},
	'toys': {'k': 440, 'beta_1': 0.5, 'beta_2': 1.7, 'alpha': 9.0, 'eps': 0.5, 'gamma': 2.5},
}

# 如果已经跑过网格搜索，可把结果填到这里，脚本会优先使用这里的值。
BEST_UNISPECREC_PARAMS = {
	# 'books': {'power': 0.0, 'gamma': 0.0, 'normalization_method': 'zscore'},
	# 'games': {'power': 0.0, 'gamma': 0.0, 'normalization_method': 'zscore'},
	# 'toys': {'power': 0.0, 'gamma': 0.0, 'normalization_method': 'zscore'},
}


def _resolve_unispecrec_params(interaction_data, power, gamma, normalization_method):
	if power is not None and gamma is not None:
		return {
			'power': power,
			'gamma': gamma,
			'normalization_method': normalization_method,
		}

	if interaction_data in BEST_UNISPECREC_PARAMS:
		params = BEST_UNISPECREC_PARAMS[interaction_data]
		return {
			'power': params['power'],
			'gamma': params['gamma'],
			'normalization_method': params.get('normalization_method', normalization_method),
		}

	raise ValueError(
		"UniSpecRec 参数未提供。请传入 --power 和 --gamma，"
		"或先运行网格搜索脚本并把最优参数写入 BEST_UNISPECREC_PARAMS。"
	)


def main(
	interaction_data,
	semantic_data,
	normalization_method='zscore',
	power=None,
	gamma=None,
	device='cuda',
):
	if interaction_data not in BEST_SGFCF_PARAMS:
		raise ValueError(f"Unknown dataset: {interaction_data}. Available: {list(BEST_SGFCF_PARAMS.keys())}")

	unispec_params = _resolve_unispecrec_params(
		interaction_data=interaction_data,
		power=power,
		gamma=gamma,
		normalization_method=normalization_method,
	)
	sgfcf_params = BEST_SGFCF_PARAMS[interaction_data]

	datahandler = DataHandler(
		interaction_data=interaction_data,
		semantic_data=semantic_data,
		device=device,
		seed=42,
	)
	metric = Metric(device=device)

	sgfcf_model = SGFCF(
		rate_matrix=datahandler.rate_matrix,
		model_config=SGFCF.ModelConfig(
			k=sgfcf_params['k'],
			beta_1=sgfcf_params['beta_1'],
			beta_2=sgfcf_params['beta_2'],
			alpha=sgfcf_params['alpha'],
			eps=sgfcf_params['eps'],
			gamma=sgfcf_params['gamma'],
			device=device,
		),
	)
	cf_pred_matrix = sgfcf_model.predict()

	unispecrec_model = UniSpecRec(
		cf_pred_matrix=cf_pred_matrix,
		user_semantic_embeddings=datahandler.user_semantic_embeddings,
		item_semantic_embeddings=datahandler.item_semantic_embeddings,
		rate_matrix=datahandler.rate_matrix,
		model_config=UniSpecRec.ModelConfig(
			power=unispec_params['power'],
			gamma=unispec_params['gamma'],
			normalization_method=unispec_params['normalization_method'],
			device=device,
		),
	)

	valid_result = unispecrec_model.test(metric=metric, test_rate_matrix=datahandler.valid_rate_matrix)
	test_result = unispecrec_model.test(metric=metric, test_rate_matrix=datahandler.test_rate_matrix)

	print(f"SGFCF Params ({interaction_data}): {sgfcf_params}")
	print(f"UniSpecRec Params ({interaction_data}): {unispec_params}")
	print("Validation:", ", ".join([f"{k}: {v:.4f}" for k, v in valid_result.items()]))
	print("Test:", ", ".join([f"{k}: {v:.4f}" for k, v in test_result.items()]))


"""输入python -m goodscripts.final.UniSpecRec(SGFCF) --interaction_data games --semantic_data qwen --power 0.6 --gamma 0.35 --device cuda"""
if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="UniSpecRec(SGFCF) Evaluation Script")
	parser.add_argument('--interaction_data', type=str, required=True, choices=['books', 'games', 'toys'])
	parser.add_argument('--semantic_data', type=str, required=True, choices=['nvidia', 'llama', 'qwen'])
	parser.add_argument('--normalization_method', type=str, default='zscore', choices=['direct', 'max', 'minmax', 'zscore', 'l2', 'softmax'])
	parser.add_argument('--power', type=float, default=None, help='UniSpecRec power，不传则尝试读取 BEST_UNISPECREC_PARAMS')
	parser.add_argument('--gamma', type=float, default=None, help='UniSpecRec gamma，不传则尝试读取 BEST_UNISPECREC_PARAMS')
	parser.add_argument('--device', type=str, default='cuda')
	args = parser.parse_args()

	main(
		interaction_data=args.interaction_data,
		semantic_data=args.semantic_data,
		normalization_method=args.normalization_method,
		power=args.power,
		gamma=args.gamma,
		device=args.device,
	)
