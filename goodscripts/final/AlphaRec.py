import sys
sys.path.append('..')
import torch
from utils import *
from models import *

# 9种最优参数: {(interaction_data, semantic_data): {param: value}}
BEST_PARAMS = {
    # nv-embed-v2
    ('books', 'nvidia'): {'model_pred_similarity': 'cos', 'loss_temperature': 0.1},
    ('games', 'nvidia'): {'model_pred_similarity': 'dot', 'loss_temperature': 0.25},
    ('toys',  'nvidia'): {'model_pred_similarity': 'dot', 'loss_temperature': 0.15},
    # llama-3.2-3b
    ('books', 'llama'):  {'model_pred_similarity': 'cos', 'loss_temperature': 0.15},
    ('games', 'llama'):  {'model_pred_similarity': 'cos', 'loss_temperature': 0.25},
    ('toys',  'llama'):  {'model_pred_similarity': 'cos', 'loss_temperature': 0.15},
    # qwen3-embedding-8b
    ('books', 'qwen'):   {'model_pred_similarity': 'cos', 'loss_temperature': 0.1},
    ('games', 'qwen'):   {'model_pred_similarity': 'dot', 'loss_temperature': 0.3},
    ('toys',  'qwen'):   {'model_pred_similarity': 'dot', 'loss_temperature': 0.15},
}

def main(
    interaction_data,
    semantic_data,
    num_epochs=200,
    device='cuda'
):
    key = (interaction_data, semantic_data)
    if key not in BEST_PARAMS:
        raise ValueError(f"Unknown combination: {key}. Available: {list(BEST_PARAMS.keys())}")

    params = BEST_PARAMS[key]
    model_pred_similarity = params['model_pred_similarity']
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
    model = AlphaRec(
        rate_matrix=datahandler.rate_matrix,
        user_semantic_embeddings=datahandler.user_semantic_embeddings,
        item_semantic_embeddings=datahandler.item_semantic_embeddings,
        model_config=AlphaRec.ModelConfig(
            num_layers=2,
            latent_dim=32,
            device=device,
            similarity=model_pred_similarity
        ),
        loss_config=AlphaRec.InfoNCELossConfig(
            type='infonce',
            similarity='cos',
            temperature=loss_temperature,
            is_inbatch=True,
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

"""输入python -m goodscripts.final.AlphaRec --interaction_data books --semantic_data nvidia --num_epochs 200 --device cuda"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AlphaRec Training Script")
    parser.add_argument('--interaction_data', type=str, required=True, choices=['books', 'games', 'toys'])
    parser.add_argument('--semantic_data', type=str, required=True, choices=['nvidia', 'llama', 'qwen'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(
        interaction_data=args.interaction_data,
        semantic_data=args.semantic_data,
        num_epochs=args.num_epochs,
        device=args.device,
    )