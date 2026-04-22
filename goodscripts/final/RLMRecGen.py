import sys
sys.path.append('..')
from utils import *
from models import LightGCN, RLMRecGen

# 9种参数组合: {(interaction_data, semantic_data): {param: value}}
# 目前使用 LightGCN 的最优底座参数 + RLMRecGen 默认增强参数，后续可继续替换为网格搜索结果。
BEST_PARAMS = {
    ('books', 'nvidia'): {'similarity': 'cos', 'loss_temperature': 0.1, 'mask_ratio': 0.2, 'recon_weight': 1.0, 'recon_temperature': 0.1, 'hidden_dim': 0},
    ('games', 'nvidia'): {'similarity': 'dot', 'loss_temperature': 0.25, 'mask_ratio': 0.2, 'recon_weight': 1.0, 'recon_temperature': 0.1, 'hidden_dim': 0},
    ('toys', 'nvidia'): {'similarity': 'dot', 'loss_temperature': 0.15, 'mask_ratio': 0.2, 'recon_weight': 1.0, 'recon_temperature': 0.1, 'hidden_dim': 0},
    ('books', 'llama'): {'similarity': 'cos', 'loss_temperature': 0.1, 'mask_ratio': 0.2, 'recon_weight': 1.0, 'recon_temperature': 0.1, 'hidden_dim': 0},
    ('games', 'llama'): {'similarity': 'dot', 'loss_temperature': 0.25, 'mask_ratio': 0.2, 'recon_weight': 1.0, 'recon_temperature': 0.1, 'hidden_dim': 0},
    ('toys', 'llama'): {'similarity': 'dot', 'loss_temperature': 0.15, 'mask_ratio': 0.2, 'recon_weight': 1.0, 'recon_temperature': 0.1, 'hidden_dim': 0},
    ('books', 'qwen'): {'similarity': 'cos', 'loss_temperature': 0.1, 'mask_ratio': 0.2, 'recon_weight': 1.0, 'recon_temperature': 0.1, 'hidden_dim': 0},
    ('games', 'qwen'): {'similarity': 'dot', 'loss_temperature': 0.25, 'mask_ratio': 0.2, 'recon_weight': 1.0, 'recon_temperature': 0.1, 'hidden_dim': 0},
    ('toys', 'qwen'): {'similarity': 'dot', 'loss_temperature': 0.15, 'mask_ratio': 0.2, 'recon_weight': 1.0, 'recon_temperature': 0.1, 'hidden_dim': 0},
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

    datahandler = DataHandler(
        interaction_data=interaction_data,
        semantic_data=semantic_data,
        num_neg_item=1,
        batch_size=4096,
        device=device,
        seed=42,
    )
    metric = Metric(device=device)

    base_model = LightGCN(
        rate_matrix=datahandler.rate_matrix,
        model_config=LightGCN.ModelConfig(
            num_layers=2,
            latent_dim=32,
            device=device,
            similarity=params['similarity'],
        ),
        loss_config=LightGCN.InfoNCELossConfig(
            type='infonce',
            similarity='cos',
            temperature=params['loss_temperature'],
            is_inbatch=True,
        ),
    )
    model = RLMRecGen(
        base_model=base_model,
        user_semantic_embeddings=datahandler.user_semantic_embeddings,
        item_semantic_embeddings=datahandler.item_semantic_embeddings,
        model_config=RLMRecGen.ModelConfig(
            mask_ratio=params['mask_ratio'],
            recon_weight=params['recon_weight'],
            recon_temperature=params['recon_temperature'],
            hidden_dim=params['hidden_dim'],
            device=device,
        ),
    )
    train_model(
        datahandler=datahandler,
        metric=metric,
        model=model,
        num_epochs=num_epochs,
        verbose=True,
        use_amp=True,
    )


"""输入python -m goodscripts.final.RLMRecGen"""
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLMRecGen Training Script")
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
