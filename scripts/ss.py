import sys
sys.path.append('..')  # Ensure the root directory is in the path
import torch
from utils import *
from models import *
from itertools import product

DEVICE = 'cuda'
plotter = Plotter()
dataloader = DataLoader(
    interaction_data='games',
    semantic_data='qwen',
    device=DEVICE,
)
metric = Metric(device=DEVICE)
epoch = 2000
model = AlphaRec(
    rate_matrix=dataloader.rate_matrix,
    user_semantic_embeddings=dataloader.user_semantic_embeddings,
    item_semantic_embeddings=dataloader.item_semantic_embeddings,
    model_config=AlphaRec.ModelConfig(
        num_layers=2,
        latent_dim=32,
        device=DEVICE
    ),
    loss_config=AlphaRec.InfoNCELossConfig(
        type='infonce',
        similarity='cos',
        temperature=0.3,
        is_inbatch=True,
    )
)
train_model(
    model=model,
    dataloader=dataloader,
    metric=metric,
    epoch=epoch,
    batch_size=1024,
)
"""输入python -m scripts.ss"""