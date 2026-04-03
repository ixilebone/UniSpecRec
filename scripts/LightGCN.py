import sys
sys.path.append('..')  # Ensure the root directory is in the path
import torch
from utils import *
from models import *
from itertools import product

DEVICE = 'cuda'
plotter = Plotter()
datahandler = DataLoader(
    interaction_data='games',
    semantic_data='qwen',
    device=DEVICE,
)
metric = Metric(device=DEVICE)
epoch = 20000

cf_model = LightGCN(
    rate_matrix=datahandler.rate_matrix,
    model_config=LightGCN.ModelConfig(
        latent_dim=32,
        num_layers=2,
        device=DEVICE,
        similarity='dot'
    ),
    loss_config=LightGCN.InfoNCELossConfig(
        type='infonce',
        similarity='cos',
        temperature=0.3,
        is_inbatch=True,
    )
)
_, _, _ = train_model(
    model=cf_model,
    dataloader=datahandler,
    metric=metric,
    epoch=epoch,
    batch_size=4096,
)
"""python -m scripts.LightGCN"""
"""source ./.venv/bin/activate"""