import sys
sys.path.append('..')  # Ensure the root directory is in the path
import torch
from utils import *
from models import *
from itertools import product

DEVICE = 'cuda'
plotter = Plotter()
datahandler = DataHandler(
    interaction_data='games',
    semantic_data='qwen',
    num_neg_item=1,
    batch_size=1024,
    device=DEVICE,
    seed=42,
)
metric = Metric(device=DEVICE)
epoch = 2000
model = AlphaRec(
    rate_matrix=datahandler.rate_matrix,
    user_semantic_embeddings=datahandler.user_semantic_embeddings,
    item_semantic_embeddings=datahandler.item_semantic_embeddings,
    model_config=AlphaRec.ModelConfig(
        num_layers=2,
        latent_dim=32,
        device=DEVICE,
        similarity='dot'
    ),
    loss_config=AlphaRec.InfoNCELossConfig(
        type='infonce',
        similarity='cos',
        temperature=0.3,
        is_inbatch=True,
    )
)
"""
train_model(
    model=model,
    dataloader=datahandler,
    metric=metric,
    epoch=epoch,
    batch_size=1024,
)
"""

train(
    datahandler=datahandler,
    metric=metric,
    model=model,
    num_epochs=100,
    verbose=True,  # 打印每个epoch的调试信息
    use_amp=False
)
"""输入python -m scripts.AlphaRec"""