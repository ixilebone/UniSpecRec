import torch
import pickle
import scipy.sparse
import numpy as np
import math
from typing import Literal

def load_tensor_from_pickle(path: str, device: str='cpu') -> torch.Tensor:
    with open(path, 'rb') as f:
        data = pickle.load(f)
        match data:

            case scipy.sparse.coo_matrix() | scipy.sparse.csr_matrix():
                return torch.tensor(data.toarray(), dtype=torch.float32, device=device)
            case torch.Tensor():
                return data.to(dtype=torch.float32, device=device)
            case np.matrix() | np.ndarray():
                return torch.tensor(data, dtype=torch.float32, device=device)
            case _:
                raise ValueError(f"Unsupported data type: {type(data)}")

import pathlib
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader


class InteractionDataset(TorchDataset):
    def __init__(
        self,
        rate_matrix: torch.Tensor,
        check_conflict: bool = False,
        seed: int | None = None
    ):
        self.check_conflict = check_conflict
        self.seed = seed

        # Dataset 全部在 CPU 上操作，训练循环中再传到 GPU
        self._rate_matrix_cpu = rate_matrix.cpu()
        self._precompute_sampling_data()

    def _precompute_sampling_data(self):
        rm = self._rate_matrix_cpu
        num_users = rm.shape[0]

        nz = rm.nonzero(as_tuple=False)
        nnz = nz.size(0)

        if nnz == 0:
            self._items_per_user = torch.zeros(num_users, 1, dtype=torch.long)
            self._items_per_user_cnt = torch.zeros(num_users, dtype=torch.long)
            self._num_edges = 0
            return

        rows, cols = nz[:, 0], nz[:, 1]

        self._edge_users = rows
        self._edge_items = cols
        self._num_edges = nnz

    def __len__(self) -> int:
        if self._num_edges == 0:
            raise RuntimeError("No valid interactions for sampling.")
        return self._num_edges

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._edge_users[idx], self._edge_items[idx]


class NegSamplingCollate:
    """在 batch 级别一次性生成所有负样本，避免 per-sample 的 Python 调度开销"""

    def __init__(self, num_items: int, num_neg_item: int):
        self.num_items = num_items
        self.num_neg_item = num_neg_item

    def __call__(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict:
        user_indices, pos_item_indices = zip(*batch)
        return {
            'user_index': torch.stack(user_indices),
            'pos_item_index': torch.stack(pos_item_indices),
            'neg_item_indices': torch.randint(
                0, self.num_items, (len(batch), self.num_neg_item), dtype=torch.long
            )
        }


class FastNegSamplingDataLoader:
    """直接在 CPU Tensor 上按批采样，绕过 DataLoader/collate 的进程与 Python 调度开销。"""

    def __init__(
        self,
        edge_users: torch.Tensor,
        edge_items: torch.Tensor,
        num_items: int,
        batch_size: int,
        num_neg_item: int,
        shuffle: bool,
        pin_memory: bool = True,
        generator: torch.Generator | None = None
    ):
        self.edge_users = edge_users
        self.edge_items = edge_items
        self.num_items = num_items
        self.batch_size = batch_size
        self.num_neg_item = num_neg_item
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.generator = generator

        self._num_edges = edge_users.shape[0]

    def __len__(self) -> int:
        if self._num_edges == 0:
            return 0
        return math.ceil(self._num_edges / self.batch_size)

    def __iter__(self):
        if self._num_edges == 0:
            return

        if self.shuffle:
            order = torch.randperm(self._num_edges, generator=self.generator)
            users = self.edge_users[order]
            items = self.edge_items[order]
        else:
            users = self.edge_users
            items = self.edge_items

        for start in range(0, self._num_edges, self.batch_size):
            end = min(start + self.batch_size, self._num_edges)
            bs = end - start

            user_index = users[start:end]
            pos_item_index = items[start:end]
            neg_item_indices = torch.randint(
                0,
                self.num_items,
                (bs, self.num_neg_item),
                dtype=torch.long,
                generator=self.generator
            )

            if self.pin_memory:
                user_index = user_index.pin_memory()
                pos_item_index = pos_item_index.pin_memory()
                neg_item_indices = neg_item_indices.pin_memory()

            yield {
                'user_index': user_index,
                'pos_item_index': pos_item_index,
                'neg_item_indices': neg_item_indices
            }


class DataHandler:
    def __init__(
        self,
        interaction_data: Literal['games', 'toys', 'books'],
        semantic_data: Literal['llama', 'nvidia', 'qwen'] | None = None,
        device: str = "cuda",
        batch_size: int = 32,
        num_neg_item: int = 64,
        check_conflict: bool = False,
        num_workers: int = 16,
        use_fast_sampling: bool = True,
        seed: int | None = 42
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_neg_item = num_neg_item
        self.check_conflict = check_conflict
        self.seed = seed
        self.num_workers = num_workers
        self.use_fast_sampling = use_fast_sampling

        self.project_dir = pathlib.Path(__file__).parent.parent
        self.rate_matrix_path = self.project_dir / f'data/{interaction_data}/trn_mat.pkl'
        self.test_rate_matrix_path = self.project_dir / f'data/{interaction_data}/tst_mat.pkl'
        self.valid_rate_matrix_path = self.project_dir / f'data/{interaction_data}/val_mat.pkl'

        self.rate_matrix = load_tensor_from_pickle(self.rate_matrix_path, device=device)
        self.test_rate_matrix = load_tensor_from_pickle(self.test_rate_matrix_path, device=device)
        self.valid_rate_matrix = load_tensor_from_pickle(self.valid_rate_matrix_path, device=device)
        self.num_users = self.rate_matrix.shape[0]
        self.num_items = self.rate_matrix.shape[1]

        self.num_users = self.rate_matrix.shape[0]
        self.num_items = self.rate_matrix.shape[1]
        self.num_interactions = int(self.rate_matrix.count_nonzero().item())

        self.semantic_data = semantic_data
        if semantic_data is not None:
            semantic_data2llm_map = {
                'llama': 'llama-3.2-3b',
                'nvidia': 'nv-embed-v2',
                'qwen': 'qwen3-embedding-8b',
            }
            llm_name = semantic_data2llm_map.get(semantic_data, semantic_data)

            user_emb_path = self.project_dir / f'data/{interaction_data}/{llm_name}_users.pkl'
            item_emb_path = self.project_dir / f'data/{interaction_data}/{llm_name}_items.pkl'

            self.user_semantic_embeddings = torch.nn.functional.normalize(
                load_tensor_from_pickle(user_emb_path, device=device),
                p=2,
                dim=1
            )
            self.item_semantic_embeddings = torch.nn.functional.normalize(
                load_tensor_from_pickle(item_emb_path, device=device),
                p=2,
                dim=1
            )
        else:
            self.user_semantic_embeddings = None
            self.item_semantic_embeddings = None

        self._train_dataloader = None
        self._sample_generator = None
        if self.seed is not None:
            self._sample_generator = torch.Generator(device='cpu')
            self._sample_generator.manual_seed(self.seed)

    def create_torch_dataloader(
        self,
        rate_matrix: torch.Tensor,
        shuffle: bool = True
    ) -> TorchDataLoader:
        dataset = InteractionDataset(
            rate_matrix=rate_matrix,
            check_conflict=self.check_conflict,
            seed=self.seed
        )
        collate_fn = NegSamplingCollate(
            num_items=rate_matrix.shape[1],
            num_neg_item=self.num_neg_item
        )

        return TorchDataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn
        )

    def get_train_dataloader(self, shuffle: bool = True) -> TorchDataLoader | FastNegSamplingDataLoader:
        if self._train_dataloader is None:
            if self.use_fast_sampling:
                train_dataset = InteractionDataset(
                    rate_matrix=self.rate_matrix,
                    check_conflict=self.check_conflict,
                    seed=self.seed
                )
                self._train_dataloader = FastNegSamplingDataLoader(
                    edge_users=train_dataset._edge_users,
                    edge_items=train_dataset._edge_items,
                    num_items=self.num_items,
                    batch_size=self.batch_size,
                    num_neg_item=self.num_neg_item,
                    shuffle=shuffle,
                    pin_memory=True,
                    generator=self._sample_generator
                )
            else:
                self._train_dataloader = self.create_torch_dataloader(
                    rate_matrix=self.rate_matrix,
                    shuffle=shuffle
                )
        return self._train_dataloader

    def get_info(self) -> dict:
        return {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_interactions': self.num_interactions,
            'batch_size': self.batch_size,
            'num_neg_item': self.num_neg_item,
            'device': self.device,
            'semantic_data': self.semantic_data,
            'num_workers': self.num_workers,
            'use_fast_sampling': self.use_fast_sampling
        }
