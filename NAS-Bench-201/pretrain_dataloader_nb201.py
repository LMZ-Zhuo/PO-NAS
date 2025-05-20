import torch
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

class NASDataset(Dataset):

    def __init__(self, data_dict):
        self.arch_ids = data_dict["arch_ids"]
        self.arch = data_dict["arch"]
        self.metrics = data_dict["metrics"]

    def __len__(self):
        return len(self.arch_ids)

    def __getitem__(self, idx):
        return {
            'arch': self.arch[idx],
            'metrics': self.metrics[idx],
            'arch_id': self.arch_ids[idx],
        }


def nas_collate_fn(batch):
    return {
        'arch': Batch.from_data_list([item['arch'] for item in batch]),
        'metrics': torch.stack([item['metrics'] for item in batch]),
        'arch_ids': torch.stack([item['arch_id'] for item in batch])
    }


def get_nas_dataloader(dataset_path, batch_sizes=(32, 64), ratios=(0.8, 0.2)):

    dataset_dict = torch.load(dataset_path)
    metric_names = dataset_dict["metric_names"]
    full_dataset = NASDataset(dataset_dict)

    train_set, test_set = split_nas_dataset(full_dataset, ratios)

    loaders = []
    for subset, batch_size in zip([train_set, test_set], batch_sizes):
        loaders.append(DataLoader(
            subset,
            batch_size=batch_size,
            collate_fn=nas_collate_fn,
            shuffle=(subset == train_set),
            num_workers=0,
            pin_memory=True
        ))
    return tuple(loaders), metric_names


def split_nas_dataset(dataset, ratios=(0.8, 0.2), seed=42, stratify_col=3):

    metrics = dataset.metrics.numpy()
    bins = np.quantile(metrics[:, stratify_col], np.linspace(0, 1, 11))
    strat_labels = np.digitize(metrics[:, stratify_col], bins[:-1])
    idx = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=ratios[1],
        stratify=strat_labels,
        random_state=seed
    )

    return Subset(dataset, train_idx), Subset(dataset, test_idx)