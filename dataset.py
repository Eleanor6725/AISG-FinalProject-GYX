import os
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


TensorTuple = Tuple[torch.Tensor, ...]


def _infer_color_attr_from_data(data: torch.Tensor) -> torch.Tensor:
    """Infer color attribute for ColoredMNIST tensors.

    Convention: attr=0 for red images, attr=1 for green images.
    This fallback is used for old .pt files that do not explicitly store
    color/group metadata.
    """
    if data.dim() == 2:
        # Flattened ColoredMNIST: [N, 3*28*28]
        channels = data.view(data.size(0), 3, -1)
        red_score = channels[:, 0, :].mean(dim=1)
        green_score = channels[:, 1, :].mean(dim=1)
    elif data.dim() == 4:
        # Image tensor: [N, C, H, W]
        red_score = data[:, 0].mean(dim=(1, 2))
        green_score = data[:, 1].mean(dim=(1, 2))
    else:
        raise ValueError(
            f"Cannot infer ColoredMNIST color attribute from data shape {tuple(data.shape)}"
        )
    return (green_score > red_score).long()


def normalize_tensor_tuple(tensors: Sequence[torch.Tensor], default_env: int = 0) -> TensorTuple:
    """Normalize stored dataset tensors to (data, label, env, group, attr).

    New files should contain five fields. For compatibility with the original
    homework framework, old formats are also supported:
      - train/val: (data, label, env)
      - test:      (data, label)
    In old formats, attr is inferred from image color and group=2*label+attr.
    """
    tensors = tuple(tensors)
    if len(tensors) == 5:
        data, label, env, group, attr = tensors
    elif len(tensors) == 3:
        data, label, env = tensors
        attr = _infer_color_attr_from_data(data)
        group = 2 * label.long() + attr.long()
    elif len(tensors) == 2:
        data, label = tensors
        env = torch.full_like(label.long(), fill_value=default_env)
        attr = _infer_color_attr_from_data(data)
        group = 2 * label.long() + attr.long()
    else:
        raise ValueError(
            "Expected a dataset tuple of length 2, 3, or 5; "
            f"got length {len(tensors)}."
        )

    label = label.long()
    env = env.long()
    group = group.long()
    attr = attr.long()

    expected_group = 2 * label + attr
    bad_group = (group != expected_group).sum().item()
    if bad_group != 0:
        raise ValueError(
            f"Invalid group metadata: {bad_group} samples do not satisfy group=2*label+attr."
        )
    if not torch.all((label == 0) | (label == 1)):
        raise ValueError('ColoredMNIST expects binary labels 0/1.')
    if not torch.all((attr == 0) | (attr == 1)):
        raise ValueError('ColoredMNIST expects binary attr values 0/1.')

    return data, label, env, group, attr


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def summarize_tensors(name: str, tensors: Sequence[torch.Tensor], max_groups: int = 8):
    """Print a compact sanity-check summary for one split.

    This output is intentionally report-friendly: it shows whether the y-attr
    groups are present, whether the shortcut correlation changes by split/env,
    and whether the metadata is internally consistent.
    """
    data, label, env, group, attr = normalize_tensor_tuple(tensors)
    label_np = _to_numpy(label)
    env_np = _to_numpy(env)
    group_np = _to_numpy(group)
    attr_np = _to_numpy(attr)

    print('-' * 80)
    print(f'Dataset diagnostics: {name}')
    print(f'n={len(label_np)}, data_shape={tuple(data.shape)}')

    group_counts = {int(g): int((group_np == g).sum()) for g in sorted(np.unique(group_np).tolist())}
    env_counts = {int(e): int((env_np == e).sum()) for e in sorted(np.unique(env_np).tolist())}
    attr_counts = {int(a): int((attr_np == a).sum()) for a in sorted(np.unique(attr_np).tolist())}
    label_counts = {int(y): int((label_np == y).sum()) for y in sorted(np.unique(label_np).tolist())}
    print(f'label_counts={label_counts}')
    print(f'attr_counts={attr_counts}  # attr=0 red, attr=1 green')
    print(f'env_counts={env_counts}')
    print(f'group_counts={group_counts}  # group=2*y+attr')

    # y-attr cross table. This is the most direct way to inspect shortcut direction.
    print('y-attr cross table counts:')
    for y in [0, 1]:
        row = []
        y_mask = label_np == y
        denom = int(y_mask.sum())
        for a in [0, 1]:
            cnt = int(((label_np == y) & (attr_np == a)).sum())
            pct = cnt / denom if denom > 0 else float('nan')
            row.append(f'attr={a}: {cnt} ({pct:.3f})')
        print(f'  y={y}: ' + ', '.join(row))

    print('per-env y-attr shortcut direction:')
    for e in sorted(np.unique(env_np).tolist()):
        e_mask = env_np == e
        total = int(e_mask.sum())
        if total == 0:
            continue
        # Since attr equals y in the original train shortcut and attr != y in the reversed shortcut,
        # P(attr==y) gives a compact summary of shortcut direction.
        p_match = float((attr_np[e_mask] == label_np[e_mask]).mean())
        print(f'  env={int(e)}: n={total}, P(attr==y)={p_match:.3f}, P(attr!=y)={1-p_match:.3f}')

    if len(np.unique(group_np)) > max_groups:
        print(f'Warning: found more than {max_groups} groups: {sorted(np.unique(group_np).tolist())}')


class TensorLoader(object):
    """Dataloader wrapper used by the ColoredMNIST experiments.

    Every split is normalized to:
        data, label, env, group, attr

    The training split additionally exposes a stable sample id:
        data, label, env, group, attr, ids
    """

    def __init__(self, batch_size, path, split, workers, data_tensors: Optional[Sequence[TensorTuple]] = None):
        self.path = path
        self.split = split

        if data_tensors is None:
            train_tensors = torch.load(os.path.join(path, 'train.pt'))
        else:
            train_tensors = data_tensors[0]
        data, label, env, group, attr = normalize_tensor_tuple(train_tensors, default_env=0)
        ids = torch.arange(len(label), dtype=torch.long)
        self.training_dataset = TensorDataset(data, label, env, group, attr, ids)

        if data_tensors is None:
            val_tensors = torch.load(os.path.join(path, 'val.pt'))
        else:
            val_tensors = data_tensors[1]
        self.validation_dataset = TensorDataset(*normalize_tensor_tuple(val_tensors, default_env=0))

        self.test_dataset = {}
        for i, split_name in enumerate(split):
            if data_tensors is None:
                test_tensors = torch.load(os.path.join(path, f'{split_name}.pt'))
            else:
                test_tensors = data_tensors[i + 2]
            self.test_dataset[split_name] = TensorDataset(*normalize_tensor_tuple(test_tensors, default_env=2))

        self._training_loader = DataLoader(
            dataset=self.training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )
        self._training_loader_sequential = DataLoader(
            dataset=self.training_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        self._validation_loader = DataLoader(
            dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        self._test_loader = {}
        for key in self.test_dataset.keys():
            self._test_loader[key] = DataLoader(
                dataset=self.test_dataset[key],
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
                drop_last=False,
            )

    @property
    def training_loader(self):
        return self._training_loader

    @property
    def training_loader_sequential(self):
        return self._training_loader_sequential

    @property
    def validation_loader(self):
        return self._validation_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def feature_dim(self):
        return len(self.training_dataset[0][0])

    @property
    def n_envs(self):
        env = self.training_dataset.tensors[2]
        return int(env.max().item()) + 1

    @property
    def n_groups(self):
        group = self.training_dataset.tensors[3]
        return int(group.max().item()) + 1

    def print_diagnostics(self):
        summarize_tensors('train', self.training_dataset.tensors[:5])
        summarize_tensors('val', self.validation_dataset.tensors)
        for split_name, dataset in self.test_dataset.items():
            summarize_tensors(split_name, dataset.tensors)
        print('-' * 80)
