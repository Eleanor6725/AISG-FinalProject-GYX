# import os
# import torch
# from torch.utils.data import DataLoader, TensorDataset


# class TensorLoader(object):

#     def __init__(self, batch_size, path, split, workers, data_tensors=None):
#         ''' @param data_tensors (train, val, test)
#         '''
#         if data_tensors is None:
#             data, label, env = torch.load(os.path.join(path, 'train.pt'))
#         else:
#             data, label, env = data_tensors[0]
#         ids = torch.arange(len(label))
#         self.training_dataset = TensorDataset(data, label, env, ids)
#         if data_tensors is None:
#             self.validation_dataset = TensorDataset(*torch.load(os.path.join(path, 'val.pt')))
#         else:
#             self.validation_dataset = TensorDataset(*data_tensors[1])
#         self.test_dataset = {}
#         for i, group in enumerate(split):
#             if data_tensors is None:
#                 self.test_dataset[group] = TensorDataset(*torch.load(os.path.join(path, f'{group}.pt')))
#             else:
#                 self.test_dataset[group] = TensorDataset(*data_tensors[i+2])
#         self._training_loader = DataLoader(
#             dataset=self.training_dataset,
#             batch_size=batch_size,
#             shuffle=True,
#             num_workers=workers,
#             pin_memory=True,
#             drop_last=False
#         )
#         self._training_loader_sequential = DataLoader(
#             dataset=self.training_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=workers,
#             pin_memory=True,
#             drop_last=False
#         )
        
#         self._validation_loader = DataLoader(
#             dataset=self.validation_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=workers,
#             pin_memory=True,
#             drop_last=False
#         )

#         self._test_loader = {}
#         for key in self.test_dataset.keys():
#             self._test_loader[key] =  DataLoader(
#                 dataset=self.test_dataset[key],
#                 batch_size=batch_size,
#                 shuffle=False,
#                 num_workers=workers,
#                 pin_memory=True,
#                 drop_last=False
#             )        

#     @property
#     def training_loader(self):
#         return self._training_loader
    
#     @property
#     def training_loader_sequential(self):
#         return self._training_loader_sequential

#     @property
#     def validation_loader(self):
#         return self._validation_loader
    
#     @property
#     def test_loader(self):
#         return self._test_loader
    
#     @property
#     def feature_dim(self):
#         return len(self.training_dataset[0][0])
    

import os
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


TensorTuple = Tuple[torch.Tensor, ...]


def _infer_color_attr_from_data(data: torch.Tensor) -> torch.Tensor:
    """Infer color attribute for ColoredMNIST tensors.

    We use the convention attr=0 for red images and attr=1 for green images.
    This fallback is only used for old .pt files that do not explicitly store
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
    # red if red channel has larger average activation; otherwise green.
    return (green_score > red_score).long()


def normalize_tensor_tuple(tensors: Sequence[torch.Tensor], default_env: int = 0) -> TensorTuple:
    """Normalize stored dataset tensors to (data, label, env, group, attr).

    New files should already contain these five fields. For compatibility with
    the original homework framework, we also support old formats:
      - train/val: (data, label, env)
      - test:      (data, label)
    In old formats, attr is inferred from the image color and group is defined as
    group = 2 * label + attr.
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

    return (
        data,
        label.long(),
        env.long(),
        group.long(),
        attr.long(),
    )


class TensorLoader(object):
    """Dataloader wrapper used by the ColoredMNIST experiments.

    Every split is normalized to the following schema:
        data, label, env, group, attr

    The training split additionally exposes a stable sample id:
        data, label, env, group, attr, ids

    This is the key change needed before implementing JTT: stage-1 errors can
    be mapped back to original training samples through `ids`.
    """

    def __init__(self, batch_size, path, split, workers, data_tensors: Optional[Sequence[TensorTuple]] = None):
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
            # ColoredMNIST test environment is conventionally env=2.
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
    def n_groups(self):
        return int(self.training_dataset.tensors[3].max().item()) + 1

    @property
    def n_envs(self):
        return int(self.training_dataset.tensors[2].max().item()) + 1

