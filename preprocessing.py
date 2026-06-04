# -*- coding: utf-8 -*-

# MIT License
# Copyright (c) 2019 Reiichiro Nakano
# Modifications for AISG final project scaffold:
#   - keep ColoredMNIST generation from the homework framework;
#   - optionally store env, color attribute, and y-attr group metadata;
#   - remain compatible with old .pt files via TensorLoader fallback.

import os

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green."""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([
            arr,
            np.zeros((h, w, 2), dtype=dtype),
        ], axis=2)
    else:
        arr = np.concatenate([
            np.zeros((h, w, 1), dtype=dtype),
            arr,
            np.zeros((h, w, 1), dtype=dtype),
        ], axis=2)
    return arr


def _tuple_has_project_metadata(path):
    try:
        tensors = torch.load(path)
        return isinstance(tensors, (tuple, list)) and len(tensors) >= 5
    except Exception:
        return False


class ColoredMNIST(datasets.VisionDataset):
    """Colored MNIST dataset generator.

    Stored project schema:
        data, label, env, group, attr

    where attr=0 denotes red, attr=1 denotes green, and
        group = 2 * label + attr.

    Note: If old homework .pt files already exist, we keep them by default and
    let `dataset.TensorLoader` infer attr/group. Use force_regenerate=True to
    overwrite them with the explicit five-field schema.
    """

    def __init__(self, root='./data', env='train1', transform=None, target_transform=None, force_regenerate=False):
        super(ColoredMNIST, self).__init__(root, transform=transform, target_transform=target_transform)
        self.prepare_colored_mnist(force_regenerate=force_regenerate)

        # The experiment uses TensorLoader on saved .pt files. These fields are
        # kept only for compatibility with the original VisionDataset interface.
        self.data_label_tuples = []
        if env in ['train1', 'train2', 'test', 'all_train']:
            return
        raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')

    def __getitem__(self, index):
        img, target = self.data_label_tuples[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def _image_to_tensor(self, image):
        if self.transform is not None:
            return self.transform(image).reshape(-1)
        return transforms.ToTensor()(image).reshape(-1)

    def prepare_colored_mnist(self, force_regenerate=False):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        train_path = os.path.join(colored_mnist_dir, 'train.pt')
        val_path = os.path.join(colored_mnist_dir, 'val.pt')
        test_path = os.path.join(colored_mnist_dir, 'test.pt')

        files_exist = os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)
        if files_exist and not force_regenerate:
            if all(_tuple_has_project_metadata(p) for p in [train_path, val_path, test_path]):
                print('Colored MNIST dataset with project metadata already exists')
            else:
                print('Colored MNIST dataset already exists in old format; TensorLoader will infer attr/group metadata')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)

        train_set = []
        test_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Binary label: 0 for digits 0-4, 1 for digits 5-9.
            binary_label = 0 if label < 5 else 1

            # Flip label with 20% probability.
            if np.random.uniform() < 0.2:
                binary_label = binary_label ^ 1

            # In this framework convention: attr=0 red, attr=1 green.
            # The original assignment makes color highly correlated with label
            # in train and anti-correlated in test.
            color_red = binary_label == 0

            # Flip the color with an environment-dependent probability.
            if idx < 20000:
                env_id = 0
                color_flip_p = 0.4
            elif idx < 40000:
                env_id = 1
                color_flip_p = 0.1
            else:
                env_id = 2
                color_flip_p = 0.9

            if np.random.uniform() < color_flip_p:
                color_red = not color_red

            attr = 0 if color_red else 1
            group = 2 * binary_label + attr
            colored_arr = color_grayscale_arr(im_array, red=color_red)
            tensor = self._image_to_tensor(Image.fromarray(colored_arr))

            if idx < 40000:
                train_set.append((tensor, binary_label, env_id, group, attr))
            else:
                test_set.append((tensor, binary_label, env_id, group, attr))

        train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=42)

        def stack_dataset(dataset):
            dataset = list(zip(*dataset))
            return [
                torch.stack(dataset[0]),
                torch.tensor(dataset[1], dtype=torch.long),
                torch.tensor(dataset[2], dtype=torch.long),
                torch.tensor(dataset[3], dtype=torch.long),
                torch.tensor(dataset[4], dtype=torch.long),
            ]

        train_set = stack_dataset(train_set)
        val_set = stack_dataset(val_set)
        test_set = stack_dataset(test_set)

        if not os.path.exists(colored_mnist_dir):
            os.makedirs(colored_mnist_dir)

        torch.save(train_set, train_path)
        torch.save(val_set, val_path)
        torch.save(test_set, test_path)


def plot_dataset_digits(dataset):
    fig = plt.figure(figsize=(13, 8))
    columns = 6
    rows = 3
    ax = []
    for i in range(columns * rows):
        img, label = dataset[i]
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title('Label: ' + str(label))
        plt.imshow(img)
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x):
        x = x.view(-1, 3 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x).flatten()
        return logits


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x).flatten()
        return logits
