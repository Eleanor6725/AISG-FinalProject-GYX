

import numpy as np
import torch
from tqdm import tqdm

from dataset import *
from metric import compute_metrics
from model import *
from register import Register


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def unpack_batch(bundle_batch, device=None):
    """Unpack both old and new dataset formats.

    New training batch: data, label, env, group, attr, ids
    New eval batch:     data, label, env, group, attr
    Old training batch: data, label, env, ids
    Old eval batch:     data, label, env  or data, label
    """
    input_batch = bundle_batch[0]
    label_batch = bundle_batch[1]

    env_batch = None
    group_batch = None
    attr_batch = None
    ids = None

    if len(bundle_batch) == 6:
        _, _, env_batch, group_batch, attr_batch, ids = bundle_batch
    elif len(bundle_batch) == 5:
        _, _, env_batch, group_batch, attr_batch = bundle_batch
    elif len(bundle_batch) == 4:
        _, _, env_batch, ids = bundle_batch
    elif len(bundle_batch) == 3:
        _, _, env_batch = bundle_batch
    elif len(bundle_batch) == 2:
        # Last-resort compatibility. Loss functions need an env id, so use zeros.
        env_batch = torch.zeros_like(label_batch, dtype=torch.long)
    else:
        raise ValueError(f'Unsupported batch length: {len(bundle_batch)}')

    if group_batch is None:
        group_batch = torch.zeros_like(label_batch, dtype=torch.long)
    if attr_batch is None:
        attr_batch = torch.zeros_like(label_batch, dtype=torch.long)

    if device is not None:
        input_batch = input_batch.to(device)
        label_batch = label_batch.to(device)
        env_batch = env_batch.to(device)
        group_batch = group_batch.to(device)
        attr_batch = attr_batch.to(device)
        if ids is not None:
            ids = ids.to(device)

    return input_batch, label_batch, env_batch, group_batch, attr_batch, ids


global trainer_register
trainer_register = Register('trainer_register')


@trainer_register.register
class ERM(object):
    def __init__(self, device, model, optimizer, dataset: TensorLoader, loss_fn: Loss, regularizer: Loss, reset_model=True, **kwargs):
        self._device = device
        self._model = model.to(self._device)
        self._optimizer = optimizer
        self._dataset = dataset
        self._loss_fn = loss_fn
        self._regularizer = regularizer
        self._reg_lambda = kwargs['reg_lambda']

        if reset_model:
            reset_parameters(self._model, kwargs['model_init'])

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=None, start_step=0, **kwargs):
        if metrics is None:
            metrics = []
        iterator = iter(cycle(self._dataset.training_loader))
        for i in tqdm(range(start_step, start_step + num_training_updates), desc='Training'):
            self._model.train()
            input_batch, label_batch, env_batch, group_batch, attr_batch, ids = unpack_batch(
                next(iterator), device=self._device
            )

            predict = self._model(input_batch).squeeze()
            loss = self._loss_fn(
                predict,
                label_batch,
                env_batch,
                group=group_batch,
                attr=attr_batch,
                ids=ids,
                reduction='mean',
            )
            if self._regularizer is not None:
                regularization = self._regularizer(
                    predict,
                    label_batch,
                    env_batch,
                    group=group_batch,
                    attr=attr_batch,
                    ids=ids,
                    network=self._model.head(),
                    risk=self._loss_fn,
                )
            else:
                regularization = 0
            total_loss = loss + self._reg_lambda * regularization

            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

            if eval_steps > 0 and (i + 1) % eval_steps == 0:
                val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
                print(f'train loss:{loss} val loss:{val_loss}')
                loss_inspect = self._loss_fn(
                    predict,
                    label_batch,
                    env_batch,
                    group=group_batch,
                    attr=attr_batch,
                    ids=ids,
                    reduction='none',
                )
                for env_id in env_batch.unique():
                    print(f'Loss of env {env_id}:', loss_inspect[env_batch == env_id].mean().detach().cpu().numpy())
                for group_id in group_batch.unique():
                    print(f'Loss of group {group_id}:', loss_inspect[group_batch == group_id].mean().detach().cpu().numpy())
                print('Validation metrics:', metric_dict)

            if logging_steps > 0 and (i + 1) % logging_steps == 0:
                print(f'train loss:{loss} regularization:{regularization}')

        return None

    def evaluate(self, dataloader, metrics=None, loss_reduction=True, return_loss=True):
        if metrics is None:
            metrics = []
        self._model.eval()
        sample = 0
        loss_list = []
        loss = 0
        predict_list = []
        label_list = []
        env_list = []
        group_list = []
        attr_list = []

        with torch.no_grad():
            for bundle_batch in tqdm(dataloader, desc='Evaluating'):
                input_batch, label_batch, env_batch, group_batch, attr_batch, ids = unpack_batch(
                    bundle_batch, device=self._device
                )
                batch_size = input_batch.shape[0]
                sample += batch_size

                predict = self._model(input_batch).squeeze()

                predict_list.append(predict.detach().cpu())
                label_list.append(label_batch.detach().cpu())
                env_list.append(env_batch.detach().cpu())
                group_list.append(group_batch.detach().cpu())
                attr_list.append(attr_batch.detach().cpu())

                if return_loss:
                    if loss_reduction:
                        batch_loss = self._loss_fn(
                            predict,
                            label_batch,
                            env_batch,
                            group=group_batch,
                            attr=attr_batch,
                            ids=ids,
                        )
                        loss += batch_loss * batch_size
                    else:
                        loss_list.append(self._loss_fn(
                            predict,
                            label_batch,
                            env_batch,
                            group=group_batch,
                            attr=attr_batch,
                            ids=ids,
                            reduction='none',
                        ).detach().cpu())

        predict_matrix = torch.cat(predict_list, dim=0).numpy()
        label_array = torch.cat(label_list, dim=0).numpy()
        env_array = torch.cat(env_list, dim=0).numpy()
        group_array = torch.cat(group_list, dim=0).numpy()
        attr_array = torch.cat(attr_list, dim=0).numpy()

        if return_loss:
            if loss_reduction:
                loss /= sample
            else:
                loss = torch.cat(loss_list)

        if len(metrics) > 0:
            metric_dict = compute_metrics(
                predict_matrix,
                label_array,
                metrics=metrics,
                group_matrix=group_array,
                attr_matrix=attr_array,
                env_matrix=env_array,
            )
            if return_loss:
                return loss, metric_dict
            else:
                return metric_dict
        else:
            return loss

    def get_training_sample_statistics(self):
        """Collect per-training-sample predictions/losses/errors.

        This utility is added specifically for the next JTT implementation.
        JTT stage 1 trains an ERM model, calls this function on the sequential
        train loader, then upweights samples with errors == 1 in stage 2.
        """
        self._model.eval()
        ids_list = []
        logits_list = []
        labels_list = []
        envs_list = []
        groups_list = []
        attrs_list = []
        losses_list = []

        with torch.no_grad():
            for bundle_batch in tqdm(self._dataset.training_loader_sequential, desc='Collecting train stats'):
                input_batch, label_batch, env_batch, group_batch, attr_batch, ids = unpack_batch(
                    bundle_batch, device=self._device
                )
                logits = self._model(input_batch).squeeze()
                losses = self._loss_fn(
                    logits,
                    label_batch,
                    env_batch,
                    group=group_batch,
                    attr=attr_batch,
                    ids=ids,
                    reduction='none',
                )

                ids_list.append(ids.detach().cpu())
                logits_list.append(logits.detach().cpu())
                labels_list.append(label_batch.detach().cpu())
                envs_list.append(env_batch.detach().cpu())
                groups_list.append(group_batch.detach().cpu())
                attrs_list.append(attr_batch.detach().cpu())
                losses_list.append(losses.detach().cpu())

        ids = torch.cat(ids_list).numpy()
        logits = torch.cat(logits_list).numpy()
        labels = torch.cat(labels_list).numpy()
        envs = torch.cat(envs_list).numpy()
        groups = torch.cat(groups_list).numpy()
        attrs = torch.cat(attrs_list).numpy()
        losses = torch.cat(losses_list).numpy()
        preds = (logits > 0).astype(np.int64)
        errors = (preds != labels).astype(np.int64)

        return {
            'ids': ids,
            'logits': logits,
            'labels': labels,
            'envs': envs,
            'groups': groups,
            'attrs': attrs,
            'losses': losses,
            'preds': preds,
            'errors': errors,
        }
