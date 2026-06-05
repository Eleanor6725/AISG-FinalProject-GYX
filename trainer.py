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



@trainer_register.register
class JTT(ERM):
    """Just Train Twice (JTT) trainer.

    Stage 1 trains an ordinary ERM model for a short time. Then we collect the
    training samples that the stage-1 model misclassifies. Stage 2 reinitializes
    the model and trains weighted ERM, where stage-1 error samples receive a
    larger weight.

    The implementation uses per-sample loss weights instead of a weighted
    sampler. This is simpler and works with the stable sample ids exposed by
    TensorLoader.
    """

    method_name = 'JTT'

    def _reset_model_and_optimizer(self, model_init='default'):
        reset_parameters(self._model, model_init)
        self._optimizer.state.clear()
        self._optimizer.zero_grad()

    @staticmethod
    def _print_error_breakdown(stats, name):
        errors = stats['errors'].astype(bool)
        n = len(errors)
        print(f'{name} error rate: {errors.mean():.6f} ({errors.sum()}/{n})')

        for key in ['envs', 'groups', 'attrs']:
            values = stats[key]
            label = key[:-1] if key.endswith('s') else key
            for value in sorted(np.unique(values).tolist()):
                mask = values == value
                err_cnt = int(errors[mask].sum())
                total = int(mask.sum())
                err_rate = err_cnt / total if total > 0 else float('nan')
                print(f'{name} {label}={int(value)}: error_rate={err_rate:.6f}, errors={err_cnt}, total={total}')

    @staticmethod
    def _validate_stage1_stats(stats, n_train):
        ids = stats['ids'].astype(np.int64)
        if ids.min() < 0 or ids.max() >= n_train:
            raise ValueError('Sample ids are out of range; cannot build weights safely.')
        if len(np.unique(ids)) != n_train:
            raise ValueError('Expected one unique id for each training sample.')
        for key in ['labels', 'groups', 'attrs', 'losses', 'errors']:
            if len(stats[key]) != n_train:
                raise ValueError(f'Stage-1 stats field {key} has wrong length: {len(stats[key])} != {n_train}')
        return ids

    def _normalize_weights(self, weights, normalize=True):
        weights = weights.astype(np.float32)
        if normalize:
            mean_weight = float(weights.mean())
            if mean_weight > 0:
                weights = weights / mean_weight
        return weights

    def _print_weight_breakdown(self, stats, weights, method_name='weights'):
        ids = stats['ids'].astype(np.int64)
        labels = stats['labels'].astype(np.int64)
        attrs = stats['attrs'].astype(np.int64)
        groups = stats['groups'].astype(np.int64)
        errors = stats['errors'].astype(bool)
        losses = stats['losses']

        print(f'{method_name} weight summary: min={weights.min():.6f}, max={weights.max():.6f}, '
              f'mean={weights.mean():.6f}, std={weights.std():.6f}')

        for group_id in sorted(np.unique(groups).tolist()):
            mask = groups == group_id
            ids_g = ids[mask]
            print(
                f'{method_name} group={int(group_id)}: '
                f'count={int(mask.sum())}, errors={int(errors[mask].sum())}, '
                f'error_rate={float(errors[mask].mean()):.6f}, '
                f'mean_stage1_loss={float(losses[mask].mean()):.6f}, '
                f'mean_weight={float(weights[ids_g].mean()):.6f}, '
                f'total_weight={float(weights[ids_g].sum()):.6f}'
            )

        shortcut_match = attrs == labels
        for flag, name in [(True, 'attr==y'), (False, 'attr!=y')]:
            mask = shortcut_match == flag
            if mask.sum() == 0:
                continue
            ids_s = ids[mask]
            print(
                f'{method_name} shortcut subset {name}: count={int(mask.sum())}, '
                f'error_rate={float(errors[mask].mean()):.6f}, '
                f'mean_stage1_loss={float(losses[mask].mean()):.6f}, '
                f'mean_weight={float(weights[ids_s].mean()):.6f}, '
                f'total_weight={float(weights[ids_s].sum()):.6f}'
            )

        # Weighted y-attr table is the most important diagnostic for the proposed method.
        print(f'{method_name} weighted y-attr table:')
        for y in [0, 1]:
            row = []
            y_mask = labels == y
            denom = float(weights[ids[y_mask]].sum()) if y_mask.sum() > 0 else 0.0
            for a in [0, 1]:
                mask = (labels == y) & (attrs == a)
                mass = float(weights[ids[mask]].sum()) if mask.sum() > 0 else 0.0
                pct = mass / denom if denom > 0 else float('nan')
                row.append(f'attr={a}: mass={mass:.3f} ({pct:.3f})')
            print(f'  y={y}: ' + ', '.join(row))
        total_mass = float(weights[ids].sum())
        shortcut_mass = float(weights[ids[shortcut_match]].sum())
        print(
            f'{method_name} weighted shortcut direction: '
            f'P_w(attr==y)={shortcut_mass/total_mass:.6f}, '
            f'P_w(attr!=y)={1-shortcut_mass/total_mass:.6f}'
        )

    def _build_jtt_weights(self, stats, up_weight, normalize=True):
        n_train = len(self._dataset.training_dataset)
        weights = np.ones(n_train, dtype=np.float32)
        ids = self._validate_stage1_stats(stats, n_train)

        error_ids = ids[stats['errors'].astype(bool)]
        weights[error_ids] = float(up_weight)

        raw_weights = weights.copy()
        weights = self._normalize_weights(weights, normalize=normalize)

        print(f'JTT up_weight={up_weight}, normalize={normalize}')
        print(f'JTT raw emphasized samples: {len(error_ids)}/{n_train}')
        print(f'JTT raw weight mean={raw_weights.mean():.6f}, normalized weight mean={weights.mean():.6f}')
        self._print_weight_breakdown(stats, weights, method_name='JTT')
        return torch.tensor(weights, dtype=torch.float32)

    def _run_stage1_and_collect_stats(self, stage1_steps, logging_steps, eval_steps, metrics, kwargs):
        print('=' * 80)
        print(f'{self.method_name} Stage 1: ERM warm-up for {stage1_steps} updates')
        print('=' * 80)
        self._loss_fn.update_weight(None)
        super().train(stage1_steps, logging_steps, eval_steps, metrics, start_step=0, **kwargs)

        stats = self.get_training_sample_statistics()
        self._print_error_breakdown(stats, name=f'{self.method_name} Stage 1')

        if bool(kwargs.get('jtt_eval_stage1', True)):
            print('=' * 80)
            print(f'{self.method_name} Stage 1 validation/test diagnostics before reinitialization')
            print('=' * 80)
            val_loss, val_metrics = self.evaluate(self._dataset.validation_loader, metrics)
            print(f'{self.method_name} Stage 1 validation loss:', val_loss)
            print(f'{self.method_name} Stage 1 validation metrics:', val_metrics)
            for split_name, loader in self._dataset.test_loader.items():
                stage1_metrics = self.evaluate(loader, metrics, return_loss=False)
                print(f'{self.method_name} Stage 1 {split_name} metrics:', stage1_metrics)
        return stats

    def _train_stage2(self, num_training_updates, logging_steps, eval_steps, metrics, kwargs, sample_weights, consistency_lambda=0.0):
        model_init = kwargs.get('model_init', 'default')
        print('=' * 80)
        print(f'{self.method_name} Stage 2: reinitialize and train weighted ERM for {num_training_updates} updates')
        if consistency_lambda > 0:
            print(f'{self.method_name} Stage 2 uses class-conditional attr consistency, lambda={consistency_lambda}')
        print('=' * 80)

        self._reset_model_and_optimizer(model_init=model_init)
        self._loss_fn.update_weight(sample_weights)

        if consistency_lambda > 0:
            self._train_weighted_with_consistency(
                num_training_updates, logging_steps, eval_steps, metrics,
                consistency_lambda=consistency_lambda, **kwargs
            )
        else:
            super().train(num_training_updates, logging_steps, eval_steps, metrics, start_step=0, **kwargs)

        self._loss_fn.update_weight(None)
        return None

    def _class_conditional_attr_consistency(self, logits, labels, attrs, ids=None):
        """Penalize different average logits for attr=0/1 within the same label.

        This is deliberately simple and batch-local. For each class y, if both
        attr values are present in the batch, we penalize
            (mean_logit[y, attr=0] - mean_logit[y, attr=1])^2.
        It discourages the model from using shortcut attr to systematically
        shift predictions among samples sharing the same class label.
        """
        penalty = logits.new_tensor(0.0)
        count = 0
        for y in [0, 1]:
            mask_y = labels == y
            mask_a0 = mask_y & (attrs == 0)
            mask_a1 = mask_y & (attrs == 1)
            if mask_a0.sum() > 0 and mask_a1.sum() > 0:
                penalty = penalty + (logits[mask_a0].mean() - logits[mask_a1].mean()).pow(2)
                count += 1
        if count > 0:
            penalty = penalty / count
        return penalty

    def _train_weighted_with_consistency(self, num_training_updates, logging_steps, eval_steps, metrics=None,
                                         start_step=0, consistency_lambda=0.0, **kwargs):
        if metrics is None:
            metrics = []
        iterator = iter(cycle(self._dataset.training_loader))
        for i in tqdm(range(start_step, start_step + num_training_updates), desc=f'{self.method_name} Stage2'):
            self._model.train()
            input_batch, label_batch, env_batch, group_batch, attr_batch, ids = unpack_batch(
                next(iterator), device=self._device
            )

            predict = self._model(input_batch).squeeze()
            base_loss = self._loss_fn(
                predict, label_batch, env_batch,
                group=group_batch, attr=attr_batch, ids=ids, reduction='mean'
            )
            consistency = self._class_conditional_attr_consistency(predict, label_batch, attr_batch, ids=ids)
            total_loss = base_loss + consistency_lambda * consistency

            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

            if eval_steps > 0 and (i + 1) % eval_steps == 0:
                val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
                print(f'train loss:{base_loss} consistency:{consistency} val loss:{val_loss}')
                print('Validation metrics:', metric_dict)

            if logging_steps > 0 and (i + 1) % logging_steps == 0:
                print(f'train loss:{base_loss} consistency:{consistency} total_loss:{total_loss}')
        return None

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=None, start_step=0, **kwargs):
        if metrics is None:
            metrics = []

        stage1_steps = int(kwargs.get('jtt_stage1_steps', 100))
        up_weight = float(kwargs.get('jtt_up_weight', 20.0))
        normalize_weights = bool(kwargs.get('jtt_normalize_weights', True))

        if stage1_steps <= 0:
            raise ValueError('JTT requires --jtt_stage1_steps > 0.')
        if up_weight < 1:
            raise ValueError('JTT expects --jtt_up_weight >= 1.')

        stats = self._run_stage1_and_collect_stats(stage1_steps, logging_steps, eval_steps, metrics, kwargs)
        sample_weights = self._build_jtt_weights(stats, up_weight=up_weight, normalize=normalize_weights)
        return self._train_stage2(num_training_updates, logging_steps, eval_steps, metrics, kwargs, sample_weights)


@trainer_register.register
class SoftJTT(JTT):
    """Soft-JTT ablation.

    This keeps JTT's two-stage structure, but replaces the hard 0/1 error set
    with a continuous difficulty score from the stage-1 per-sample loss:
        w_i = 1 + alpha * loss_i / mean(loss).
    """

    method_name = 'SoftJTT'

    def _build_soft_weights(self, stats, alpha=1.0, normalize=True):
        n_train = len(self._dataset.training_dataset)
        weights = np.ones(n_train, dtype=np.float32)
        ids = self._validate_stage1_stats(stats, n_train)

        losses = stats['losses'].astype(np.float32)
        mean_loss = float(losses.mean())
        if mean_loss <= 0:
            raise ValueError('Cannot build SoftJTT weights because stage-1 mean loss is non-positive.')
        weights[ids] = 1.0 + float(alpha) * (losses / mean_loss)
        weights = self._normalize_weights(weights, normalize=normalize)

        print(f'SoftJTT alpha={alpha}, normalize={normalize}, stage1_mean_loss={mean_loss:.6f}')
        self._print_weight_breakdown(stats, weights, method_name='SoftJTT')
        return torch.tensor(weights, dtype=torch.float32)

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=None, start_step=0, **kwargs):
        if metrics is None:
            metrics = []

        stage1_steps = int(kwargs.get('jtt_stage1_steps', 100))
        alpha = float(kwargs.get('softjtt_alpha', 1.0))
        normalize_weights = bool(kwargs.get('jtt_normalize_weights', True))

        if stage1_steps <= 0:
            raise ValueError('SoftJTT requires --jtt_stage1_steps > 0.')
        if alpha < 0:
            raise ValueError('SoftJTT expects --softjtt_alpha >= 0.')

        stats = self._run_stage1_and_collect_stats(stage1_steps, logging_steps, eval_steps, metrics, kwargs)
        sample_weights = self._build_soft_weights(stats, alpha=alpha, normalize=normalize_weights)
        return self._train_stage2(num_training_updates, logging_steps, eval_steps, metrics, kwargs, sample_weights)


@trainer_register.register
class NeutralizedSoftJTT(SoftJTT):
    """Shortcut-neutralized Soft-JTT ablation.

    After building soft difficulty weights, this class rebalances the *weighted*
    mass of the four y-attr groups. This is the key shortcut-neutralization step:
    it prevents stage-2 training from having a strong attr==y or attr!=y shortcut.
    """

    method_name = 'NeutralizedSoftJTT'

    def _neutralize_group_mass(self, stats, weights, power=1.0, eps=1e-12):
        ids = stats['ids'].astype(np.int64)
        groups = stats['groups'].astype(np.int64)
        unique_groups = sorted(np.unique(groups).tolist())
        masses = {}
        for g in unique_groups:
            mask = groups == g
            masses[g] = float(weights[ids[mask]].sum())

        target = sum(masses.values()) / len(unique_groups)
        factors = {}
        neutralized = weights.copy()
        for g in unique_groups:
            factor = (target / (masses[g] + eps)) ** float(power)
            factors[g] = factor
            mask = groups == g
            neutralized[ids[mask]] *= factor

        print(f'NeutralizedSoftJTT group-neutralization power={power}')
        print(f'NeutralizedSoftJTT pre-neutralization masses={{{", ".join([str(int(g))+": "+format(masses[g], ".3f") for g in unique_groups])}}}')
        print(f'NeutralizedSoftJTT target_mass_per_group={target:.6f}')
        print(f'NeutralizedSoftJTT correction_factors={{{", ".join([str(int(g))+": "+format(factors[g], ".6f") for g in unique_groups])}}}')
        return neutralized

    def _build_neutralized_soft_weights(self, stats, alpha=1.0, neutralize_power=1.0, normalize=True):
        n_train = len(self._dataset.training_dataset)
        ids = self._validate_stage1_stats(stats, n_train)
        losses = stats['losses'].astype(np.float32)
        mean_loss = float(losses.mean())
        if mean_loss <= 0:
            raise ValueError('Cannot build neutralized weights because stage-1 mean loss is non-positive.')

        weights = np.ones(n_train, dtype=np.float32)
        weights[ids] = 1.0 + float(alpha) * (losses / mean_loss)
        print(f'NeutralizedSoftJTT alpha={alpha}, stage1_mean_loss={mean_loss:.6f}')
        self._print_weight_breakdown(stats, self._normalize_weights(weights.copy(), normalize=True),
                                     method_name='NeutralizedSoftJTT before neutralization')

        weights = self._neutralize_group_mass(stats, weights, power=neutralize_power)
        weights = self._normalize_weights(weights, normalize=normalize)
        self._print_weight_breakdown(stats, weights, method_name='NeutralizedSoftJTT')
        return torch.tensor(weights, dtype=torch.float32)

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=None, start_step=0, **kwargs):
        if metrics is None:
            metrics = []

        stage1_steps = int(kwargs.get('jtt_stage1_steps', 100))
        alpha = float(kwargs.get('softjtt_alpha', 1.0))
        neutralize_power = float(kwargs.get('snc_neutralize_power', 1.0))
        normalize_weights = bool(kwargs.get('jtt_normalize_weights', True))

        if stage1_steps <= 0:
            raise ValueError('NeutralizedSoftJTT requires --jtt_stage1_steps > 0.')
        if alpha < 0:
            raise ValueError('NeutralizedSoftJTT expects --softjtt_alpha >= 0.')
        if neutralize_power < 0:
            raise ValueError('NeutralizedSoftJTT expects --snc_neutralize_power >= 0.')

        stats = self._run_stage1_and_collect_stats(stage1_steps, logging_steps, eval_steps, metrics, kwargs)
        sample_weights = self._build_neutralized_soft_weights(
            stats, alpha=alpha, neutralize_power=neutralize_power, normalize=normalize_weights
        )
        return self._train_stage2(num_training_updates, logging_steps, eval_steps, metrics, kwargs, sample_weights)


@trainer_register.register
class SNCJTT(NeutralizedSoftJTT):
    """Shortcut-Neutralized Consistency JTT, our proposed method.

    Ablation hierarchy:
      JTT                 : hard error upweighting
      SoftJTT             : continuous difficulty upweighting
      NeutralizedSoftJTT  : SoftJTT + weighted y-attr group mass neutralization
      SNCJTT / Ours       : NeutralizedSoftJTT + class-conditional attr consistency
    """

    method_name = 'SNCJTT'

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=None, start_step=0, **kwargs):
        if metrics is None:
            metrics = []

        stage1_steps = int(kwargs.get('jtt_stage1_steps', 100))
        alpha = float(kwargs.get('softjtt_alpha', 1.0))
        neutralize_power = float(kwargs.get('snc_neutralize_power', 1.0))
        normalize_weights = bool(kwargs.get('jtt_normalize_weights', True))
        consistency_lambda = float(kwargs.get('snc_consistency_lambda', 0.1))

        if stage1_steps <= 0:
            raise ValueError('SNCJTT requires --jtt_stage1_steps > 0.')
        if alpha < 0:
            raise ValueError('SNCJTT expects --softjtt_alpha >= 0.')
        if neutralize_power < 0:
            raise ValueError('SNCJTT expects --snc_neutralize_power >= 0.')
        if consistency_lambda < 0:
            raise ValueError('SNCJTT expects --snc_consistency_lambda >= 0.')

        stats = self._run_stage1_and_collect_stats(stage1_steps, logging_steps, eval_steps, metrics, kwargs)
        sample_weights = self._build_neutralized_soft_weights(
            stats, alpha=alpha, neutralize_power=neutralize_power, normalize=normalize_weights
        )
        return self._train_stage2(
            num_training_updates, logging_steps, eval_steps, metrics, kwargs, sample_weights,
            consistency_lambda=consistency_lambda
        )




@trainer_register.register
class DSSNCJTT(SNCJTT):
    """Difficulty-Stratified Shortcut-Neutralized Consistency JTT.

    This is the more distinctive final method. It keeps the two-stage JTT
    structure, but neutralizes the shortcut attribute inside label-conditioned
    difficulty strata rather than only balancing the four global y-attr groups.

    Stage-1 ERM gives a difficulty score d_i = loss_i / mean(loss). We split
    samples into K difficulty bins. In each stratum (label y, difficulty bin b),
    we balance the weighted mass of attr=0 and attr=1. The intuition is that a
    model can still exploit shortcuts within easy/hard subsets even when the
    global group masses look balanced.
    """

    method_name = 'DS-SNCJTT'

    def _make_difficulty_bins(self, stats, n_bins=3, mode='conditional'):
        """Assign stage-1 difficulty bins to training samples.

        mode='global' uses global loss quantiles. This was the first DS-SNCJTT
        implementation, but on ColoredMNIST it often creates strata such as
        (y=0, easy) that contain only attr=0 and no attr=1, so shortcut
        neutralization cannot be applied there.

        mode='conditional' is the default fixed version. It splits samples into
        equal-frequency bins *inside each (label, attr) cell*. Thus bin b means
        approximately the same relative difficulty percentile within each
        shortcut cell. This guarantees that, when enough samples exist, every
        (label, bin) stratum has both attrs and can be neutralized.
        """
        n_train = len(self._dataset.training_dataset)
        ids = self._validate_stage1_stats(stats, n_train)
        losses = stats['losses'].astype(np.float32)
        labels = stats['labels'].astype(np.int64)
        attrs = stats['attrs'].astype(np.int64)
        n_bins = int(n_bins)
        mode = str(mode).lower()
        if n_bins <= 0:
            raise ValueError('DS-SNCJTT expects --ds_num_bins >= 1.')
        if mode not in ['conditional', 'global']:
            raise ValueError("DS-SNCJTT expects --ds_bin_mode to be 'conditional' or 'global'.")
        if mode not in ['conditional', 'global']:
            raise ValueError("--ds_bin_mode must be either 'conditional' or 'global'.")

        bins_in_stats_order = np.zeros_like(losses, dtype=np.int64)
        edges_for_print = []

        if n_bins == 1:
            pass
        elif mode == 'global':
            qs = np.linspace(0, 1, n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(losses, qs))
            bins_in_stats_order = np.digitize(losses, edges, right=False).astype(np.int64)
            edges_for_print.append(('global', [float(x) for x in edges]))
        else:
            # Relative-difficulty bins within each (y, attr) cell.
            # Rank-based binning avoids empty bins caused by duplicate quantile edges.
            for y in sorted(np.unique(labels).tolist()):
                for a in sorted(np.unique(attrs).tolist()):
                    mask = (labels == y) & (attrs == a)
                    idx = np.where(mask)[0]
                    m = len(idx)
                    if m == 0:
                        continue
                    order = idx[np.argsort(losses[idx], kind='mergesort')]
                    if m < n_bins:
                        # Extremely small cells: create as many nonempty bins as possible.
                        local_bins = np.arange(m, dtype=np.int64)
                    else:
                        local_bins = (np.arange(m) * n_bins // m).astype(np.int64)
                        local_bins = np.minimum(local_bins, n_bins - 1)
                    bins_in_stats_order[order] = local_bins

                    # Print approximate loss ranges for interpretability.
                    ranges = []
                    for b in sorted(np.unique(local_bins).tolist()):
                        local_idx = order[local_bins == b]
                        ranges.append((int(b), float(losses[local_idx].min()), float(losses[local_idx].max()), int(len(local_idx))))
                    edges_for_print.append((f'y={int(y)},attr={int(a)}', ranges))

        bins_by_id = np.zeros(n_train, dtype=np.int64)
        bins_by_id[ids] = bins_in_stats_order

        print(
            f'DS-SNCJTT difficulty binning: mode={mode}, requested_bins={n_bins}, '
            f'actual_bins={int(bins_in_stats_order.max()) + 1}'
        )
        if mode == 'global' and len(edges_for_print) > 0:
            print('DS-SNCJTT global difficulty bin edges:', edges_for_print[0][1])
        elif mode == 'conditional':
            print('DS-SNCJTT conditional binning: bins are relative difficulty quantiles within each (label, attr) cell.')
            for name, ranges in edges_for_print:
                compact = ', '.join([f'b{b}:n={n},[{lo:.4f},{hi:.4f}]' for b, lo, hi, n in ranges])
                print(f'  {name}: {compact}')

        for b in sorted(np.unique(bins_in_stats_order).tolist()):
            mask = bins_in_stats_order == b
            print(
                f'DS-SNCJTT bin={int(b)}: count={int(mask.sum())}, '
                f'loss_mean={float(losses[mask].mean()):.6f}, '
                f'loss_min={float(losses[mask].min()):.6f}, '
                f'loss_max={float(losses[mask].max()):.6f}'
            )
        return bins_by_id

    def _print_label_difficulty_attr_table(self, stats, weights, bins_by_id, method_name='DS-SNCJTT'):
        ids = stats['ids'].astype(np.int64)
        labels = stats['labels'].astype(np.int64)
        attrs = stats['attrs'].astype(np.int64)
        bins = bins_by_id[ids].astype(np.int64)
        print(f'{method_name} weighted label-difficulty-attr table:')
        for y in sorted(np.unique(labels).tolist()):
            for b in sorted(np.unique(bins).tolist()):
                mask_yb = (labels == y) & (bins == b)
                if mask_yb.sum() == 0:
                    continue
                denom = float(weights[ids[mask_yb]].sum())
                parts = []
                for a in [0, 1]:
                    mask = mask_yb & (attrs == a)
                    mass = float(weights[ids[mask]].sum()) if mask.sum() > 0 else 0.0
                    pct = mass / denom if denom > 0 else float('nan')
                    parts.append(f'attr={a}: mass={mass:.3f} ({pct:.3f}), count={int(mask.sum())}')
                print(f'  y={int(y)}, bin={int(b)}: ' + ', '.join(parts))

    def _neutralize_label_difficulty_attr_mass(self, stats, weights, bins_by_id, power=1.0, eps=1e-12):
        ids = stats['ids'].astype(np.int64)
        labels = stats['labels'].astype(np.int64)
        attrs = stats['attrs'].astype(np.int64)
        bins = bins_by_id[ids].astype(np.int64)
        neutralized = weights.copy()
        power = float(power)

        print(f'DS-SNCJTT label-difficulty shortcut-neutralization power={power}')
        for y in sorted(np.unique(labels).tolist()):
            for b in sorted(np.unique(bins).tolist()):
                base = (labels == y) & (bins == b)
                if base.sum() == 0:
                    continue
                masses = {}
                counts = {}
                for a in [0, 1]:
                    mask = base & (attrs == a)
                    counts[a] = int(mask.sum())
                    masses[a] = float(weights[ids[mask]].sum()) if counts[a] > 0 else 0.0

                # We cannot synthesize a missing attr side. If either side is
                # absent, leave this stratum unchanged and print a warning.
                if counts[0] == 0 or counts[1] == 0:
                    print(
                        f'  y={int(y)}, bin={int(b)}: skip neutralization because one attr side is missing; '
                        f'counts={counts}, masses={masses}'
                    )
                    continue

                target = (masses[0] + masses[1]) / 2.0
                factors = {a: (target / (masses[a] + eps)) ** power for a in [0, 1]}
                for a in [0, 1]:
                    mask = base & (attrs == a)
                    neutralized[ids[mask]] *= factors[a]

                print(
                    f'  y={int(y)}, bin={int(b)}: masses={{0: {masses[0]:.3f}, 1: {masses[1]:.3f}}}, '
                    f'target={target:.3f}, factors={{0: {factors[0]:.6f}, 1: {factors[1]:.6f}}}'
                )
        return neutralized

    def _build_ds_snc_weights(self, stats, alpha=1.0, neutralize_power=1.0, n_bins=3, normalize=True, bin_mode='conditional'):
        n_train = len(self._dataset.training_dataset)
        ids = self._validate_stage1_stats(stats, n_train)
        losses = stats['losses'].astype(np.float32)
        mean_loss = float(losses.mean())
        if mean_loss <= 0:
            raise ValueError('Cannot build DS-SNCJTT weights because stage-1 mean loss is non-positive.')

        weights = np.ones(n_train, dtype=np.float32)
        weights[ids] = 1.0 + float(alpha) * (losses / mean_loss)
        bins_by_id = self._make_difficulty_bins(stats, n_bins=n_bins, mode=bin_mode)

        print(f'DS-SNCJTT alpha={alpha}, stage1_mean_loss={mean_loss:.6f}')
        self._print_weight_breakdown(stats, self._normalize_weights(weights.copy(), normalize=True),
                                     method_name='DS-SNCJTT before stratified neutralization')
        self._print_label_difficulty_attr_table(
            stats, self._normalize_weights(weights.copy(), normalize=True), bins_by_id,
            method_name='DS-SNCJTT before stratified neutralization'
        )

        weights = self._neutralize_label_difficulty_attr_mass(
            stats, weights, bins_by_id, power=neutralize_power
        )
        weights = self._normalize_weights(weights, normalize=normalize)

        self._print_weight_breakdown(stats, weights, method_name='DS-SNCJTT')
        self._print_label_difficulty_attr_table(stats, weights, bins_by_id, method_name='DS-SNCJTT')
        return torch.tensor(weights, dtype=torch.float32), torch.tensor(bins_by_id, dtype=torch.long)

    def _class_difficulty_conditional_attr_consistency(self, logits, labels, attrs, difficulty_bins):
        """Attr consistency inside each (label, difficulty-bin) stratum.

        Compared with SNCJTT's class-conditional consistency, this prevents the
        model from using shortcut attributes differently on easy/medium/hard
        examples. The penalty is batch-local and skipped for strata where one
        attr side is absent in the current batch.
        """
        penalty = logits.new_tensor(0.0)
        count = 0
        for y in [0, 1]:
            mask_y = labels == y
            for b in difficulty_bins.unique():
                mask_yb = mask_y & (difficulty_bins == b)
                mask_a0 = mask_yb & (attrs == 0)
                mask_a1 = mask_yb & (attrs == 1)
                if mask_a0.sum() > 0 and mask_a1.sum() > 0:
                    penalty = penalty + (logits[mask_a0].mean() - logits[mask_a1].mean()).pow(2)
                    count += 1
        if count > 0:
            penalty = penalty / count
        return penalty

    def _train_weighted_with_ds_consistency(self, num_training_updates, logging_steps, eval_steps, metrics=None,
                                            start_step=0, consistency_lambda=0.0, difficulty_bins_by_id=None,
                                            **kwargs):
        if metrics is None:
            metrics = []
        if difficulty_bins_by_id is None:
            raise ValueError('DS-SNCJTT requires difficulty_bins_by_id for stratified consistency.')
        difficulty_bins_by_id = difficulty_bins_by_id.to(self._device)

        iterator = iter(cycle(self._dataset.training_loader))
        for i in tqdm(range(start_step, start_step + num_training_updates), desc=f'{self.method_name} Stage2'):
            self._model.train()
            input_batch, label_batch, env_batch, group_batch, attr_batch, ids = unpack_batch(
                next(iterator), device=self._device
            )

            predict = self._model(input_batch).squeeze()
            base_loss = self._loss_fn(
                predict, label_batch, env_batch,
                group=group_batch, attr=attr_batch, ids=ids, reduction='mean'
            )
            batch_bins = difficulty_bins_by_id[ids]
            consistency = self._class_difficulty_conditional_attr_consistency(
                predict, label_batch, attr_batch, batch_bins
            )
            total_loss = base_loss + float(consistency_lambda) * consistency

            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

            if eval_steps > 0 and (i + 1) % eval_steps == 0:
                val_loss, metric_dict = self.evaluate(self._dataset.validation_loader, metrics)
                print(f'train loss:{base_loss} ds_consistency:{consistency} val loss:{val_loss}')
                print('Validation metrics:', metric_dict)

            if logging_steps > 0 and (i + 1) % logging_steps == 0:
                print(f'train loss:{base_loss} ds_consistency:{consistency} total_loss:{total_loss}')
        return None

    def _train_stage2_ds(self, num_training_updates, logging_steps, eval_steps, metrics, kwargs,
                         sample_weights, difficulty_bins_by_id, consistency_lambda=0.0):
        model_init = kwargs.get('model_init', 'default')
        print('=' * 80)
        print(f'{self.method_name} Stage 2: reinitialize and train DS-weighted ERM for {num_training_updates} updates')
        if consistency_lambda > 0:
            print(f'{self.method_name} Stage 2 uses difficulty-stratified attr consistency, lambda={consistency_lambda}')
        print('=' * 80)

        self._reset_model_and_optimizer(model_init=model_init)
        self._loss_fn.update_weight(sample_weights)

        if consistency_lambda > 0:
            self._train_weighted_with_ds_consistency(
                num_training_updates, logging_steps, eval_steps, metrics,
                consistency_lambda=consistency_lambda,
                difficulty_bins_by_id=difficulty_bins_by_id,
                **kwargs
            )
        else:
            ERM.train(self, num_training_updates, logging_steps, eval_steps, metrics, start_step=0, **kwargs)

        self._loss_fn.update_weight(None)
        return None

    def train(self, num_training_updates, logging_steps, eval_steps, metrics=None, start_step=0, **kwargs):
        if metrics is None:
            metrics = []

        stage1_steps = int(kwargs.get('jtt_stage1_steps', 100))
        alpha = float(kwargs.get('softjtt_alpha', 1.0))
        neutralize_power = float(kwargs.get('snc_neutralize_power', 1.0))
        normalize_weights = bool(kwargs.get('jtt_normalize_weights', True))
        consistency_lambda = float(kwargs.get('snc_consistency_lambda', 0.1))
        n_bins = int(kwargs.get('ds_num_bins', 3))
        bin_mode = str(kwargs.get('ds_bin_mode', 'conditional')).lower()

        if stage1_steps <= 0:
            raise ValueError('DS-SNCJTT requires --jtt_stage1_steps > 0.')
        if alpha < 0:
            raise ValueError('DS-SNCJTT expects --softjtt_alpha >= 0.')
        if neutralize_power < 0:
            raise ValueError('DS-SNCJTT expects --snc_neutralize_power >= 0.')
        if consistency_lambda < 0:
            raise ValueError('DS-SNCJTT expects --snc_consistency_lambda >= 0.')
        if n_bins <= 0:
            raise ValueError('DS-SNCJTT expects --ds_num_bins >= 1.')
        if bin_mode not in ['conditional', 'global']:
            raise ValueError("DS-SNCJTT expects --ds_bin_mode to be 'conditional' or 'global'.")

        stats = self._run_stage1_and_collect_stats(stage1_steps, logging_steps, eval_steps, metrics, kwargs)
        sample_weights, difficulty_bins_by_id = self._build_ds_snc_weights(
            stats,
            alpha=alpha,
            neutralize_power=neutralize_power,
            n_bins=n_bins,
            normalize=normalize_weights,
            bin_mode=bin_mode,
        )
        return self._train_stage2_ds(
            num_training_updates, logging_steps, eval_steps, metrics, kwargs,
            sample_weights=sample_weights,
            difficulty_bins_by_id=difficulty_bins_by_id,
            consistency_lambda=consistency_lambda,
        )


# Friendly aliases used in the final project report.
trainer_register['Ours'] = SNCJTT
trainer_register['DS-SNCJTT'] = DSSNCJTT
trainer_register['DS_SNCJTT'] = DSSNCJTT
trainer_register['OursDS'] = DSSNCJTT
