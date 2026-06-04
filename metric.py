import logging
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from register import Register


register_obj = Register('metric_register')


def _binary_predictions(predict_matrix):
    predict_result = np.zeros(len(predict_matrix), dtype=np.int64)
    predict_result[np.asarray(predict_matrix) > 0] = 1
    return predict_result


def _safe_accuracy(pred, target):
    if len(target) == 0:
        return np.nan
    return accuracy_score(target, pred)


def compute_metrics(
    predict_matrix,
    target_matrix,
    metrics,
    group_matrix=None,
    attr_matrix=None,
    env_matrix=None,
):
    """Compute metrics.

    For backward compatibility, ordinary metrics only need prediction/target.
    Group-aware metrics can additionally consume group/attribute/environment
    arrays. If a metric returns a dict, it is merged into the final metric dict.
    """
    metric_dict = {}
    for metric in metrics:
        if metric not in register_obj:
            logging.warning(f'metric:{metric} is not registered.')
            continue
        metric_class = register_obj[metric]
        metric_value = metric_class.compute(
            predict_matrix,
            target_matrix,
            group_matrix=group_matrix,
            attr_matrix=attr_matrix,
            env_matrix=env_matrix,
        )
        if isinstance(metric_value, dict):
            metric_dict.update(metric_value)
        else:
            metric_dict[metric] = metric_value
    return metric_dict


@register_obj.register
class Accuracy:
    @staticmethod
    def compute(predict_matrix, target_matrix, **kwargs):
        predict_result = _binary_predictions(predict_matrix)
        return accuracy_score(target_matrix, predict_result)


@register_obj.register
class AUC:
    @staticmethod
    def compute(predict_matrix, target_matrix, **kwargs):
        # roc_auc_score fails if only one class is present. That can happen in a
        # tiny debug split, so we return nan instead of crashing.
        try:
            return roc_auc_score(target_matrix, predict_matrix)
        except ValueError:
            return np.nan


@register_obj.register
class F1_macro:
    @staticmethod
    def compute(predict_matrix, target_matrix, **kwargs):
        predict_result = _binary_predictions(predict_matrix)
        return f1_score(target_matrix, predict_result, average='macro')


@register_obj.register
class GroupAccuracy:
    """Return one accuracy number per group: GroupAcc_0, GroupAcc_1, ..."""

    @staticmethod
    def compute(predict_matrix, target_matrix, group_matrix=None, **kwargs):
        if group_matrix is None:
            return {}
        predict_result = _binary_predictions(predict_matrix)
        group_matrix = np.asarray(group_matrix)
        result = {}
        for group_id in sorted(np.unique(group_matrix).tolist()):
            mask = group_matrix == group_id
            result[f'GroupAcc_{int(group_id)}'] = _safe_accuracy(
                predict_result[mask], target_matrix[mask]
            )
        return result


@register_obj.register
class WorstGroupAccuracy:
    @staticmethod
    def compute(predict_matrix, target_matrix, group_matrix=None, **kwargs):
        if group_matrix is None:
            return np.nan
        predict_result = _binary_predictions(predict_matrix)
        group_matrix = np.asarray(group_matrix)
        accs = []
        for group_id in sorted(np.unique(group_matrix).tolist()):
            mask = group_matrix == group_id
            accs.append(_safe_accuracy(predict_result[mask], target_matrix[mask]))
        return float(np.nanmin(accs)) if len(accs) > 0 else np.nan


@register_obj.register
class GroupGap:
    """max_g Acc_g - min_g Acc_g; lower is fairer/more balanced."""

    @staticmethod
    def compute(predict_matrix, target_matrix, group_matrix=None, **kwargs):
        if group_matrix is None:
            return np.nan
        predict_result = _binary_predictions(predict_matrix)
        group_matrix = np.asarray(group_matrix)
        accs = []
        for group_id in sorted(np.unique(group_matrix).tolist()):
            mask = group_matrix == group_id
            accs.append(_safe_accuracy(predict_result[mask], target_matrix[mask]))
        return float(np.nanmax(accs) - np.nanmin(accs)) if len(accs) > 0 else np.nan


@register_obj.register
class EnvAccuracy:
    """Return one accuracy number per environment: EnvAcc_0, EnvAcc_1, ..."""

    @staticmethod
    def compute(predict_matrix, target_matrix, env_matrix=None, **kwargs):
        if env_matrix is None:
            return {}
        predict_result = _binary_predictions(predict_matrix)
        env_matrix = np.asarray(env_matrix)
        result = {}
        for env_id in sorted(np.unique(env_matrix).tolist()):
            mask = env_matrix == env_id
            result[f'EnvAcc_{int(env_id)}'] = _safe_accuracy(
                predict_result[mask], target_matrix[mask]
            )
        return result
