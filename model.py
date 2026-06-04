from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from register import Register


class Output(nn.Module):
    def __init__(self, input_dim, output_dim, init=1, bias=False):
        super(Output, self).__init__()
        assert input_dim >= output_dim
        data = torch.zeros(output_dim, input_dim)
        for i in range(output_dim):
            data[i, i] = init
        self.weight = nn.Parameter(data)
        self.is_bias = bias
        if self.is_bias:
            data = torch.zeros(output_dim)
            self.bias = nn.Parameter(data)

    def forward(self, x):
        if self.is_bias:
            out = torch.mm(x, self.weight.t()) + self.bias
        else:
            out = torch.mm(x, self.weight.t())
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, drop_rate=0, batchnorm=False, output_features=1, bias=True):
        """
        Args:
            hidden_size: list of hidden dimensions. The number of elements equals
                the number of hidden layers.
        """
        super(MLP, self).__init__()
        self.hidden_size = [input_dim] + hidden_size
        self.hidden_layers = []
        for i in range(len(self.hidden_size) - 1):
            input_dim = self.hidden_size[i]
            output_dim = self.hidden_size[i + 1]
            self.hidden_layers.append((f'linear{i + 1}', nn.Linear(
                in_features=input_dim,
                out_features=output_dim,
                bias=bias,
            )))
            if batchnorm:
                self.hidden_layers.append((f'batchnorm{i + 1}', nn.BatchNorm1d(num_features=output_dim)))
            self.hidden_layers.append((f'relu{i + 1}', nn.ReLU()))
            self.hidden_layers.append((f'dropout{i + 1}', nn.Dropout(p=drop_rate)))
        self.hidden_layers.append((f'linear{len(self.hidden_size)}', nn.Linear(
            in_features=self.hidden_size[-1],
            out_features=output_features,
            bias=bias,
        )))

        # The homework IRM implementation treats this output layer as the fixed
        # scalar predictor w. It is intentionally excluded from the optimizer in
        # exp-mnist.py, but gradients through it are used by the IRM penalty.
        self.output_layer = Output(output_features, output_features, init=1)
        print(self.hidden_layers)
        self.fc = nn.Sequential(OrderedDict(self.hidden_layers))

    def forward(self, x):
        phi = self.fc(x)
        y = self.output_layer(phi)
        return y

    def model(self):
        return self.fc

    def head(self):
        return self.output_layer


def reset_parameters(module: nn.Module, method='default'):
    if method == 'default':
        for layer in module.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            else:
                reset_parameters(layer, method)
    elif method == 'normal':
        for p in module.parameters():
            nn.init.normal_(p)
    elif method == 'constant':
        for p in module.parameters():
            nn.init.constant_(p, 1)
    else:
        raise NotImplementedError


global loss_register
loss_register = Register('loss_register')


class Loss:
    def __init__(self, weight=None):
        self.weight = weight

    def update_weight(self, weight):
        """
        Args:
            weight: per-sample tensor or None. This is not used by ERM now, but
                it prepares the framework for JTT / Soft-JTT, where stage-1
                error samples receive larger weights in stage 2.
        """
        self.weight = weight

    def __call__(self, predict, target, env, group=None, attr=None, ids=None, reduction='mean'):
        raise NotImplementedError


@loss_register.register
class bce_loss(Loss):
    def __call__(self, predict, target, env, group=None, attr=None, ids=None, reduction='mean'):
        loss = nn.BCEWithLogitsLoss(reduction='none')(predict, target.float())

        # Optional per-sample weights, used by JTT / Soft-JTT.
        # Validation/test batches do not carry training-sample ids, so weights
        # are applied only when ids are available.
        if self.weight is not None and ids is not None:
            weight = self.weight.to(predict.device)[ids]
            loss = loss * weight.float()

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            # Keep the original homework behavior: average losses in each
            # environment first, then average over environments.
            total_loss = 0
            env_list = env.unique()
            for env_id in env_list:
                total_loss += loss[env == env_id].mean()
            total_loss /= len(env_list)
            return total_loss
        else:
            raise NotImplementedError


@loss_register.register
class groupDRO(Loss):
    def __init__(self, risk: Loss, device, n_groups=4, eta=0.05, group_by='group'):
        super(groupDRO, self).__init__()
        self.risk = risk
        self.device = device
        self.n_groups = n_groups
        self.eta = eta
        self.group_by = group_by
        self.prob = np.ones(n_groups) / n_groups

    def _select_group_ids(self, env, group):
        if self.group_by == 'env':
            return env
        if self.group_by == 'group':
            if group is None:
                raise ValueError('groupDRO with group_by="group" requires group labels.')
            return group
        raise ValueError(f'Unknown group_by={self.group_by}; expected "env" or "group".')

    def __call__(self, predict, target, env, group=None, attr=None, ids=None, reduction='mean'):
        loss = self.risk(predict, target, env, group=group, attr=attr, ids=ids, reduction='none')

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            dro_ids = self._select_group_ids(env, group)
            group_losses = []
            for i in range(self.n_groups):
                mask = dro_ids == i
                if mask.sum() > 0:
                    group_loss = loss[mask].mean()
                else:
                    group_loss = torch.tensor(0.0, device=self.device)
                group_losses.append(group_loss)

            with torch.no_grad():
                for i in range(self.n_groups):
                    # q_g <- q_g * exp(eta * loss_g)
                    self.prob[i] = self.prob[i] * np.exp(self.eta * group_losses[i].item())
                self.prob = self.prob / self.prob.sum()

            group_dro_loss = 0
            for i in range(self.n_groups):
                group_dro_loss += self.prob[i] * group_losses[i]
            return group_dro_loss
        else:
            raise NotImplementedError


def flatten_and_concat_variables(vs):
    """Flatten and concatenate variables into a single vector."""
    flatten_vs = [torch.flatten(v) for v in vs]
    return torch.cat(flatten_vs, axis=0)


@loss_register.register
class IRM(Loss):
    def __call__(self, predict, target, env, network: nn.Module, risk: Loss, group=None, attr=None, ids=None):
        """Calculate IRMv1 penalty.

        The environments remain the training environments, not the y-attr groups.
        """
        env_list = env.unique()
        penalty = 0.0

        for env_id in env_list:
            env_mask = env == env_id
            env_predict = predict[env_mask]
            env_target = target[env_mask]
            env_labels = env[env_mask]

            env_loss = risk(env_predict, env_target, env_labels, reduction='mean')
            grads = torch.autograd.grad(
                env_loss,
                network.parameters(),
                create_graph=True,
                allow_unused=True,
            )
            grads = [g for g in grads if g is not None]
            if len(grads) > 0:
                flat_grads = flatten_and_concat_variables(grads)
                penalty += torch.sum(flat_grads ** 2)

        penalty = penalty / len(env_list)
        return penalty
