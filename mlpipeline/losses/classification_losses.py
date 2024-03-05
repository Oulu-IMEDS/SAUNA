import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def create_class_loss(loss_name, **kwargs):
    if loss_name == 'CE':
        return CrossEnropy(**kwargs)
    else:
        raise ValueError(f'Not support loss {loss_name}.')


class CrossEnropy(nn.Module):
    def __init__(self, normalized=False, reduction='mean', **kwargs):
        super(CrossEnropy, self).__init__()
        self.reduction = reduction
        self.eps = 1e-7
        self.normalized = normalized

    def forward(self, input, target, normalized=None, alpha=None, *args, **kwargs):
        normalized = self.normalized if normalized is None else normalized
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1).type(torch.int64)

        if alpha is not None:
            _alpha = alpha
            if not isinstance(_alpha, torch.Tensor):
                _alpha = torch.tensor(_alpha)
        else:
            _alpha = None

        # For binary classification
        if input.dim() == 1 or (input.dim() == 2 and input.shape[1] == 1):
            if not normalized:
                logpt = F.logsigmoid(input)
            else:
                logpt = torch.log(input)
        else:  # Multi-class
            if not normalized:
                logpt = F.log_softmax(input, dim=-1)
            else:
                logpt = torch.log(input)
            logpt = logpt.gather(1, target)

        logpt = logpt.view(-1)

        loss = -logpt

        if _alpha is not None:
            if _alpha.type() != input.data.type():
                _alpha = _alpha.type_as(input.data)
            if len(_alpha.shape) == 1:
                at = _alpha.gather(0, target.data.view(-1))
            elif len(_alpha.shape) == 2:
                at = _alpha.gather(1, target).view(-1)
            else:
                raise ValueError(f'Not support alpha with dim = {len(_alpha.shape)}.')
            # at = _alpha
            loss = loss * Variable(at)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
