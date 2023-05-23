from sklearn.metrics import mean_squared_error, mean_absolute_error
from enum import Enum


class LossType(Enum):
    MAE = 1
    MSE = 2


def reg_gradient(y, tx, w):
    """
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    pred = tx.dot(w)
    grad = tx.T.dot(pred - y) * (1 / y.size)
    return grad


def calculate_loss(y, tx, w, loss_type: LossType):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        loss_type: LossType

    Returns:
        a non-negative loss
    """
    pred = tx.dot(w)
    if loss_type.name == LossType.MSE.name:
        return mean_squared_error(y, pred)
    elif loss_type.name == LossType.MAE.name:
        return mean_absolute_error(y, pred)
