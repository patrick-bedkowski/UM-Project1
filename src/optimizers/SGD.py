import numpy as np

from src.optimizers.Optimizer import Optimizer

from src.regression.stochastic_gradient import stochastic_gradient
from src.regression.reg import calculate_loss


class SGD(Optimizer):
    name = "SGD"
    n_params_to_tune = 1

    def __init__(self,
                 lambda_: float):
        """
        Implementation of SGD method.
        Args:
            q: Number of iterations for each the variance reduction gradient should be saved
            lambda_: Step size
        """
        self.lambda_ = lambda_

    def set_params(self, new_lambda):
        self.lambda_ = new_lambda

    def optimize(self, w_0, tx, y, max_iter, loss_type):
        """
        Compute Stochastic gradient Descent
        :param w_0: Initial weights vector
        :param max_iter: Maximum number of iterations
        :param tx: Built model
        :param y: Target data
        :return: List of Gradients
        """
        grads = []
        losses = []
        w = [w_0]
        n = len(y)

        for t in range(max_iter):
            i_t = np.random.choice(np.arange(n))  # get index of sample for which to compute gradient
            sto_grad = stochastic_gradient(y, tx, w[t], [i_t])
            w_next = w[t] - self.lambda_ * sto_grad

            if t % 50000 == 0:
                print(t)

            w.append(w_next)
            grads.append(sto_grad)
            losses.append(calculate_loss(y, tx, w_next, loss_type))

        return grads, losses
