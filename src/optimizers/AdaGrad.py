import numpy as np

from src.optimizers.Optimizer import Optimizer

from src.regression.reg import reg_gradient, calculate_loss
from src.regression.stochastic_gradient import stochastic_gradient


class AdaGrad(Optimizer):
    name = "AdaGrad"
    n_params_to_tune = 2

    def __init__(self,
                 lambda_: float,
                 epsilon: float = 1e-8):
        """
        Implementation of AdaGrad method.
        Args:
            lambda_:
            q: Number of iterations for each the variance reduction gradient should be saved
            epsilon:
        """
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def set_params(self, new_lambda, new_epsilon):
        self.lambda_ = new_lambda
        self.epsilon = new_epsilon

    def optimize(self, w_0, tx, y, max_iter, loss_type):
        '''Algoritm for adaptive gradient optimization.

        Adapts learing parameter - smaller rate for frequent features (well-suited for sparse data).

        Parameters
        ----------
        w_0 : ndarray of shape (D, 1)
            Initial weights of the model
        tx : ndarray of shape (N, D)
            Array of input features
        y : ndarray of shape (N, 1)
            Array of output
        max_iter : int
            Maximum number of iterations
        Returns
        -------
        grads : ndarray of shape (max_iter, D)
            Array of gradient estimators in each step of the algorithm.
        '''
        D = len(w_0)
        G_t = np.zeros((D, D))
        n = len(y)

        # Outputs
        grads = []
        losses = []
        w = [w_0]

        for t in range(max_iter):
            i_t = np.random.choice(np.arange(n))  # get index of sample for which to compute gradient
            sto_grad = stochastic_gradient(y, tx, w[t], [i_t])

            g_t = reg_gradient(y, tx, w[t])
            G_t += np.linalg.norm(g_t) ** 2

            v_k = np.diag(self.lambda_ / (np.sqrt(G_t + self.epsilon))) @ g_t
            w_next = w[t] - v_k
            w.append(w_next)

            if t % 50000 == 0:
                print(t)

            grads.append(sto_grad)
            losses.append(calculate_loss(y, tx, w_next, loss_type))

        return grads, losses
