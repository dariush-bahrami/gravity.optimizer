# Gravity Optimizer
Gravity is a kinematic approach to optimization based on gradients. For details of the proposed optimizer read paper preprint. The Gravity algorithm depicted below:

![Gravity Optimizer Algorithm](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials/Gravity%20Optimizer%20-%20Algorithm%20-%20Readme%20version.png)

For ease of use a keras implementation of the algorithm is available:
```python
class Gravity(tf.keras.optimizers.Optimizer):
    def __init__(self,
                 learning_rate=0.1,
                 alpha=0.01,
                 beta=0.9,
                 name="Gravity",
                 **kwargs):
        super(Gravity, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('alpha', alpha)
        self._set_hyper('beta', beta)
        self.epsilon = 1e-7

    def _create_slots(self, var_list):
        alpha = self._get_hyper("alpha")
        stddev = alpha / self.learning_rate
        initializer = tf.keras.initializers.RandomNormal(mean=0.0,
                                                         stddev=stddev,
                                                         seed=None)
        for var in var_list:
            self.add_slot(var, "velocity", initializer=initializer)

    @tf.function
    def _resource_apply_dense(self, grad, var):
        # Get Data
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta = self._get_hyper("beta", var_dtype)
        t = tf.cast(self.iterations, float)
        beta_hat = (beta * t + 1) / (t + 2)
        velocity = self.get_slot(var, "velocity")

        # Calculations
        max_step_grad = 1 / tf.math.reduce_max(tf.math.abs(grad))
        gradient_term = grad / (1 + (grad / max_step_grad)**2)

        # update variables
        updated_velocity = velocity.assign(beta_hat * velocity +
                                           (1 - beta_hat) * gradient_term)
        updated_var = var.assign(var - lr_t * updated_velocity)

        # updates = [updated_var, updated_velocity]
        # return tf.group(*updates)
    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        config = super(Gravity, self).get_config()
        config.update({
            'learning_rate':
            self._serialize_hyperparameter('learning_rate'),
            'decay':
            self._serialize_hyperparameter('decay'),
            'alpha':
            self._serialize_hyperparameter('alpha'),
            'beta':
            self._serialize_hyperparameter('beta'),
            'epsilon':
            self.epsilon,
        })
        return config
```

* Benchmarks are available at _Gravity Optimizer Benchmarks (Tensorflow).ipynb_ Notebook
---
PyTorch implementation of Gravity optimizer (*added in July 2021*):

```python
import torch
from torch.optim import Optimizer

class Gravity(Optimizer):
    """PyTorch implementation of Gravity optimizer.

    Gravity optimizer uses a kinematic approach to minimize the cost function. More
    detail can be found in corresponding paper at https://arxiv.org/abs/2101.09192.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.1)
        alpha (float, optional): alpha controls the V initialization (defaults to 0.01)
        beta (float, optional): beta will be used to compute running
            average of V (defaults to 0.9)
    """

    def __init__(self, params, lr=0.1, alpha=0.01, beta=0.9):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta value: {}".format(beta))
        defaults = dict(lr=lr, alpha=alpha, beta=beta, t=0)
        super(Gravity, self).__init__(params, defaults)
        self.__init_v()

    def __init_v(self):
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            std = alpha / lr
            for p in group["params"]:
                state = self.state[p]
                state["v"] = torch.empty_like(p).normal_(mean=0.0, std=std)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            beta = group["beta"]
            group["t"] += 1
            t = group["t"]
            beta_hat = (beta * t + 1) / (t + 2)
            for p in group["params"]:
                if p.grad is not None:
                    if p.grad.is_sparse:
                        raise RuntimeError("Gravity does not support sparse gradients")
                    state = self.state[p]
                    m = 1 / torch.abs(p.grad).max()
                    zeta = p.grad / (1 + (p.grad / m) ** 2)
                    state["v"] = beta_hat * state["v"] + (1 - beta_hat) * zeta
                    p -= lr * state["v"]
        return loss
```






