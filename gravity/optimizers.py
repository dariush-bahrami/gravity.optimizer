"""This module contain all optimizer classes developed in gravity project"""
import tensorflow as tf


class Gravity(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate, beta=0.9, name="Gravity", **kwargs):
        super(Gravity, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('beta', beta)
        self.epsilon = 1e-7

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "velocity")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        # Get Data
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)  # handle learning rate decay
        beta = self._get_hyper("beta", var_dtype)
        velocity = self.get_slot(var, "velocity")

        # Calculations
        max_step_grad = 1 / (tf.reduce_max(tf.abs(grad)) + self.epsilon)
        gradient_term = grad / (1 + (grad / (max_step_grad + self.epsilon))**2)

        # update variables
        updated_velocity = velocity.assign(beta * velocity +
                                           (1 - beta) * gradient_term)
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
            'beta':
            self._serialize_hyperparameter('beta'),
            'epsilon':
            self.epsilon,
        })
        return config


class Pace(tf.keras.optimizers.Optimizer):
    def __init__(self, lr, name="AdaptivePace", **kwargs):
        super().__init__(name, **kwargs)
        self.learning_rate = lr
        self.epsilon = 1e-7

    def _create_slots(self, var_list):
        pass

    @tf.function
    def _resource_apply_dense(self, grad, var):
        b = self.learning_rate
        a = b * abs(float(tf.math.reduce_std(grad)))
        step = -a * tf.math.tanh((b / a) * grad)
        updated_var = step + var
        var.assign(updated_var)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return base_config
