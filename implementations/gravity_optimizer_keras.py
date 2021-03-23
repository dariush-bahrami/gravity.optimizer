"""Keras Implementation of Gravity Optimizer"""
import tensorflow as tf


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
