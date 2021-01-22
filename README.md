# Gravity Optimizer
Gravity is a kinematic approach to optimization based on back-propagation. The algorithm demonstrated below:
![Gravity Optimizer Algorithm]()

For ease of use a keras implementation is available:
```python
import tensorflow as tf

class Gravity(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.1, alpha=0.01, beta=0.9, name="Gravity", **kwargs):
        super(Gravity, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('decay', self._initial_decay)
        self._set_hyper('alpha', alpha)
        self._set_hyper('beta', beta)
        self.epsilon = 1e-7
    def _create_slots(self, var_list):
        alpha = self._get_hyper("alpha")
        stddev = alpha/self.learning_rate
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=stddev, seed=None)
        for var in var_list:
            self.add_slot(var, "velocity", initializer=initializer)

    @tf.function
    def _resource_apply_dense(self, grad, var):
        # Get Data
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_hat = self._get_hyper("beta", var_dtype)
        t = tf.cast(self.iterations, float)
        beta = (beta_hat*t + 1 )/(t + 2)
        velocity = self.get_slot(var, "velocity")

        # Calculations
        max_step_grad = 1/tf.math.reduce_max(tf.math.abs(grad))
        gradient_term = grad / (1 + (grad/max_step_grad)**2)

        # update variables
        updated_velocity = velocity.assign(beta*velocity + (1-beta)*gradient_term) 
        updated_var = var.assign(var - lr_t*updated_velocity)       
        
        # updates = [updated_var, updated_velocity]
        # return tf.group(*updates)
    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError
    def get_config(self):
        config = super(Gravity, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'decay': self._serialize_hyperparameter('decay'),
            'alpha': self._serialize_hyperparameter('alpha'),
            'beta': self._serialize_hyperparameter('beta'),
            'epsilon': self.epsilon,
        })
        return config
```
for using gravity optimizer, after defining Gravity class as above, you can use it like other common optimizer in Keras. you should first instantiate an object as follow: 
```python
gravity_opt = Gravity(learning_rate=0.1, alpha=0.01, beta=0.9)
```
then at model compilation step pass this object as optimizer argument:
```python
model.compile(optimizer=gravity_opt)
```

* for details on implementing Gravity in Keras see _gravity_optimizer_notebook.ipynb_ 
* for details on Gravity Optimizer math read _gravity_optimizer_math.md_
