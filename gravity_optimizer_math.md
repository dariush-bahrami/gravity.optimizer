# Gravity Optimizer Math
Lets imagine an inclined plane and write some basic kinematic physics:

![inclined plane](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials/gravity_math_materials/gravity_optimizer_schematic.svg)

In this simple example lets forget the rolling part of the ball and focus on $a_L$. first we will calculate $\theta$ by using $dL$ and $dW$:
**Equation 1:** $tan(\theta) = \frac{dL}{dW} \Rightarrow \theta = tan^{-1}(\frac{dL}{dW})$
_from now on i call $\frac{dL}{dW}$ as $g$ which stands for gradient_
in this imaginary world there is a universal acceleration in L axis direction which is $a_L$. lets calculate relationship between inclined-plane's local coordinate system which is <x, y> and global coordinate system which is <W, L>. by using basic trigonometric relation we can achieve this as follow:
**Equation 2:** $W=x.cos\theta \Rightarrow \Delta W=\Delta x.cos\theta$
**Equation 3:** $L=x.sin\theta \Rightarrow \Delta L=\Delta x.sin\theta$
now lets  infer $a_x$:
**Equation 4:** $a_x=-a_L.sin\theta$
in this situation that $a_L$ and $\theta$ does not change with time we can write constant acceleration equation for position as follow:
**Equation 5:** $x=\frac{1}{2}.a_x.t^2+v_{0_x}.t+x_0$
by assuming $v_0$ is $0$ the new shape of equation 5 will be as below:
**Equation 6:** $\Delta x = \frac{1}{2}.a_x.t^2$
and by substituting equation 6 into equation 2 we can write:
**Equation 7:** $\Delta W = \frac{1}{2}.a_x.t^2 cos(\theta)$
plug $a_x$ from equation 4 in equation 7:
**Equation 8:** $\Delta W = -\frac{1}{2}.a_L.t^2.cos(\theta).sin(\theta)$
[equations](https://en.wikipedia.org/wiki/Inverse_trigonometric_functions) for calculating sine and cosine of $tan^{-1}$ are as follow: 
**Equation 9:** $sin(tan^{-1}(x))=\frac{x}{\sqrt{1+x^2}}$
**Equation 10:** $cos(tan^{-1}(x))=\frac{1}{\sqrt{1+x^2}}$
now by using equations 1, 8, 9, 10 we can write:
**Equation 11:** $\Delta W = -\frac{1}{2}.a_L.t^2.\frac{g}{1+g^2}$
Equation 11 is our raw parameter-update equation. but as you can see there is a lot of hyper-parameters in this equation which need a lot of time for tuning. beside that these hyper-parameters are not intuitive.

## Learning Rate
 we used to work with more familiar hyper-parameters like learning rate. so lets wrap up every hyper-parameter until now as a single one which is learning rate:
**Equation 12:** $l=\frac{1}{2}.a_L.t^2$
then:
**Equation 13:** $\Delta W = \frac{-lg}{1+g^2}$
this is more intuitive equation but it does not look so reasonable! how we can be sure about  equation 12? is that really learning rate or just a name for an unknown hyper-parameter?
lets look at optimization methods in different way. we can simplify all back-prop based optimization methods as a function of gradient. every optimization method gets gradient and calculate a step for parameter to take. for example vanilla Gradient Descent is a linear function of gradient:
**Equation 14:** $\Delta W = -lg$
we can plot that as below:
![SGD](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials/gravity_math_materials/sgd.svg)
now lets plot gravity from Equation 13 in this plot and compare it to GD for a given value of learning rate:
![Gravity & SGD](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials/gravity_math_materials/gravity_and_sgd.svg)
it is obvious from this plot that **for small value of gradient Gravity behave like Gradient Descent**. in math words $\lim_{g \to 0} Gravity(g) = GD(g)$. but how much small? lets try different learning rate and plot that too:
![Gravity Learning Rate Comparison](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials/gravity_math_materials/gravity_plot_lr_compare.svg)
in this plot we can see by changing learning rate the gradient at which maximum step occurs(i.e. Extremum) will not change. no matter what learning rate we choose, the maximum step always occurs at $g=1$ which corresponds to 45°.
## Max-Step Grad
We need more control. so far only parameters we encounter are $a_L$ and $t$ which both of them are universal. by universal i mean they are same for every parameter in weight matrix. we want specific control on this particular parameter.well, say hello to Galileo Galilei.
![Galileo Galilei](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Justus_Sustermans_-_Portrait_of_Galileo_Galilei%2C_1636.jpg/472px-Justus_Sustermans_-_Portrait_of_Galileo_Galilei%2C_1636.jpg)
you probably heard the myth which states that Galileo had dropped balls from the Leaning Tower of Pisa to demonstrate that their time of descent was independent of their mass. While this story has been retold in popular accounts, there is no account by Galileo himself of such an experiment, and it is generally accepted by historians that it was at most a thought experiment  which did not actually take place. However, most of his experiments with falling bodies were carried out using inclined planes where both the issues of timing and air resistance were much reduced. in fact inclined plane acts like a slow motion video recorded by a high speed camera. when you increase the angle $\theta$ the falling time will be reduced and everything happens quickly. in contrast by reducing $\theta$ we bend time and slow down ball's falling motion.
we can do what Galileo did by changing $\theta$ in a way that help us to reach minimum of $L$ more quickly and without divergence. we will do this by tweaking gradient with a coefficient called $m$ which $m>0$. lets change equations which deal with g:
**Equation 15:** $tan(\theta) = \frac{1}{m}\frac{dL}{dW} \Rightarrow \theta = tan^{-1}( \frac{1}{m}\frac{dL}{dW})$
**Equation 16:** $\Delta W = -\frac{1}{2}.a_L.t^2.\frac{\frac{g}{m}}{1+(\frac{g}{m})^2}$
**Equation 17:** $l=\frac{1}{2m}.a_L.t^2$
**Equation 18:** $\Delta W = \frac{-lg}{1+(\frac{g}{m})^2}$
lets look at effect of m in plot:
![Gravity m Comparison](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials/gravity_math_materials/gravity_plot_m_compare.svg)
as can be seen m actually is gradient which at that, maximum step will occur. in math language if:
$f(g)=\Delta W = \frac{-lg}{1+(\frac{g}{m})^2}$ and $f'(g)=\frac{\partial f}{\partial g}$
then:
 $f'(m)=0$
this parameter enables us to tweak angle of inclined plane in our benefit. the maximum step for given $m$ and $l$ will be:
**Equation 19:** $\Delta W_{max} = \frac{lm}{2}$
$m$ has two effects; first one is its effect on linear part of the curve and second one is maximum step value. higher $m$ leads to wider linear part and also bigger step for weights with big $g$. in other words by increasing $m$ wider range of gradients will be treated linearly and weights with larger gradient value will take larger steps.
**A Little About Gradient Descent Divergence**
the cause of divergence in vanilla gradient descent at larger learning rates is weights with large gradients. in fact in gradient descent optimization method infinite amount of $\Delta W$ is possible! one common scenario in gradient descent divergence is as follow:
1. consider a wight with large gradient
2. this weight takes bigger step relative to others (_linearly proportional to their gradient ratio_)
3. after applying optimizer the weight goes too far and now has larger gradient which leads to another big step
4. steps 1 to 3 will happen forever
Sean Harrington in his awesome article [*Gradient Descent: High Learning Rates & Divergence*](https://thelaziestprogrammer.com/sharrington/math-of-machine-learning/gradient-descent-learning-rate-too-high) explain this more intuitively:
![Gradient Descent Divergence](https://github.com/dariush-bahrami/gravity.optimizer/raw/master/materials/gravity_math_materials/gd_divergence.png)
> 1.  We start at the white point in the “valley”, and calculate the gradient at that point.
    2.  We multiply our learning rates by our gradient and move along this vector to our new point (the slightly greenish point to the left of the white point)
        * _Because our learning rate was so high, combined with the magnitude of the gradient, we “jumped over” our local minimum._
    3.  We calculate our gradient at point 2, and make our next move, again, jumping over our local minimum
        * _Our gradient at point 2 is even greater than the gradient at point 1!_
        * _Our next step will again, jump over our valley, and we will rinse and repeat for eternity…_
    4.  Due to the convex, valley-like curve of our objective function, as we continue to jump from side to side, the gradient at each jump grows higher. Our error increases quadratically with each “jump”, and our algorithm diverges to infinite error.

### Choosing M (Max-Step Grad)
we want to limit the $\Delta W$ for weights with larger $g$. Given the fact that gradients are constantly changing during training, it is obvious that we can not choose $m$ in advance. So we suggest to select it based on the current gradient matrix. To avoid divergence, a gradient matrix with larger gradients must have a smaller $m$. Based on this we suggest that you select m as follows:
**Equation 19:** $m=\frac{1}{max(abs(G))}$
in this equation G is gradient matrix. geometrical interpretation of this equation is as follow:
1. we found largest gradient and therefore steepest $\theta$
2. calculate complementary angle correspond to it: $\alpha=\frac{\pi}{2}-\theta$
3. and choose $m=tan(\alpha)$
## Moving Average
Most of the common optimizers in deep learning like momentum, Adam and RMSProp use moving average in order to stabilize loss reduction therefore we tested exponential moving average on gravity and results was promising. the main issue with gravity before applying moving average was initial delay in cost decay although after some initial epochs without any reduce in loss the optimizer do its job very well. we first define gradient term, $\zeta$,  as follow:
**Equation 20:** $\zeta = \frac{g}{1+(\frac{g}{m})^2}$
also we define velocity, V, as follow:
**Equation 21:** $V_{t}=\beta V_{t-1}+(1-\beta)\zeta$
in equation 21 $\beta$ is a positive real number which $0<\beta<1$. $V_{t}$ is $V$ value in current update step and $V_{t-1}$ is $V$ at previous update step and we initial $V$ with 0 at $t=0$. below figure shows the effect of different values of $\beta$ on loss reduction:
![beta_tuning_loss.png](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials/gravity_math_materials/beta_tuning_loss.png)

after some experiments involving changing  $\beta$ value we find out the optimal value for $\beta$ in most cases is $0.9$ although in some cases tuning my be necessary. although moving average helped gravity in initial speed but still we can see some delay. for solving this issue we propose an alternative value of $\beta$. any specific value of $\beta$ averages over a number of previous data. actually we can find number of data included in average by below [equation](https://www.youtube.com/watch?v=NxTFlzBjS-4):
**Equation 22:** $number \ of \ averaged \ data \approx \frac{1}{1-\beta}$
but at the first epochs there is no enough data to be averaged. and also at the beginning of the training which $t$ is small and  $V_0=0$ the value value of equation 21 will be very small. there is a solution known as bias correction which modifies $V_t$ as follow:
**Equation 23:** $V_{t}=\frac{\beta V_{t-1}+(1-\beta)\zeta}{1-\beta^t}$
the logic behind equation 23 is by increasing the value of $t$ the denominator approach to $1$  and equations 23 and 21will output almost same result. but at the  beginning of the training the value of $1-\beta^t$ is small and dividing equation 21 to this small value enlarge total outcome.
we tried to use bias correction but at large values of $\beta$ (closer to 1) we encounter overflow in our code therefore we tried to solve the problem of equation 21 at initial steps. we propose an alternative to beta as follow:
**Equation 23:** $\hat{\beta} = \frac{\beta t+1}{t+2}$
value of equation 23 at large $t$ will became $\beta$. but at smaller values correct the value of $\beta$. lets drive the equation 23. by choosing any $\beta$ we will average almost over $\frac{1}{1-\beta}$ data. let say we want to use a variable value of $\beta$ which increase over time and tends to $1$, therefore always average over all data, we call this variable $\hat{\beta}$. for averaging over all data at each step we want to the amount of data that we average on be t+2 because at first step $t=0$ there is 2 data $V_0$ and $V_1$. then we can write:
$\frac{1}{1-\hat{\beta}}=t+2 \Rightarrow \hat{\beta}=\frac{t+1}{t+2}$
by increasing amount of $t$ the value of $\beta$ tends to $1$ which will average over all data. below table show the value of $\hat{\beta}$ for different values of $t$:
![beta_1_table.png](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials\gravity_math_materials\beta_1_table.png)
as you can see at each step the amount of data that we average on is equal to total available data. for averaging over $\frac{1}{1-\beta}$ we modify above equation as $\hat{\beta} = \frac{\beta t+1}{t+2}$ which is equation 23. now by increasing the value of $t$ the value of $\hat{\beta}$ became more and more close to $\beta$. below table shows the value of $\hat{\beta}$ for $\beta=0.5$:
![beta_0.5_table.png](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials\gravity_math_materials\beta_0.5_table.png)
as you can see always we average over exactly 2 data. but for larger values of $\beta$ it takes more steps to $\hat{\beta}$ became closer to $\beta$:
![beta_0.75_table.png](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials\gravity_math_materials\beta_0.75_table.png)
and for our recommended value of $\beta$, $0.9$, the behaviour of $\hat{\beta}$ is as follow:
![beta_0.9_table.png](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials\gravity_math_materials\beta_0.9_table.png)
in addition to modifying $\beta$ in equation 21 we came out with another solution for increasing the speed of optimizer at early steps  by using non-zero initial $V$.  instead of zero we initialized $V$ with random numbers with normal distribution.




