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
Equation 11 is our raw parameter-update equation. but as you can see there is a lot of hyper-parameters in this equation which need a lot of time for tuning. beside that these hyper-parameters are not intuitive. we used to work with more familiar hyper-parameters like learning rate. so lets wrap up every hyper-parameter until now as a single one which is learning rate:
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
in this plot we can see by changing learning rate the gradient at which maximum step occurs(i.e. Extremum) will not change. no matter what learning rate we choose, the maximum step always occurs at $g=1$ which corresponds to 45°. but we need more control. so far only parameters we encounter are $a_L$ and $t$ which both of them are universal. by universal i mean they are same for every parameter in weight matrix. we want specific control on this particular parameter.well, say hello to Galileo Galilei.
![Galileo Galilei](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Justus_Sustermans_-_Portrait_of_Galileo_Galilei%2C_1636.jpg/472px-Justus_Sustermans_-_Portrait_of_Galileo_Galilei%2C_1636.jpg)
you probably heard the myth which states that Galileo had dropped balls from the Leaning Tower of Pisa to demonstrate that their time of descent was independent of their mass. While this story has been retold in popular accounts, there is no account by Galileo himself of such an experiment, and it is generally accepted by historians that it was at most a thought experiment  which did not actually take place. However, most of his experiments with falling bodies were carried out using inclined planes where both the issues of timing and air resistance were much reduced. in fact inclined plane acts like a slow motion video recorded by a high speed camera. when you increase the angle $\theta$ the falling time will be reduced and everything happens quickly. in contrast by reducing $\theta$ we bend time and slow down ball's falling motion.
we can do what Galileo did by changing $\theta$ in a way that help us to reach minimum of $L$ more quickly and without divergence. we will do this by tweaking gradient with a coefficient called $m$ which $m>0$. lets change equations which deal with g:
**Equation 15:** $tan(\theta) = \frac{1}{m}\frac{dL}{dW} \Rightarrow \theta = tan^{-1}( \frac{1}{m}\frac{dL}{dW})$
**Equation 16:** $\Delta W = -\frac{1}{2}.a_L.t^2.\frac{\frac{g}{m}}{1+(\frac{g}{m})^2}$
**Equation 17:** $l=\frac{1}{2m}.a_L.t^2$
**Equation 18:** $\Delta W = \frac{-lg}{1+(\frac{g}{m})^2}$
lets look at effect of m in plot:
![Gravity m Comparison](https://raw.githubusercontent.com/dariush-bahrami/gravity.optimizer/master/materials/gravity_math_materials/gravity_plot_m_compare.svg)
as can be seen m actually is gradient which at that maximum step will occur. in math language if:
$f(g)=\Delta W = \frac{-lg}{1+(\frac{g}{m})^2}$ and $f'(g)=\frac{\partial f}{\partial g}$
then:
 $f'(m)=0$
this parameter enable us to tweak angle of inclined plane in our benefit. the maximum step for given $m$ and $l$ will be:
**Equation 19:** $\Delta W_{max} = \frac{lm}{2}$
$m$ has two effects; first one is its effect on linear part of the curve and second one is maximum step value. higher $m$ leads to wider linear part and also bigger step for weights with big $g$. in other words by increasing $m$ wider range of gradients will be treated linearly and weights with larger gradient value will take larger steps.
the cause of divergence in vanilla gradient descent at larger learning rates is weights with large gradients. in fact in gradient descent optimization method infinite amount of $\Delta W$ is possible! one common scenario in gradient descent divergence is as follow:
1. wrong weight has a large gradient (_because it is wrong_)
2. wrong weight takes bigger step relative to others (_linearly proportional to their gradient ratio_)
3. wrong weight gone to far and now is more wrong which leads to another big step
Sean Harrington in his awesome article *Gradient Descent: High Learning Rates & Divergence* explain this more intuitively:
> 1.  We start at the white point in the “valley”, and calculate the gradient at that point.
    2.  We multiply our learning rates by our gradient and move along this vector to our new point (the slightly greenish point to the left of the white point)
    * _Because our learning rate was so high, combined with the magnitude of the gradient, we “jumped over” our local minimum._
    3.  We calculate our gradient at point 2, and make our next move, again, jumping over our local minimum
    * _Our gradient at point 2 is even greater than the gradient at point 1!_
    * _Our next step will again, jump over our valley, and we will rinse and repeat for eternity…_
    4.  Due to the convex, valley-like curve of our objective function, as we continue to jump from side to side, the gradient at each jump grows higher. Our error increases quadratically with each “jump”, and our algorithm diverges to infinite error.








every  $n×m$ weight matrix represents a $n*m$ dimension space. still we can examine them and plot them one by one as what we done so far. based on experiments we carried out in order to choosing best value of $m$ we concluded that the value of m should depends on other parameters.



