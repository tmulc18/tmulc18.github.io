---
layout: post
title: Physics Priors for Deep Learning
mathjax: true
author: Tommy Mulc
comments: true
---

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

We can use knowledge in Physics to guide systems we are trying to model.  This is exactly what the papers Hamiltonian Neural Networks and Langarangian Neural Networks do.  We'll step throught a simple example that allows us to understand their work more easily.  In the Lagrangian paper, the authors model the dynamics of double pendulum; we'll use a single pendulum instead, which is what was used in the Hamiltonian paper.  Additonally, all our implementation will use TensorFlow.  The authors used PyTorch for the the Hamiltonian work and JAX for the Lagrangian work.  It turns out that JAX is the better tool for implimenting Lagrangian Neural Networks, but TensorFlow does work (although I emailed the authors, and they had only used JAX, so it's kindof news for them that this impliminetation works!).  

Outline:
1. Model the system 
2. Create a simulation
3. Learn dynamics of the system
4. Hamiltonian formulation
5. Lagrangian formulation
6. Fun with models

# System modeling
We would like to be able to simulate a double pedulum, and to do that we are going to model the system from first principles.

The double pendulum can be modeled use force diagrams. You can write equations that describe the acclerations of each of the masses, where each is a function of the angles and lengths of pendelums.    

<div class="imgcap_noborder" style="display: block; margin-left: auto; margin-right: auto; width:80%">
	<img src="/assets/DeepPhysicsPrior/double-pendulum.svg">
	<div class="thecap" style="text-align:left">Double Pendulum Diagrams.</div>
</div>

The full derivation can be found [here](https://www.myphysicslab.com/pendulum/double-pendulum-en.html), but below is the final result.

<div class="imgcap_noborder" style="display: block; margin-left: auto; margin-right: auto; width:80%">
        <img src="/assets/DeepPhysicsPrior/dp_eqn.png">
</div>

Note that to get the dynamics for the single pendulum, we can set $$ L_2 = m_2 = 0$$.  Then, because we are only interested in the first pendulum, our system state is

$$ q = [\theta_1, \theta'_1]^T $$

and the change in state (you might hear that the second derivative of a state described as the "dynamics") is 

$$ q' = [\theta'_1, \frac{-g \cdot \sin(\theta_1)}{L_1}]^T.$$

# Simulation 
I found [this](www.people.fas.harvard.edu/~djmorin/chap1.pdf) reference helpful when looking at how physics simulation work.

There are basically two ways to create a simulation.

The first is to assume that you have some function $$ m: R^{1+n} \rightarrow R^n $$, such that for every $$t$$ you can plug it and the initial state, $$q_0$$, into $$m$$ to get the complete state information.  You might encounter these types of systems in an intro differential equations course (think linear ODE).  Our single pendulum system doesn't fall into this category, but an object falling from the sky with no air does.  To simulate these systems, you take the initial state and evalute $$m$$ over all values of $$ \{t_0, t_1, ... , t_\tau\}$$ and look at the results.

The second type of system you might encounter won't have a DE that your can solve directly, thus you'll have to approximate the solution numerically.  You have some given intial state $$q_0$$ and a function $$R^n \rightarrow R^n$$ that defines how the state is changing at state $$q$$.  Armed with this, you can make small steps approximate the path of the state.  The simplest way to do this is with Euler's method, but the more accureate way, which the author's use in their paper, is RK4.  If you aren't familiar with Euler's method, here is how you'd take an Euler-step to get the next state

$$ q_{t+\Delta} = q_t + \Delta \cdot q'_t. $$

Using this method, we can simulate our single pendulum.

```python
import numpy as np

def f_analytical(theta, w, m=1.0,l=1.0,g=9.8):
  """Returns the dynamics [w, dw/dt]."""
  w_t = -g*np.sin(theta)/l
  return np.stack([w,w_t])

h = .01
steps = 300
t = np.linspace(0,steps*h,steps)

theta = 0.43 # initialize starting position
w = 0 # intitialize velocity to zero

results = np.zeros_like(t)
for i,t_ in enumerate(t):
  results[i] = theta
  w, w_t = f_analytical(theta,w)
  w = w + h*w_t
  theta = theta + h*w
```
Below are plots of the state and even an animation of our simulated states.

<div class="imgcap_noborder" style="display: block; margin-left: auto; margin-right: auto; width:50%">
        <img src="/assets/DeepPhysicsPrior/sim_1.png">
        <div class="thecap" style="text-align:center">Simulation results of single pendulum.</div>
        <img src="/assets/DeepPhysicsPrior/single_pendulum_analytic.gif">
        <div class="thecap" style="text-align:center">Video animation of single pendulum.</div>
</div>





