# TDVP NQS Mathematical Theory

## Goal

This project aims to implement a variational Monte Carlo framework for optimizing a time-dependent neural quantum state (NQS) by minimizing a physically meaningful loss function derived from the Schrödinger equation.

The concrete target is the transverse-field Ising model, using a neural wavefunction of the form

$$\Psi_\theta(\sigma,t)=\sqrt{p_\theta(\sigma,t)}\,e^{i\phi_\theta(\sigma,t)}.$$

The implementation should support:

1. Monte Carlo sampling from the Born distribution $p_\theta(\sigma,t) \propto |\Psi_\theta(\sigma,t)|^2$.
2. Evaluation of the time-local loss using sampled configurations.
3. Evaluation of the gradient of the loss with respect to network parameters.
4. Optimization of the parameters with autodiff for the pathwise term, plus the standard VMC correction from the sampling measure.

The project is focused on making the loss and its gradient explicit in terms of:

* $\log p_\theta(\sigma,t)$
* $\phi_\theta(\sigma,t)$
* spin-flip neighbors of each sampled configuration

This avoids relying on more abstract operator notation during implementation.

## Basic Theory

### 1. Wavefunction parametrization

We write the neural quantum state as

$$\Psi_\theta(\sigma,t)=\sqrt{p_\theta(\sigma,t)}\,e^{i\phi_\theta(\sigma,t)}.$$

Then

$$\log \Psi_\theta(\sigma,t)=\frac12\log p_\theta(\sigma,t)+i\phi_\theta(\sigma,t).$$

Its time derivative is

$$\partial_t \log \Psi_\theta(\sigma,t)=\frac12\partial_t \log p_\theta(\sigma,t)+i\partial_t \phi_\theta(\sigma,t).$$

### 2. Model Hamiltonian

For the transverse-field Ising model,

$$H=J\sum_{\langle i,j\rangle}\sigma_i^z\sigma_j^z+h\sum_i \sigma_i^x.$$

In the computational z-basis, the diagonal contribution is

$$E_{ZZ}(\sigma)=J\sum_{\langle i,j\rangle}\sigma_i\sigma_j,$$

and the off-diagonal X term connects $\sigma$ to configurations $\sigma^{(i)}$ obtained by flipping one spin.

### 3. Local energy

The local energy is

$$E_{\mathrm{loc}}(\sigma,t)=\frac{\langle \sigma|H|\Psi_\theta(t)\rangle}{\Psi_\theta(\sigma,t)}.$$

For the transverse-field Ising model,

$$E_{\mathrm{loc}}(\sigma,t)=J\sum_{\langle i,j\rangle}\sigma_i\sigma_j+h\sum_i \frac{\Psi_\theta(\sigma^{(i)},t)}{\Psi_\theta(\sigma,t)}.$$

Using the amplitude-phase form,

$$\frac{\Psi_\theta(\sigma^{(i)},t)}{\Psi_\theta(\sigma,t)}
=\exp\!\left[\frac12\Delta_i\log p(\sigma,t)+i\Delta_i\phi(\sigma,t)\right],$$

where

$$\Delta_i\log p(\sigma,t)=\log p_\theta(\sigma^{(i)},t)-\log p_\theta(\sigma,t),$$
$$\Delta_i\phi(\sigma,t)=\phi_\theta(\sigma^{(i)},t)-\phi_\theta(\sigma,t).$$

Therefore,

$$E_{\mathrm{loc}}(\sigma,t)=J\sum_{\langle i,j\rangle}\sigma_i\sigma_j+h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}e^{i\Delta_i\phi(\sigma,t)}.$$

So the real and imaginary parts are

$$\Re E_{\mathrm{loc}}(\sigma,t)=J\sum_{\langle i,j\rangle}\sigma_i\sigma_j+h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}\cos\bigl(\Delta_i\phi(\sigma,t)\bigr),$$

$$\Im E_{\mathrm{loc}}(\sigma,t)=h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}\sin\bigl(\Delta_i\phi(\sigma,t)\bigr).$$

### 4. Local residual and loss

Define the local residual

$$L_{\mathrm{loc}}(\sigma,t)=\partial_t\log\Psi_\theta(\sigma,t)+iE_{\mathrm{loc}}(\sigma,t).$$

Writing

$$L_{\mathrm{loc}}(\sigma,t)=A(\sigma,t)+iB(\sigma,t),$$

we obtain

$$A(\sigma,t)=\frac12\partial_t\log p_\theta(\sigma,t)-h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}\sin\bigl(\Delta_i\phi(\sigma,t)\bigr),$$

$$B(\sigma,t)=\partial_t\phi_\theta(\sigma,t)+J\sum_{\langle i,j\rangle}\sigma_i\sigma_j+h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}\cos\bigl(\Delta_i\phi(\sigma,t)\bigr).$$

The time-local loss is the variance of the residual under the Born distribution:

$$L(t)=\operatorname{Var}_{p_\theta}(A)+\operatorname{Var}_{p_\theta}(B).$$

With Monte Carlo samples $\sigma_n\sim p_\theta(\sigma,t)$, define

$$A_n=A(\sigma_n,t),\qquad B_n=B(\sigma_n,t),$$

$$\bar A=\frac1N\sum_{n=1}^N A_n,\qquad \bar B=\frac1N\sum_{n=1}^N B_n,$$

$$\ell_n=(A_n-\bar A)^2+(B_n-\bar B)^2.$$

Then the Monte Carlo estimator of the loss is

$$\hat L(t)=\frac1N\sum_{n=1}^N \ell_n.$$

### 5. Gradient of the loss

Let $\theta_k$ be one variational parameter. The gradient contains two pieces:

1. A pathwise derivative term obtained by autodiff on fixed samples.
2. A sampling-distribution correction term from VMC.

The correct estimator is

$$\partial_{\theta_k} L
=\mathbb E_{p_\theta}\bigl[\partial_{\theta_k}\ell(\sigma,t)\bigr]
+\mathbb E_{p_\theta}\bigl[\ell(\sigma,t)\,\partial_{\theta_k}\log p_\theta(\sigma,t)\bigr].$$

Using the fact that

$$\mathbb E_{p_\theta}[\partial_{\theta_k}\log p_\theta]=0,$$

this can be written as a covariance form,

$$\partial_{\theta_k} L
=\mathbb E_{p_\theta}\bigl[\partial_{\theta_k}\ell(\sigma,t)\bigr]
+\operatorname{Cov}_{p_\theta}\bigl(\ell(\sigma,t),\partial_{\theta_k}\log p_\theta(\sigma,t)\bigr).$$

With samples, the practical estimator is

$$\partial_{\theta_k}\hat L
\approx
\frac1N\sum_{n=1}^N \partial_{\theta_k}\ell_n
+\frac1N\sum_{n=1}^N (\ell_n-\bar\ell)\left(\partial_{\theta_k}\log p_n-\overline{\partial_{\theta_k}\log p}\right),$$

where

$$\bar\ell=\frac1N\sum_{n=1}^N \ell_n.$$

The per-sample derivative is

$$\partial_{\theta_k}\ell_n
=2(A_n-\bar A)(\partial_{\theta_k}A_n-\partial_{\theta_k}\bar A)
+2(B_n-\bar B)(\partial_{\theta_k}B_n-\partial_{\theta_k}\bar B),$$

with

$$\partial_{\theta_k}\bar A=\frac1N\sum_m \partial_{\theta_k}A_m,
\qquad
\partial_{\theta_k}\bar B=\frac1N\sum_m \partial_{\theta_k}B_m.$$

The needed derivatives are

$$\partial_{\theta_k}A_n
=\frac12\partial_{\theta_k}\partial_t\log p_n
-h\sum_i \partial_{\theta_k}\left(e^{\frac12\Delta_i\log p_n}\sin(\Delta_i\phi_n)\right),$$

$$\partial_{\theta_k}B_n
=\partial_{\theta_k}\partial_t\phi_n
+h\sum_i \partial_{\theta_k}\left(e^{\frac12\Delta_i\log p_n}\cos(\Delta_i\phi_n)\right).$$

Expanding by the chain rule,

$$\partial_{\theta_k}\left(e^{\frac12\Delta_i\log p}\sin(\Delta_i\phi)\right)
=e^{\frac12\Delta_i\log p}
\left[
\frac12\partial_{\theta_k}(\Delta_i\log p)\sin(\Delta_i\phi)
+\cos(\Delta_i\phi)\partial_{\theta_k}(\Delta_i\phi)
\right],$$

$$\partial_{\theta_k}\left(e^{\frac12\Delta_i\log p}\cos(\Delta_i\phi)\right)
=e^{\frac12\Delta_i\log p}
\left[
\frac12\partial_{\theta_k}(\Delta_i\log p)\cos(\Delta_i\phi)
-\sin(\Delta_i\phi)\partial_{\theta_k}(\Delta_i\phi)
\right].$$
