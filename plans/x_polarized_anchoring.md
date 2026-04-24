# Implementation Plan: X-Polarized Initial Condition Anchoring

This plan outlines the steps to anchor the initial NQS wavefunction to an X-polarized state at $t=0$. An X-polarized state is an equal superposition of all computational basis states, meaning:
*   Probability target: $p(\sigma) = 2^{-N} \implies \log p(\sigma) = -N \log 2$
*   Phase target: $\phi(\sigma) = 0$

We will enforce this using a two-step approach:
1.  **Pretraining**: An initial optimization phase that explicitly minimizes the MSE loss against the X-polarized targets at $t=0$.
2.  **Lagrangian Penalty**: During the TDVP time evolution, we will add a penalty term ($\lambda \times \text{MSE}$) to the total gradient to prevent the wavefunction from drifting away from the initial condition at $t=0$.

## 1. Updates to `TrainingConfig` (`src/TDVP.py`)

Add the following fields to `TrainingConfig` to control the new behavior:
```python
    # Initial Condition Anchoring (t=0)
    pretrain_steps: int = 0
    pretrain_lr: float = 0.005
    lambda_ic: float = 0.0
```

## 2. Implement Pretraining Phase (`src/TDVP.py`)

We will introduce a `pretrain_model` function that runs before the main `train_loop`.

### The MSE Loss Function
```python
def x_polarized_mse_loss(wf_model, configurations, n_sites):
    wf_view = _ModelWavefunctionView(wf_model)
    logp, phi = wf_view(configurations, t=0.0)
    
    target_logp = -n_sites * jnp.log(2.0)
    target_phi = 0.0
    
    loss_logp = jnp.mean((logp - target_logp)**2)
    loss_phi = jnp.mean((phi - target_phi)**2)
    
    return loss_logp + loss_phi
```

### The Pretrain Loop
*   Create a separate optimizer for pretraining (e.g., standard Adam with `pretrain_lr`).
*   Create a `@nnx.jit` wrapped `pretrain_step` that computes the gradient of `x_polarized_mse_loss` and updates the parameters.
*   In `train_loop`, if `config.pretrain_steps > 0`, execute the pretraining steps using uniformly random binary configurations (since the target distribution is uniform, we don't even need MCMC sampling for the pretraining phase; we can just generate random bits).

## 3. Implement Lagrangian Penalty during TDVP (`src/TDVP.py`)

During the main joint-time TDVP step, if `config.lambda_ic > 0.0`, we need to compute the gradient of the initial condition MSE and add it to the TDVP gradient.

### Updating `train_loop`
Inside the main joint step (where we aggregate `grad_total` over time slices):
```python
if config.lambda_ic > 0.0:
    # 1. Generate random configurations (or use the current MCMC chains)
    # 2. Compute the gradient of the MSE loss at t=0
    # 3. Add lambda_ic * mse_grad to grad_total
```
To be efficient, we can evaluate the MSE penalty using the MCMC chains sampled at $t=0$ (if $t=0$ is part of the time slices), or simply generate a batch of uniformly random configurations specifically for evaluating the penalty gradient at $t=0.0$.

## Next Steps

Please review this plan. If you approve, I will exit Plan Mode and apply these changes to `src/TDVP.py` and the example scripts.