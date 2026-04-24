# Implementation Plan

This document outlines the roadmap for implementing a time-dependent neural quantum state (NQS) optimization framework using TDVP and Variational Monte Carlo (VMC).

For the detailed mathematical derivation of the loss and gradient, please refer to [math.md](math.md).

## Current Status Scan

1. `src/wavefunction.py`
   - **Persistent `tSpinNQS`**: Holds a single `tNQS` module instance.
   - **MLP Time Encoding**: Refactored `Encoder` to use a 2-layer MLP for time features, added to spin embeddings.
   - **Streamlined API**: Only `__call__(configuration, t)` is exposed, returning `(logp, phi)`.

2. `src/hamiltonian.py`
   - **TransverseIsingHamiltonian**: Implemented with explicit $JZZ + hX$ and OBC.
   - **Local Energy**: Efficiently computed using `jax.vmap` for both configuration batching and spin-flip sites.

3. `src/sampler.py`
   - **Spacetime Trajectory Sampling**: `metropolis_hastings_trajectory` uses `jax.lax.scan` for warm-starting across time slices.

4. `src/loss.py`
   - **Single-Pass Derivatives**: Uses `jax.jvp` to compute $\partial_t \log p$ and $\partial_t \phi$ in one forward pass.

5. `src/grad.py`
   - **Unified Spacetime Gradient**: `tdvp_vmc_trajectory_gradient` computes the unified gradient over a whole trajectory using `nnx.vmap`.

6. `src/TDVP.py`
   - **Vectorized Trajectory Driver**: Training loop optimized for joint trajectory optimization with `@nnx.jit`.
   - **Multi-device Support**: Automatic chain distribution using JAX SPMD `NamedSharding`.
   - **Initial Condition Anchoring**: Pretraining and Lagrangian penalty support for $t=0$, using a baseline mean log-probability for improved stability.

7. `src/observables.py`
   - **Per-site Measurement**: Estimates $\langle Z_i(t) \rangle$ and $\langle X_i(t) \rangle$.
   - **JIT-ready**: Registered as PyTrees.

## Implementation Plan

### Phase 1 — Wavefunction and Parameter Lifecycle (Done)
### Phase 2 — Hamiltonian Correctness (Done)
### Phase 3 — Trajectory Sampling (Done)
### Phase 4 — Residual Loss (Done)
### Phase 5 — Unified Spacetime Gradient (Done)
### Phase 6 — Training Driver (Done)
### Phase 7 — Validation and Benchmarking (Done)
### Phase 8 — Observables and Anchoring (Done)

## Minimal Objective Aligned

1. Generate warm-started samples $\sigma_{n,t}$.
2. Compute $A_{n,t}$, $B_{n,t}$, and joint trajectory $\hat L$.
3. Compute unified spacetime gradient using autodiff plus VMC correction.
4. Update parameters with an optimizer such as AdamW or Muon.
5. Anchoring of initial condition at $t=0$.

## Programming Style

* use JAX and flax to forward and backward the neural network
* use flax.nnx as in existing code base
* write what has been added and changed in history.md and git commit.
