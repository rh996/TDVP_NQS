# TDVP Neural Quantum States (NQS)

This project implements a high-performance **Variational Monte Carlo (VMC)** framework for optimizing time-dependent neural quantum states using the **Time-Dependent Variational Principle (TDVP)**. 

The framework is specifically designed to simulate the dynamics of the **Transverse-Field Ising Model (TFIM)** on a 1D spin chain, leveraging **JAX** and **Flax (NNX)** for hardware-accelerated, vectorized computations.

## Key Features

*   **Spacetime Vectorization**: Optimizes the entire time trajectory jointly. Sampling is warm-started across time using `jax.lax.scan`, and gradients are computed in a single unified pass using `jax.vmap`.
*   **Transformer-based Wavefunction**: A time-dependent NQS $\Psi_\theta(\sigma, t)$ using a Transformer architecture with additive MLP-based time encoding for high expressivity.
*   **Unbiased VMC Gradients**: Implements the exact mathematical gradient derived from the Schrödinger residual, including both the pathwise autodiff term and the sampling-measure covariance correction.
*   **Initial Condition Anchoring**: Ensures physical correctness by anchoring the $t=0$ state using a two-step process: MSE pretraining followed by a Lagrangian penalty during evolution.
*   **Multi-Core Scaling**: Transparently scales across all available CPU or TPU cores (e.g., TPU v5e) using JAX SPMD `NamedSharding`.
*   **Observables Library**: Built-in support for measuring site-averaged and per-site magnetization trajectories $\langle Z_i(t) \rangle$ and $\langle X_i(t) \rangle$.

## Installation

```bash
pip install -r requirements.txt
# For TPU support:
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Quick Start

You can run a full TDVP training run from an initial X-polarized state using the provided example script:

```bash
python example/train_fully_polarized_chain.py \
    --n-sites 8 \
    --optimizer-name adamw \
    --n-steps 500 \
    --time-steps 10 \
    --t-initial 0.0 \
    --t-final 1.0 \
    --n-chains 8 \
    --n-samples-per-chain 1000 \
    --thinning 10 \
    --pretrain-steps 100 \
    --lambda-ic 10.0
```

### Common Arguments:
*   `--n-sites`: Number of spins in the chain.
*   `--optimizer-name`: Choice of `adamw` or `muon`.
*   `--time-steps`: Number of joint time slices in the optimization window.
*   `--n-chains`: Number of parallel MCMC chains (should be a multiple of your CPU/TPU core count).
*   `--pretrain-steps`: Number of steps to anchor the $t=0$ state before evolution.
*   `--lambda-ic`: Strength of the Lagrangian penalty for the initial condition.

## Project Structure

*   `src/wavefunction.py`: Transformer-based NQS definition.
*   `src/TDVP.py`: Main training loop and vectorized trajectory driver.
*   `src/sampler.py`: Warm-started spacetime Metropolis-Hastings sampler.
*   `src/grad.py`: Unified spacetime VMC gradient estimator.
*   `src/loss.py`: Schrödinger residual loss and time-derivative kernels.
*   `src/hamiltonian.py`: Transverse-field Ising Model implementation.
*   `src/observables.py`: Monte Carlo and exact observable measurement.

## Documentation

*   **[math.md](math.md)**: Detailed mathematical derivation of the loss function and VMC gradient estimator.
*   **[Program.md](Program.md)**: Project roadmap and current implementation status.
*   **[history.md](history.md)**: Chronological log of architectural changes and optimizations.
