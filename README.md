# TDVP Neural Quantum States (NQS)

This project implements a **Variational Monte Carlo (VMC)** framework for optimizing time-dependent neural quantum states using the **Time-Dependent Variational Principle (TDVP)**.

The framework is designed to simulate 1D spin-chain dynamics for the **Transverse-Field Ising Model (TFIM)** and **Long-Range TFIM (LRTFIM)**, using **JAX** and **Flax NNX** for hardware-accelerated, vectorized computation.

## Key Features

*   **Spacetime Vectorization**: Optimizes the entire time trajectory jointly. Sampling is scanned across time, and gradients are computed through a unified trajectory estimator.
*   **Transformer-based Wavefunctions**: Time-dependent NQS models $\Psi_\theta(\sigma, t)$ using Transformer blocks with MLP-based time conditioning, including original `tSpinNQS` and symmetry-preserving `tSpinNQS_Z2`.
*   **Autoregressive NQS**: Supports original `AutoregressiveNQS` and Z2-constrained `AutoregressiveNQS_Z2` amplitude models with direct exact sampling from the learned Born distribution.
*   **Unbiased VMC Gradients**: Implements the exact mathematical gradient derived from the Schrödinger residual, including both the pathwise autodiff term and the sampling-measure covariance correction.
*   **Initial Condition Anchoring**: Ensures physical correctness by anchoring the $t=0$ state using a two-step process: MSE pretraining followed by a Lagrangian penalty during evolution.
*   **Long-Range TFIM**: Includes long-range Ising interactions $J \sum_{i<j} \sigma_i^z \sigma_j^z / |i-j|^\alpha$.
*   **Gradient Clipping**: Optional global-norm clipping through Optax for stabilizing large TDVP updates.
*   **Optional Unique AR Batches**: Autoregressive samples can be compressed with static-shape `jnp.unique(..., size=batch_size)` and count-weighted estimators for diagnostics or future optimization.
*   **Multi-Core Scaling**: Transparently scales across all available CPU or TPU cores (e.g., TPU v5e) using JAX SPMD `NamedSharding`.
*   **Observables Library**: Built-in support for $\langle Z_i(t) \rangle$, $\langle X_i(t) \rangle$, and sampled energy curves $\langle H(t) \rangle$.

## Installation

```bash
pip install -r requirements.txt
# For TPU support:
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Quick Start

You can run a full TDVP training run from an initial X-polarized state using the standard TFIM example:

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

For autoregressive training on the long-range TFIM:

```bash
python example/train_autoregressive_lrtfim.py \
    --n-sites 8 \
    --alpha 1.2 \
    --optimizer-name adamw \
    --learning-rate 0.001 \
    --gradient-clip-norm 1.0 \
    --n-steps 500 \
    --time-steps 10 \
    --t-initial 0.0 \
    --t-final 1.0 \
    --n-chains 8 \
    --n-samples-per-chain 1000 \
    --pretrain-steps 100 \
    --lambda-ic 10.0
```

This writes loss, energy, and magnetization plots plus periodic checkpoints under the output directory.

### Common Arguments:
*   `--n-sites`: Number of spins in the chain.
*   `--optimizer-name`: Choice of `adamw` or `muon`.
*   `--time-steps`: Number of joint time slices in the optimization window.
*   `--n-chains`: Number of parallel MCMC chains, or the logical chain axis for autoregressive batches.
*   `--n-samples-per-chain`: Samples per chain/logical chain at each time slice.
*   `--gradient-clip-norm`: Optional global gradient clipping threshold.
*   `--use-unique-ar-samples`: Optional count-weighted unique-sample path for autoregressive sampling.
*   `--alpha`: Long-range TFIM decay exponent for LRTFIM examples.
*   `--pretrain-steps`: Number of steps to anchor the $t=0$ state before evolution.
*   `--lambda-ic`: Strength of the Lagrangian penalty for the initial condition.

## Examples

*   `example/train_fully_polarized_chain.py`: Standard TFIM training from a fully polarized initial chain.
*   `example/train_mcmc_tspinnqs.py`: Explicit MCMC training example using the original `tSpinNQS`.
*   `example/train_simple_nqs.py`: Training with the simpler NQS architecture.
*   `example/train_autoregressive_no_z2.py`: Short-range TFIM training with original non-Z2 `AutoregressiveNQS`.
*   `example/train_autoregressive.py`: Short-range TFIM training with `AutoregressiveNQS_Z2`.
*   `example/train_lrtfim.py`: Long-range TFIM training with the non-autoregressive model.
*   `example/train_autoregressive_lrtfim.py`: Long-range TFIM training with `AutoregressiveNQS_Z2`, energy-curve measurement, and checkpoints every 200 steps.
*   `example/train_save_reload_measure.py`: Save, reload, resume, and measure workflow.

## Project Structure

*   `src/wavefunction.py`: Original, Z2-symmetric, simple, and autoregressive NQS definitions.
*   `src/TDVP.py`: Main training loop, optimizer construction, checkpointing, and trajectory driver.
*   `src/sampler.py`: Warm-started Metropolis-Hastings sampler and direct autoregressive sampler.
*   `src/grad.py`: Unified spacetime VMC gradient estimator with optional count weighting.
*   `src/loss.py`: Schrödinger residual loss and weighted batch loss support.
*   `src/hamiltonian.py`: Short-range TFIM and long-range TFIM local-energy implementations.
*   `src/observables.py`: Monte Carlo and exact observables, plus sampled energy curves.

## Documentation

*   **[math.md](math.md)**: Detailed mathematical derivation of the loss function and VMC gradient estimator.
*   **[Program.md](Program.md)**: Project roadmap and current implementation status.
*   **[history.md](history.md)**: Chronological log of architectural changes and optimizations.
