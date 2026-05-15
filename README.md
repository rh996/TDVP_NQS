# TDVP Neural Quantum States

This repository implements time-dependent neural quantum states for 1D spin-chain dynamics with the Time-Dependent Variational Principle (TDVP). The code is written in JAX and Flax NNX, with both Metropolis-Hastings VMC and direct autoregressive sampling.

The main target systems are:

- Short-range transverse-field Ising model (TFIM)
- Long-range transverse-field Ising model (LRTFIM)

The current development path focuses on autoregressive wavefunctions, exact Born-rule sampling, symmetry-aware architectures, and stable spacetime training losses.

## Main Features

- `tSpinNQS`: original time-dependent transformer NQS for MCMC training.
- `tSpinNQS_Z2`: Z2-symmetric transformer NQS.
- `AutoregressiveNQS`: non-Z2 autoregressive amplitude model plus phase network.
- `AutoregressiveNQS_Z2`: Z2-constrained autoregressive amplitude model.
- `NeuralGalerkinNQS`: fixed unnormalized uniform `psi_0` plus trainable transformer basis states with explicit time coefficients.
- Direct autoregressive sampling with logical `n_chains` batching.
- MCMC trajectory sampling for non-autoregressive models.
- Transformer attention with XSA: exclusive self-attention projects out the self-value direction.
- Attention residual mixer over completed residual blocks and current partial block.
- RoPE positional encoding in symmetry-preserving attention paths.
- Stratified random time collocation across `[t_initial, t_final]`.
- Initial-state anchoring at `t_initial`.
- Residual loss modes:
  - `variance`: `(A - <A>)^2 + (B - <B>)^2`
  - `schrodinger_l2`: `A^2 + B^2`
  - `phase_speed`: `A^2 + (B - <B>)^2`
- Optimizers: `adamw` and `muon`.
- Optional gradient clipping.
- Checkpoint save/reload utilities.
- Observable measurement for `Z_i(t)`, `X_i(t)`, and energy curves.
- Exact statevector export for small autoregressive systems.

## Installation

```bash
pip install -r requirements.txt
```

For TPU environments, install the matching JAX TPU wheel:

```bash
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Quick Start

### Autoregressive TFIM with Z2 Symmetry

This is the default autoregressive example. It now defaults to `muon` and `phase_speed`.

```bash
python example/train_autoregressive.py \
  --n-sites 8 \
  --n-steps 500 \
  --n-chains 8 \
  --n-samples-per-chain 1000 \
  --time-steps 10 \
  --t-initial 0.0 \
  --t-final 1.0 \
  --pretrain-steps 100 \
  --lambda-ic 10.0 \
  --gradient-clip-norm 1.0
```

### Autoregressive TFIM Without Z2

```bash
python example/train_autoregressive_no_z2.py \
  --n-sites 8 \
  --n-steps 500 \
  --n-chains 8 \
  --n-samples-per-chain 1000 \
  --time-steps 10 \
  --pretrain-steps 100 \
  --lambda-ic 10.0
```

### Autoregressive Long-Range TFIM

```bash
python example/train_autoregressive_lrtfim.py \
  --n-sites 8 \
  --alpha 1.2 \
  --n-steps 500 \
  --n-chains 8 \
  --n-samples-per-chain 1000 \
  --time-steps 10 \
  --pretrain-steps 100 \
  --lambda-ic 10.0
```

This example writes loss, energy, magnetization plots, and checkpoints every 200 steps.

### MCMC tSpinNQS Example

```bash
python example/train_mcmc_tspinnqs.py \
  --n-sites 8 \
  --n-steps 500 \
  --n-chains 32 \
  --n-samples-per-chain 100 \
  --time-steps 10
```

### Neural Galerkin TFIM

```bash
python example/train_neural_galerkin_tfim.py \
  --n-sites 8 \
  --num-basis 4 \
  --num-modes 4 \
  --n-steps 500 \
  --n-chains 16 \
  --n-samples-per-chain 256 \
  --time-steps 10
```

This model uses MCMC and `variance` loss by default because the Galerkin wavefunction is not autoregressively normalized.

## Important Arguments

- `--optimizer-name`: `muon` or `adamw`. Autoregressive examples default to `muon`.
- `--residual-loss-mode`: `phase_speed`, `schrodinger_l2`, or `variance`. Autoregressive examples default to `phase_speed`.
- `--n-sites`: number of spins.
- `--n-steps`: TDVP optimization steps.
- `--time-steps`: number of time collocation points per training step.
- `--fixed-time-grid`: disables random time collocation and uses the fixed grid.
- `--measure-time-steps`: number of post-training time points for observable plots.
- `--measure-t-initial`: initial time for post-training observable plots.
- `--measure-t-final`: final time for post-training observable plots. AR examples default to one extra training window beyond `t_final`.
- `--n-chains`: parallel chains for MCMC, or logical AR batch groups.
- `--n-samples-per-chain`: samples per chain/logical chain per time slice.
- `--pretrain-steps`: initial-state pretraining steps at `t_initial`.
- `--lambda-ic`: initial-condition anchor strength during TDVP training.
- `--gradient-clip-norm`: optional global gradient clipping threshold.
- `--use-unique-ar-samples`: compress AR samples with static-shape unique counts.
- `--save-statevector-max-sites`: maximum system size for exhaustive `psi(x,t)` export.
- `--alpha`: long-range power-law exponent for LRTFIM examples.
- `--num-basis`: number of trainable Neural Galerkin basis states.
- `--num-modes`: number of exponential time-coefficient modes per basis.

## Loss Notation

The wavefunction is represented as:

```math
\psi_\theta(x,t) = \exp\left[\frac{1}{2}\log p_\theta(x,t) + i\phi_\theta(x,t)\right].
```

The local energy is:

```math
E_\mathrm{loc}(x,t) = \frac{(H\psi_\theta)(x,t)}{\psi_\theta(x,t)}
= E_R(x,t) + iE_I(x,t).
```

The residual components used by the code are:

```math
A = \frac{1}{2}\partial_t \log p_\theta - E_I
```

```math
B = \partial_t \phi_\theta + E_R
```

The local Schrodinger residual divided by the wavefunction is:

```math
\frac{i\partial_t\psi_\theta - H\psi_\theta}{\psi_\theta}
= -B + iA.
```

For autoregressive training, the default `phase_speed` loss is:

```math
L = \left\langle A^2 + (B - \langle B\rangle)^2 \right\rangle.
```

Autoregressive sampling parameterizes a normalized probability distribution directly, so there is no free global amplitude rescaling. The remaining gauge freedom is the global phase shift, which appears as the constant part of `B`.

## Time Sampling

When `random_time_collocation=True`, each training step uses stratified random time samples:

- `t_initial` is always included for the anchor.
- The remaining `time_steps - 1` points are sampled one per interval across `[t_initial, t_final]`.

For example, with `time_steps=5` and `[0, 1]`, the sampled times are:

```text
[0.0, sample in [0.00, 0.25), sample in [0.25, 0.50), sample in [0.50, 0.75), sample in [0.75, 1.00)]
```

Use `--fixed-time-grid` to train only on the deterministic grid.

## Examples

- `example/train_autoregressive.py`: Z2 autoregressive TFIM training.
- `example/train_autoregressive_no_z2.py`: non-Z2 autoregressive TFIM training.
- `example/train_autoregressive_lrtfim.py`: Z2 autoregressive long-range TFIM training with energy curves and checkpoints.
- `example/train_neural_galerkin_tfim.py`: Neural Galerkin TFIM training with MCMC and variance loss.
- `example/train_fully_polarized_chain.py`: MCMC TDVP from a fully polarized chain.
- `example/train_mcmc_tspinnqs.py`: explicit MCMC example using `tSpinNQS`.
- `example/train_simple_nqs.py`: simple NQS architecture example.
- `example/train_lrtfim.py`: non-autoregressive long-range TFIM example.
- `example/train_save_reload_measure.py`: checkpoint, reload, resume, and measure workflow.

## Project Layout

- `src/wavefunction.py`: wavefunction architectures, XSA, RoPE, attention residuals, AR models.
- `src/TDVP.py`: training loop, optimizers, checkpointing, stratified time sampling.
- `src/sampler.py`: Metropolis-Hastings and autoregressive trajectory samplers.
- `src/grad.py`: spacetime VMC gradient estimators with optional count weighting.
- `src/loss.py`: residual components `A`, `B`, and loss modes.
- `src/hamiltonian.py`: short-range and long-range TFIM Hamiltonians.
- `src/observables.py`: observables, statevector utilities, and energy curves.
- `tests/`: unit tests for wavefunctions, Hamiltonians, samplers, losses, gradients, TDVP, checkpoints, and observables.

## Outputs

Examples typically write to `example/outputs/...` unless `--output-dir` is provided. Depending on the script, outputs include:

- `loss.png`
- `z_trajectory.png`
- `x_trajectory.png`
- `energy_curve.png`
- `final_wavefunction.pkl`
- `final_psi_xt.npz` for small systems
- periodic checkpoints under `checkpoints/`

## Testing

Run the focused test suite:

```bash
pytest tests/test_wavefunction_phase1.py \
  tests/test_loss_phase4.py \
  tests/test_grad_phase5.py \
  tests/test_tdvp_phase6.py -q
```

Run all tests:

```bash
pytest -q
```

## Additional Notes

- AR sampler acceptance is reported as `1.0` because AR sampling is direct, not a Metropolis accept/reject process.
- For AR examples, total sample count is `n_chains * n_samples_per_chain`.
- For small `N`, saved `final_psi_xt.npz` can be loaded to inspect the learned wavefunction exactly over the computational basis.
- Large `N` exact statevector export is disabled by `--save-statevector-max-sites`.

## Documentation

- [math.md](math.md): mathematical derivation of residual losses and VMC gradients.
- [Batch Autoregressive Sampling.md](Batch%20Autoregressive%20Sampling.md): autoregressive sampling notes.
- [Program.md](Program.md): roadmap and implementation status.
- [history.md](history.md): chronological development notes.
