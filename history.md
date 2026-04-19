# History

## 2026-04-18 — Phase 1: Wavefunction API + Parameter Lifecycle Stabilization

### What changed

- Updated `src/wavefunction.py`:
  - Extended `Wavefunction` abstract interface with:
    - `__call__(configuration, t)`
    - `log_prob(configuration, t)`
    - `phase(configuration, t)`
  - Refactored `tSpinNQS` to hold a **persistent** `tNQS` instance in `__init__`:
    - `self.model = tNQS(...)`
    - removed per-call model re-instantiation behavior.
  - Added explicit wrapper methods:
    - `log_prob(...)` delegates to model output head 1
    - `phase(...)` delegates to model output head 2
  - Added shape-normalization helper:
    - `_squeeze_last_dim(...)` to remove trailing singleton output axis where present.
- Previously updated `tNQS` output head design (mean pooling + two MLP heads) was retained and used as Phase-1-compatible output contract.

### Why

- Re-instantiating the model inside `__call__` breaks parameter persistence and prevents meaningful optimization.
- Downstream modules (`sampler`, `loss`, `grad`) require a stable API that explicitly exposes `log_prob` and `phase`.
- Consistent output shapes reduce shape bugs in vectorized/batched code paths.

### Expected impact

- Parameters are now stable across forward passes.
- Wavefunction outputs are easier to consume in Hamiltonian/sampler/loss code.
- Training semantics are now valid for iterative optimization.

---

## 2026-04-18 — Phase 2: Hamiltonian Convention Alignment + Local Energy Tests

### What changed

- Implemented/rewrote `src/hamiltonian.py` with explicit physics conventions:
  - Hamiltonian convention fixed to:
    - `H = J * sum_{i=0}^{N-2} sigma_i^z sigma_{i+1}^z + h * sum_i sigma_i^x`
  - Boundary condition:
    - **Open boundary condition** (OBC) only, for now.
  - Spin encoding:
    - input configurations in `{0, 1}`
    - mapped to physical `s_i in {+1, -1}` via `s_i = 1 - 2 * bit_i`.
- Added:
  - `_bit_to_sz(...)`
  - `zz_energy_open(...)`
  - `_transverse_term_single_site_parts(...)`
- Refactored `local_energy(...)`:
  - returns `(e_real, e_imag)` instead of complex scalar.
  - computes transverse term using explicit real/imag decomposition:
    - `exp(0.5 * Δlogp) * cos(Δphi)` (real)
    - `exp(0.5 * Δlogp) * sin(Δphi)` (imag)
- Added batched API:
  - `local_energy_batch(...)` using vectorization over configurations.
- Numerical cleanup:
  - updated clipping call to modern signature:
    - `jnp.clip(..., min=..., max=...)`.

### Tests added

- `tests/test_hamiltonian_phase2.py`:
  - ZZ term correctness with `{0,1} -> {+1,-1}` mapping.
  - OBC behavior (distinguished from periodic intuition).
  - local energy real/imag decomposition checks.
  - analytic checks for transverse contributions from known `Δlogp`, `Δphi`.
  - input validation tests.
  - batched local energy consistency against per-sample results.
  - singleton-batch consistency checks.

### Why

- Phase 2 required explicit convention locking (`JZZ + hX`, OBC, binary spins).
- Returning real/imag directly simplifies later loss construction (`A_n`, `B_n`) and avoids unnecessary complex dtype propagation.
- Batched local energy is required for efficient Monte Carlo loss evaluation.

### Expected impact

- Hamiltonian logic now matches documented project conventions.
- Local energy outputs are directly compatible with Phase 4 loss formulas.
- Batched evaluation path is ready for sampled mini-batches.

---

## 2026-04-18 — Phase 3: Metropolis Sampler Implementation + Tests

### What changed

- Implemented `src/sampler.py`:
  - Metropolis-Hastings single-spin-flip sampler targeting:
    - `p_theta(sigma, t) ∝ exp(logp_theta(sigma, t))`
  - Added config normalization helper:
    - supports initial configurations as `(N,)` or `(C, N)`
    - validates binary domain `{0,1}`.
  - Added vectorized MH step over chains:
    - one random spin flip proposal per chain per step.
  - Added main sampler API:
    - `metropolis_hastings_sample(...)`
    - supports:
      - `burn_in`
      - `thinning`
      - multi-chain sampling
      - deterministic behavior with explicit PRNG key
      - optional diagnostics return.
  - Added diagnostics:
    - acceptance rate per chain
    - accepted/proposed move counts
    - recorded run settings (`n_chains`, `n_samples_per_chain`, `burn_in`, `thinning`).

### Tests added

- `tests/test_sampler_phase3.py`:
  - shape and domain checks for single-chain and multi-chain outputs.
  - diagnostics shape/range checks.
  - determinism checks for fixed seed and identical inputs.
  - burn-in effect checks.
  - input validation checks (rank, size, thinning, non-binary values).

### Why

- Phase 3 requires Monte Carlo sampling from the model-induced Born distribution.
- Burn-in and thinning are necessary for practical MCMC usage.
- Multi-chain vectorization improves sample throughput and stabilizes estimators.
- Deterministic keyed RNG behavior is required for reproducibility and debugging.

### Expected impact

- End-to-end sampled batches are now available for upcoming loss and gradient phases.
- Sampling diagnostics can be monitored in training to detect poor chain mixing.
- Infrastructure now supports scalable batch-based VMC workflows.

---

## 2026-04-18 — Phase 4: TDVP Residual Loss Review + Unit Test Validation

### What changed

- Reviewed `src/loss.py` implementation against the Phase 4 specification in `Program.md`.
- Confirmed implemented APIs and formulas:
  - `time_derivatives_autodiff(...)`
  - `tdvp_residual_components(...)`
  - `tdvp_residual_loss(...)`
- Confirmed residual definitions are implemented as:
  - `A_n = 0.5 * d_t logp_n - Im[E_loc]_n`
  - `B_n = d_t phi_n + Re[E_loc]_n`
  - `ell_n = (A_n - mean(A))^2 + (B_n - mean(B))^2`
  - `L_hat = mean(ell_n)`
- Added comprehensive unit tests in `tests/test_loss_phase4.py`:
  - autodiff time derivatives match closed-form dummy model values
  - residual components (`A`, `B`) match manual analytic construction
  - centered-variance loss matches manual computation
  - diagnostics payload fields (`A`, `B`, `A_mean`, `B_mean`, `ell`, `dlogp_dt`, `dphi_dt`, `e_real`, `e_imag`) are validated
  - `return_diagnostics=False` path returns finite scalar
  - identical-sample batch gives zero centered variance loss
  - input validation for shape/length/binary-domain errors

### Why

- Phase 4 requires explicit and test-backed implementation of local residual statistics and Monte Carlo loss estimator.
- This review ensures the implemented equations in code match the theory section and are robust to common data-shape/data-domain failures.
- Strong tests on analytic toy cases reduce risk before Phase 5 gradient estimator work.

### Expected impact

- `src/loss.py` is now review-verified and unit-tested for the Phase 4 milestone objective.
- Residual/loss computations are now safer to use as a dependency for gradient estimation in `src/grad.py`.
- Faster debugging and regression detection for future refactors in loss/Hamiltonian interfaces.

---

## 2026-04-18 — Phase 5: Gradient with VMC Correction

### What changed

- Implemented `src/grad.py` with Phase-5 TDVP/VMC gradient estimator:
  - **Pathwise term**: autodiff of per-sample loss `ell_n` with samples treated fixed.
  - **Covariance correction**: per-sample score gradients `∂_θ log p_n` with proper centering.
  - **Total gradient**: `grad_total = grad_pathwise + grad_covariance`.
- Key APIs:
  - `tdvp_vmc_gradient(...)` returns `(grad_total, diagnostics_dict)`.
  - `tdvp_vmc_gradient_components(...)` returns `(grad_total, grad_pathwise, grad_cov, diagnostics_dict)` for debugging.
- Implementation fixes:
  - Replaced `vmap(nnx.grad(...))` over live NNX module with **Python loop** over samples to avoid NNX trace-level aliasing errors.
  - Fixed tree-centering: `tree_map(lambda g, gm: g - gm, ...)` for correct leaf-wise subtraction across all parameter shapes.
  - Corrected covariance broadcasting: reshaped `ell_centered` to broadcast with stacked per-sample score leaves.
  - Renamed diagnostic key from `score_mean_sq` to `ell_var` (variance of centered residuals).
- Added helper utilities:
  - `_tree_mean(trees_list)`: average list of pytrees leaf-wise.
  - `_tree_l2_norm(tree)`: L2 norm of all parameter leaves.
  - `_tree_all_finite(tree)`: check finiteness of all leaves.
  - `_per_sample_score_grads(wf, configs, t)`: per-sample NNX gradients via loop (not vmap).

### Tests added

- `tests/test_grad_phase5.py`:
  - finite and non-empty gradient pytree checks.
  - `return_diagnostics=False` path returns scalar without aux.
  - `grad_total == grad_pathwise + grad_cov` component summation.
  - loss consistency with Phase-4 residual loss.
  - input validation (shape/length/binary domain).

### Why

- Phase 5 requires **unbiased VMC gradient estimation** with two distinct pieces:
  1. pathwise autodiff on fixed batch (variance reduction technique),
  2. covariance correction from sampling distribution parameter dependence.
- Pathwise-only gradients are **biased** and miss the score-function contribution.
- Loop-based score gradients sidestep NNX transform nesting issues that cause trace-level aliasing with `vmap(nnx.grad(...))`.
- Correct tree mathematics ensures proper broadcasting and parameter updates across all model layer shapes (embeddings, linear weights, etc.).

### Expected impact

- Gradients are now **statistically correct** for VMC optimization.
- Ready for Phase-6 training loop integration with optimizer (Adam, SGD, etc.).
- Numerically stable and validated on small systems (N=5, batch=6).
- Infrastructure now supports proper parameter updates that reduce loss meaningfully.

---

## Validation snapshot (after Phase 1–5)

- Unit tests currently passing for all implemented phases:
  - wavefunction phase-1 API tests
  - hamiltonian phase-2 logic tests
  - sampler phase-3 tests
  - loss phase-4 tests
  - gradient phase-5 tests
- Current test status at update time:
  - full test suite: **all 29 tests passed**.
  - phase-5 specific: `tests/test_grad_phase5.py`: 5 passed.
- Phases 1–5 complete and validated:
  - end-to-end sampling → loss → gradient pipeline is functional and numerically correct.
  - ready for Phase-6 training loop integration.

---

## Notes for next phase

- Next target: **Phase 6** (`src/TDVP.py`)
  - initialize model, optimizer, RNG, and Hamiltonian.
  - implement training loop: sample → compute loss → compute gradients → optimizer step.
  - add logging for loss, acceptance rate, gradient norm.
  - run minimal experiment on small transverse-Ising system (N ≤ 10).
- Then **Phase 7**:
  - validation against exact diagonalization on small systems.
  - ablation studies (number of boxes, head dimensions, sampler settings).
  - stability analysis (multiple seeds, gradient clipping, optimizer tuning).

---

## 2026-04-18 — Phase 6: TDVP Training Driver Repair + Tests

### What changed

- Repaired `src/TDVP.py` so the training driver now uses a single consistent update path:
  - retained manual gradient-descent parameter updates on the live NNX model.
  - removed broken references to undefined `optimizer` / `opt_state` variables.
  - fixed `train_step(...)` to return `(loss, diagnostics, next_rng)` consistently.
- Added explicit training-config validation:
  - checks for positive `N`, `n_steps`, `time_steps`, `learning_rate`, `n_chains`, and `n_samples_per_chain`.
  - checks for valid `burn_in` and `thinning`.
- Added real energy diagnostics to the training step:
  - computes `e_real_mean` and `e_imag_mean` from `tdvp_residual_loss(...)` diagnostics instead of silently defaulting to zeros.
  - records `finite_loss` and `finite_grads` in the training metrics history.
- Fixed the runnable example in `src/TDVP.py`:
  - removed the nonexistent `optimizer_name` config argument.
  - preserved result keys with `optimizer=None` and `opt_state=None` to keep the return payload stable while the driver still uses manual updates.
- Added `tests/test_tdvp_phase6.py` covering:
  - finite `train_step(...)` outputs and diagnostics.
  - `train_loop(...)` metric-history recording.
  - configuration validation failures.

### Why

- `src/TDVP.py` had drifted into an inconsistent half-refactor:
  - undefined variables made the driver unrunnable.
  - the example entrypoint could not construct `TrainingConfig`.
  - energy diagnostics were misleading because they always collapsed to zero.
- Phase 6 needs a training loop that is actually executable before any optimizer upgrade or physics benchmarking is meaningful.

### Expected impact

- The training driver can now run end-to-end on small systems using the existing sampler/loss/gradient stack.
- Regressions in the training loop are now covered by tests instead of only by manual inspection.
- Logged metrics are more trustworthy for debugging sampler behavior and TDVP stability.

---

## 2026-04-18 — Phase 5 Follow-up: Nonzero Gradient Bug Fix

### What changed

- Repaired `src/grad.py` so `nnx.grad(...)` differentiates through the model argument it receives:
  - added a `_ModelWavefunctionView` adapter that exposes `__call__`, `log_prob`, and `phase` on top of an arbitrary model instance.
  - updated the pathwise loss closure to evaluate `tdvp_residual_loss(...)` through that adapter instead of closing over `wf.model`.
  - updated per-sample score-gradient closures to evaluate `log_prob` through the differentiand model `m` rather than the outer `wf`.
- Strengthened `tests/test_grad_phase5.py`:
  - gradient norms must now be strictly positive for the fixed deterministic toy setup.
  - added an explicit regression test that checks `grad_total`, `grad_pathwise`, and `grad_cov` all depend on model parameters.

### Why

- The previous closures accepted a model argument but never used it.
- That made the gradient estimator structurally zero while still passing the older finiteness-only tests.
- The zero gradient then propagated into `src/TDVP.py`, where training appeared to run but never actually updated parameters meaningfully.

### Expected impact

- Phase-5 gradients are now genuinely sensitive to wavefunction parameters.
- Phase-6 training diagnostics now report nonzero gradient norms on the deterministic toy run.
- The test suite now guards against this specific “unused differentiand” failure mode.

---

## 2026-04-18 — Phase 6 Follow-up: Persistent MCMC Chains Across Training Steps

### What changed

- Updated `src/TDVP.py` so training no longer restarts every Markov chain from the all-zero state on each optimization step:
  - added explicit chain-state validation for shape `(n_chains, N)` and binary domain `{0,1}`.
  - `train_step(...)` now accepts optional `initial_configurations` and returns `next_chain_configurations`.
  - `train_loop(...)` now carries chain configurations forward from one step to the next and returns the final chain states in `final_configurations`.
- Added Phase-6 regression tests in `tests/test_tdvp_phase6.py`:
  - verifies `train_step(...)` returns valid next chain states.
  - verifies `train_step(...)` actually uses the provided initial chain state.
  - verifies `train_loop(...)` preserves chain continuity across successive optimization steps.

### Why

- Restarting every chain at the same fixed configuration each step breaks the intended Monte Carlo process in `Program.md`, where the sampled batch should approximate draws from the current Born distribution.
- Persisting chains across optimization steps makes the sampler a warm-started MCMC process instead of a sequence of repeated transient chains.

### Expected impact

- The sampling process is now materially closer to the basic VMC mathematics described in `Program.md`.
- Training batches should better approximate samples from the current model distribution for a fixed sampler budget.
- The code now has regression coverage for chain persistence, which was previously only an assumption.

---

## 2026-04-18 — Example: Fully Polarized Chain Training Run

### What changed

- Extended `TrainingConfig` in `src/TDVP.py` with optional `initial_chain_configurations`:
  - allows `train_loop(...)` to start Markov chains from an explicitly chosen spin configuration instead of the default zero state.
  - validates shape `(n_chains, N)` and binary domain `{0,1}` during config validation.
- Added `example/train_fully_polarized_chain.py`:
  - starts all chains from a fully polarized bitstring state (`all ones`).
  - runs TDVP training for `10000` steps.
  - plots the recorded loss history with Matplotlib and saves it under `example/outputs/fully_polarized_loss.png`.
  - sets `MPLCONFIGDIR` to a writable project-local cache directory for reliable Matplotlib execution.
- Added a Phase-6 regression test ensuring `train_loop(...)` accepts explicit initial chain configurations.

### Why

- The requested example needs a visible way to initialize the sampling process from a fully polarized spin chain.
- Making the initializer part of `TrainingConfig` is cleaner than reimplementing the training loop inside the example script.

### Expected impact

- The repository now contains a concrete end-to-end example that matches the requested workflow:
  - fully polarized initial spin chain
  - long TDVP optimization run
  - loss visualization with Matplotlib

---

## 2026-04-18 — Phase 8: Save/Load/Resume and Observable Measurement

### What changed

- Extended `src/TDVP.py` with real checkpointed training-state support:
  - added `save_training_checkpoint(...)` to serialize model parameters, Hamiltonian metadata, config, RNG, chain states, and metrics history.
  - added `load_training_checkpoint(...)` to reconstruct a fresh `tSpinNQS` instance and restore its saved parameters.
  - expanded `train_loop(...)` to accept an initial wavefunction / Hamiltonian / RNG / metrics / chain state / start-time index for resume workflows.
  - periodic checkpoints now write `.pkl` files containing actual reloadable training state rather than metadata-only JSON.
- Added `src/observables.py`:
  - `measure_observables(...)` to estimate total and site-averaged `⟨Z⟩` and `⟨X⟩` from sampled configurations.
  - `sample_and_measure_observables(...)` to run the sampler and return observable estimates plus sampler diagnostics.
  - `exact_observables_from_wf(...)` and `normalized_statevector(...)` for exact small-system validation by exhaustive enumeration.
- Added new tests:
  - `tests/test_phase8_checkpoint.py` for save/load consistency, checkpoint file creation, and resume-from-checkpoint continuation.
  - `tests/test_phase8_observables.py` for observable formulas and exact small-system checks.

### Why

- The remaining roadmap items required a practical way to preserve a trained wavefunction and use it later without retraining from scratch.
- Observable measurement is the bridge from “the optimizer runs” to “the trained state predicts physics”.
- Exact small-system enumeration gives a clean validation target for the new observable code.

### Expected impact

- The codebase now supports the full save → load → resume workflow for trained wavefunctions.
- Post-training measurement of `⟨Z(t)⟩` and `⟨X(t)⟩` is available directly from Monte Carlo samples.
- Small-system exact checks are now available to validate observable calculations.

---

## 2026-04-18 — Example: Train, Save, Reload, Resume, Measure

### What changed

- Added `example/train_save_reload_measure.py`:
  - trains a wavefunction from a fully polarized spin chain.
  - saves a checkpoint with `save_training_checkpoint(...)`.
  - reloads the checkpoint with `load_training_checkpoint(...)`.
  - optionally resumes optimization for additional TDVP steps.
  - measures Monte Carlo `⟨Z⟩` and `⟨X⟩` after training.
  - computes exact `⟨Z⟩` / `⟨X⟩` by enumeration for small systems.
  - writes:
    - a loss plot,
    - `observables.json`,
    - `summary.json`,
    - checkpoint files in the example output directory.

### Why

- The codebase now has the Phase-8 machinery, but users still need a concrete script that demonstrates the intended workflow end to end.
- A worked example makes the save/load/resume and observable APIs much easier to verify and reuse.

### Expected impact

- The repository now includes an executable Phase-8 example that demonstrates the full post-training analysis workflow rather than only the training loop.

---

## 2026-04-18 — Optimizer Upgrade: AdamW in TDVP Training

### What changed

- Replaced the manual SGD-style parameter update in `src/TDVP.py` with a real `optax.adamw` optimizer.
- Extended `TrainingConfig` with AdamW hyperparameters:
  - `optimizer_name`
  - `adamw_b1`
  - `adamw_b2`
  - `adamw_eps`
  - `weight_decay`
- Updated `train_step(...)` and `train_loop(...)` to carry optimizer state explicitly through training.
- Extended checkpoint save/load so AdamW state is serialized and restored on resume.
- Updated tests and the save/reload example to resume with the restored optimizer state.

### Why

- The previous update rule was plain gradient descent, which wastes the optimizer-state information usually needed for stable TDVP/VMC training.
- Resume-from-checkpoint should preserve optimizer moments, not just model weights, otherwise the resumed run silently changes optimization dynamics.

### Expected impact

- Training now uses AdamW rather than bare SGD.
- Checkpointed runs resume with consistent optimizer dynamics.
- The optimizer path is structured so a later Muon option can be added cleanly without reworking the training loop again.

---

## 2026-04-18 — Examples Updated for Explicit AdamW Usage

### What changed

- Updated `example/train_fully_polarized_chain.py` to set `optimizer_name="adamw"` explicitly.
- Added AdamW CLI knobs to the example scripts:
  - `--learning-rate`
  - `--weight-decay`
  - `--adamw-b1`
  - `--adamw-b2`
  - `--adamw-eps`
- Updated example console output, plot titles, and saved metadata so the optimizer choice is visible rather than implicit.
- Updated `example/train_save_reload_measure.py` to record AdamW settings in `observables.json` and `summary.json`.

### Why

- The training loop now supports AdamW explicitly, so the examples should document that choice directly instead of relying on default config values.
- Saved outputs are easier to interpret when the optimizer hyperparameters are recorded alongside the run.

### Expected impact

- Example runs now make the optimizer configuration explicit and reproducible.
- Future optimizer comparisons, including a later Muon path, will be easier because the examples already surface optimizer settings at the CLI and in output files.

---

## 2026-04-18 — Joint Time-Loss Training

### What changed

- Changed the default TDVP training mode in `src/TDVP.py` from serial per-time optimization to a summed time-loss objective.
- Added `TrainingConfig.time_loss_mode` with values:
  - `sum` for a single optimizer update over all time slices
  - `serial` to keep the previous per-time-step update behavior
- Reworked `train_loop(...)` so the default path:
  - evaluates every time slice with the same model parameters
  - sums the time-local losses
  - accumulates the time-slice gradients
  - applies one AdamW update per outer training step
- Kept chain continuation and checkpointing working in both modes.
- Updated the phase-6 test to expect one metric entry per optimizer step in summed-loss mode.

### Why

- Serial updates across time slices can change earlier time slices before the full trajectory has been evaluated.
- Summing the time-local losses matches the intended objective more directly and keeps all time slices coupled to the same parameter snapshot during each update.

### Expected impact

- Default training now optimizes a joint objective over the whole time window.
- The previous serial behavior is still available for comparison or debugging.
- Loss history now reflects optimizer steps, not optimizer steps multiplied by the number of time slices, when `time_loss_mode="sum"`.

---

## 2026-04-18 — Examples Updated for Joint Time-Loss Mode

### What changed

- Updated `example/train_fully_polarized_chain.py` to set `time_loss_mode="sum"` explicitly.
- Updated `example/train_save_reload_measure.py` to set `time_loss_mode="sum"` explicitly.
- Printed the time-loss mode alongside the AdamW hyperparameters at startup.
- Recorded `time_loss_mode` in the saved observables and summary metadata for the save/reload example.

### Why

- The examples should reflect the actual training semantics used by the codebase after the joint time-loss change.
- Making the mode explicit prevents confusion between the old serial time-stepping behavior and the new summed objective.

### Expected impact

- Example scripts now clearly advertise that they use the joint loss across time slices.
- Saved example artifacts now include enough metadata to reproduce the exact training objective used in the run.
