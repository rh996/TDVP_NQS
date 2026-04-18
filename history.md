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

## Validation snapshot (after Phase 1–3)

- Unit tests currently passing for implemented phases:
  - wavefunction phase-1 API tests
  - hamiltonian phase-2 logic tests
  - sampler phase-3 tests
- Current test status at update time:
  - all tests in suite passed (`18 passed`).

---

## Notes for next phase

- Next target: **Phase 4** (`src/loss.py`)
  - compute per-sample residual components `A_n`, `B_n`
  - use `local_energy_batch(...)`
  - output scalar centered-variance loss and useful diagnostics.
- Then **Phase 5** (`src/grad.py`)
  - implement pathwise gradient + covariance correction.