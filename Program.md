Project Goal and Theoretical Basis

Implementation Plan (Based on Current Codebase Status)

Current Status Scan

1. `src/wavefunction.py`
   - Implemented:
     - `Encoder` with token embedding, position embedding, and explicit time channel.
     - Attention-based stack (`TransformerLayer` + `BoxLayer`).
     - `tNQS` model with mean pooling and two MLP heads.
   - Current output contract:
     - `tNQS.__call__(configuration, t) -> (x1, x2)` where `x1` and `x2` are intended to map to `\log p` and `\phi`.
   - Gaps / risks:
     - `tSpinNQS` currently instantiates a new `tNQS` inside `__call__`, which re-creates parameters on each forward call. This will break training semantics and parameter persistence.
     - `Wavefunction` interface is still minimal and does not yet enforce explicit `log_prob` and `phase` naming.

2. `src/hamiltonian.py`
   - Implemented:
     - `TransverseIsingHamiltonian` dataclass and `local_energy`.
     - Off-diagonal spin flip ratio logic using model outputs.
   - Observed details:
     - Spin flip uses `1 - configuration[i]` (consistent with 0/1 encoding).
     - ZZ term currently uses nearest-neighbor open-chain bonds (`0..N-2`), not periodic boundary.
   - Gaps / risks:
     - Need explicit sign-convention check against the theory section (`J Σ σ_i σ_j`) and encoding choice (`0/1` vs `±1`) to ensure physical correctness.
     - `local_energy` should clearly document expected output semantics from `wf`.

3. `src/sampler.py`, `src/loss.py`, `src/grad.py`, `src/TDVP.py`
   - Status from scan:
     - Files exist in repository structure but currently need implementation (or are not yet functionally populated).
   - Impact:
     - End-to-end VMC/TDVP loop is not yet available.
     - Core roadmap should prioritize these modules in dependency order.

4. `README.md`
   - No actionable implementation guidance currently present.

Execution Plan

Phase 1 — Stabilize Wavefunction API and Parameter Lifecycle
1. Define and lock API contract:
   - `wf(configuration, t)` returns `(logp, phi)` with shape `(batch, 1)` or `(batch,)` consistently.
2. Refactor `tSpinNQS`:
   - Hold a persistent `tNQS` module instance in `__init__`.
   - Avoid re-instantiation in `__call__`.
3. Add helper methods (thin wrappers):
   - `log_prob(configuration, t)`
   - `phase(configuration, t)`
4. Add shape checks and type checks at the wavefunction boundary.

Deliverable:
- A trainable, persistent NNX model wrapper with stable outputs for downstream modules.

Phase 2 — Hamiltonian Correctness and Tests
1. Clarify spin convention:
   - Decide and document whether physical spin variable in formulas is derived from binary state (`s = 1 - 2*bit`) or represented directly.
2. Align ZZ implementation with selected convention and sign.
3. Add boundary-condition flag:
   - `open` (default) and optional `periodic`.
4. Add unit checks for `local_energy` on tiny systems (`N=2,3`) against brute-force calculations.

Deliverable:
- Verified `local_energy` that is convention-consistent and test-backed.

Phase 3 — Sampler Implementation (`src/sampler.py`)
1. Implement single-spin Metropolis-Hastings on bitstrings.
2. Target distribution:
   - proportional to `exp(logp(configuration, t))` (if model returns log probability directly).
3. Add:
   - burn-in
   - thinning
   - acceptance-rate reporting
   - vectorized multi-chain sampling with `vmap` where practical.
4. Ensure JAX-compatible RNG handling.

Deliverable:
- Deterministic-reproducible sampler returning a batch of configurations and diagnostics.

Phase 4 — Loss Implementation (`src/loss.py`)
1. Implement per-sample neighbor construction (`σ^(i)` for all i).
2. Compute:
   - `Δ_i log p`, `Δ_i φ`
   - real/imag local energy components
   - `A_n`, `B_n`
3. Compute centered variance loss:
   - `\hat L = mean((A-\bar A)^2 + (B-\bar B)^2)`.
4. Add numerical safety:
   - clip exponent arguments
   - dtype consistency
   - optional debug stats.

Deliverable:
- Pure function to evaluate local residual statistics and scalar loss from sampled batch.

Phase 5 — Gradient with VMC Correction (`src/grad.py`)
1. Pathwise term:
   - autodiff of per-sample loss with samples treated fixed.
2. Sampling-measure correction:
   - covariance term with score function `∂_θ log p`.
3. Compose full gradient estimator:
   - `grad = pathwise + covariance`.
4. Add gradient sanity checks:
   - finite norms
   - agreement with finite differences on toy settings (small N).

Deliverable:
- Unbiased practical gradient estimator aligned with the theory section.

Phase 6 — Training Driver (`src/TDVP.py`)
1. Initialize model, optimizer, RNG streams, Hamiltonian, and schedule over time points.
2. Loop:
   - sample → loss stats → gradient estimate → optimizer step.
3. Log:
   - loss
   - acceptance rate
   - gradient norm
   - energy components
4. Add checkpointing and restart support.

Deliverable:
- Runnable script/function for end-to-end optimization on small transverse Ising systems.

Phase 7 — Validation and Benchmarking
1. Small-system validation (`N<=10`):
   - compare energies / observables with exact diagonalization at selected times (if available).
2. Ablations:
   - number of boxes
   - heads / head_dim
   - sampler chain count and burn-in
3. Stability checks:
   - multiple seeds
   - clipping thresholds
   - optimizer settings.

Deliverable:
- Confidence that implementation is numerically stable and physically consistent.

Phase 8 — Checkpointed Wavefunction Reuse and Observable Measurement
1. Save trained neural-network state:
   - serialize the wavefunction parameters after training.
   - save enough metadata to reconstruct the model architecture and Hamiltonian settings.
2. Resume / reload wavefunction:
   - load saved parameters into a fresh `tSpinNQS` instance.
   - support continuing training from a checkpointed wavefunction state.
3. Measure observables after training:
   - estimate `⟨Z(t)⟩` and/or `⟨X(t)⟩` from Monte Carlo samples drawn from the trained wavefunction.
   - define clearly whether `⟨Z(t)⟩` means site-averaged magnetization, total magnetization, or another convention.
4. Add tests / validation:
   - save → load → forward-pass consistency checks.
   - observable sanity checks on small systems or analytically simple wavefunctions.

Deliverable:
- A workflow that can train, save, reload, resume, and evaluate physically meaningful observables such as `⟨Z(t)⟩` and `⟨X(t)⟩`.

Immediate Next Tasks (Concrete Priority Queue)

1. Fix `tSpinNQS` parameter persistence bug (highest priority).
2. Finalize and document spin/sign convention in `hamiltonian.py`.
3. Implement `sampler.py` minimal Metropolis version.
4. Implement `loss.py` for `A_n`, `B_n`, and `\hat L`.
5. Implement `grad.py` with pathwise + covariance estimator.
6. Wire everything in `TDVP.py` and run a minimal training experiment.
7. Add checkpoint save/load support for the trained wavefunction.
8. Implement post-training measurement of observables such as `⟨Z(t)⟩` and `⟨X(t)⟩`.

Definition of Done (Minimal Objective Aligned)

The first milestone is complete when the code can:

1. Sample configurations from the model-induced Born distribution.
2. Compute `A_n`, `B_n`, and `\hat L(t)` on sampled batches.
3. Compute parameter gradients using pathwise + covariance correction.
4. Perform optimizer updates for multiple steps without shape/NaN failures.
5. Reproduce sane trends on a small transverse-field Ising test case.

Change Management Requirements

For each implementation step:
1. Record changes in `history.md`:
   - what changed
   - why
   - expected numerical/physics impact.
2. Commit to git with focused messages by module:
   - `wavefunction`
   - `hamiltonian`
   - `sampler`
   - `loss`
   - `grad`
   - `TDVP`.

Goal

This project aims to implement a variational Monte Carlo framework for optimizing a time-dependent neural quantum state (NQS) by minimizing a physically meaningful loss function derived from the Schrödinger equation.

The concrete target is the transverse-field Ising model, using a neural wavefunction of the form

\Psi_\theta(\sigma,t)=\sqrt{p_\theta(\sigma,t)}\,e^{i\phi_\theta(\sigma,t)}.

The implementation should support:

1. Monte Carlo sampling from the Born distribution p_\theta(\sigma,t) \propto |\Psi_\theta(\sigma,t)|^2.
2. Evaluation of the time-local loss using sampled configurations.
3. Evaluation of the gradient of the loss with respect to network parameters.
4. Optimization of the parameters with autodiff for the pathwise term, plus the standard VMC correction from the sampling measure.

The project is focused on making the loss and its gradient explicit in terms of:

* \log p_\theta(\sigma,t)
* \phi_\theta(\sigma,t)
* spin-flip neighbors of each sampled configuration

This avoids relying on more abstract operator notation during implementation.

Basic Theory

1. Wavefunction parametrization

We write the neural quantum state as

\Psi_\theta(\sigma,t)=\sqrt{p_\theta(\sigma,t)}\,e^{i\phi_\theta(\sigma,t)}.

Then

\log \Psi_\theta(\sigma,t)=\frac12\log p_\theta(\sigma,t)+i\phi_\theta(\sigma,t).

Its time derivative is

\partial_t \log \Psi_\theta(\sigma,t)=\frac12\partial_t \log p_\theta(\sigma,t)+i\partial_t \phi_\theta(\sigma,t).

2. Model Hamiltonian

For the transverse-field Ising model,

H=J\sum_{\langle i,j\rangle}\sigma_i^z\sigma_j^z+h\sum_i \sigma_i^x.

In the computational z-basis, the diagonal contribution is

E_{ZZ}(\sigma)=J\sum_{\langle i,j\rangle}\sigma_i\sigma_j,

and the off-diagonal X term connects \sigma to configurations \sigma^{(i)} obtained by flipping one spin.

3. Local energy

The local energy is

E_{\mathrm{loc}}(\sigma,t)=\frac{\langle \sigma|H|\Psi_\theta(t)\rangle}{\Psi_\theta(\sigma,t)}.

For the transverse-field Ising model,

E_{\mathrm{loc}}(\sigma,t)=J\sum_{\langle i,j\rangle}\sigma_i\sigma_j+h\sum_i \frac{\Psi_\theta(\sigma^{(i)},t)}{\Psi_\theta(\sigma,t)}.

Using the amplitude-phase form,

\frac{\Psi_\theta(\sigma^{(i)},t)}{\Psi_\theta(\sigma,t)}
=\exp\!\left[\frac12\Delta_i\log p(\sigma,t)+i\Delta_i\phi(\sigma,t)\right],

where

\Delta_i\log p(\sigma,t)=\log p_\theta(\sigma^{(i)},t)-\log p_\theta(\sigma,t),
\Delta_i\phi(\sigma,t)=\phi_\theta(\sigma^{(i)},t)-\phi_\theta(\sigma,t).

Therefore,

E_{\mathrm{loc}}(\sigma,t)=J\sum_{\langle i,j\rangle}\sigma_i\sigma_j+h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}e^{i\Delta_i\phi(\sigma,t)}.

So the real and imaginary parts are

\Re E_{\mathrm{loc}}(\sigma,t)=J\sum_{\langle i,j\rangle}\sigma_i\sigma_j+h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}\cos\bigl(\Delta_i\phi(\sigma,t)\bigr),

\Im E_{\mathrm{loc}}(\sigma,t)=h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}\sin\bigl(\Delta_i\phi(\sigma,t)\bigr).

4. Local residual and loss

Define the local residual

L_{\mathrm{loc}}(\sigma,t)=\partial_t\log\Psi_\theta(\sigma,t)+iE_{\mathrm{loc}}(\sigma,t).

Writing

L_{\mathrm{loc}}(\sigma,t)=A(\sigma,t)+iB(\sigma,t),

we obtain

A(\sigma,t)=\frac12\partial_t\log p_\theta(\sigma,t)-h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}\sin\bigl(\Delta_i\phi(\sigma,t)\bigr),

B(\sigma,t)=\partial_t\phi_\theta(\sigma,t)+J\sum_{\langle i,j\rangle}\sigma_i\sigma_j+h\sum_i e^{\frac12\Delta_i\log p(\sigma,t)}\cos\bigl(\Delta_i\phi(\sigma,t)\bigr).

The time-local loss is the variance of the residual under the Born distribution:

L(t)=\operatorname{Var}_{p_\theta}(A)+\operatorname{Var}_{p_\theta}(B).

With Monte Carlo samples \sigma_n\sim p_\theta(\sigma,t), define

A_n=A(\sigma_n,t),\qquad B_n=B(\sigma_n,t),

\bar A=\frac1N\sum_{n=1}^N A_n,\qquad \bar B=\frac1N\sum_{n=1}^N B_n,

\ell_n=(A_n-\bar A)^2+(B_n-\bar B)^2.

Then the Monte Carlo estimator of the loss is

\hat L(t)=\frac1N\sum_{n=1}^N \ell_n.

5. Gradient of the loss

Let \theta_k be one variational parameter. The gradient contains two pieces:

1. A pathwise derivative term obtained by autodiff on fixed samples.
2. A sampling-distribution correction term from VMC.

The correct estimator is

\partial_{\theta_k} L
=\mathbb E_{p_\theta}\bigl[\partial_{\theta_k}\ell(\sigma,t)\bigr]
+\mathbb E_{p_\theta}\bigl[\ell(\sigma,t)\,\partial_{\theta_k}\log p_\theta(\sigma,t)\bigr].

Using the fact that

\mathbb E_{p_\theta}[\partial_{\theta_k}\log p_\theta]=0,

this can be written as a covariance form,

\partial_{\theta_k} L
=\mathbb E_{p_\theta}\bigl[\partial_{\theta_k}\ell(\sigma,t)\bigr]
+\operatorname{Cov}_{p_\theta}\bigl(\ell(\sigma,t),\partial_{\theta_k}\log p_\theta(\sigma,t)\bigr).

With samples, the practical estimator is

\partial_{\theta_k}\hat L
\approx
\frac1N\sum_{n=1}^N \partial_{\theta_k}\ell_n
+\frac1N\sum_{n=1}^N (\ell_n-\bar\ell)\left(\partial_{\theta_k}\log p_n-\overline{\partial_{\theta_k}\log p}\right),

where

\bar\ell=\frac1N\sum_{n=1}^N \ell_n.

The per-sample derivative is

\partial_{\theta_k}\ell_n
=2(A_n-\bar A)(\partial_{\theta_k}A_n-\partial_{\theta_k}\bar A)
+2(B_n-\bar B)(\partial_{\theta_k}B_n-\partial_{\theta_k}\bar B),

with

\partial_{\theta_k}\bar A=\frac1N\sum_m \partial_{\theta_k}A_m,
\qquad
\partial_{\theta_k}\bar B=\frac1N\sum_m \partial_{\theta_k}B_m.

The needed derivatives are

\partial_{\theta_k}A_n
=\frac12\partial_{\theta_k}\partial_t\log p_n
-h\sum_i \partial_{\theta_k}\left(e^{\frac12\Delta_i\log p_n}\sin(\Delta_i\phi_n)\right),

\partial_{\theta_k}B_n
=\partial_{\theta_k}\partial_t\phi_n
+h\sum_i \partial_{\theta_k}\left(e^{\frac12\Delta_i\log p_n}\cos(\Delta_i\phi_n)\right).

Expanding by the chain rule,

\partial_{\theta_k}\left(e^{\frac12\Delta_i\log p}\sin(\Delta_i\phi)\right)
=e^{\frac12\Delta_i\log p}
\left[
\frac12\partial_{\theta_k}(\Delta_i\log p)\sin(\Delta_i\phi)
+\cos(\Delta_i\phi)\partial_{\theta_k}(\Delta_i\phi)
\right],

\partial_{\theta_k}\left(e^{\frac12\Delta_i\log p}\cos(\Delta_i\phi)\right)
=e^{\frac12\Delta_i\log p}
\left[
\frac12\partial_{\theta_k}(\Delta_i\log p)\cos(\Delta_i\phi)
-\sin(\Delta_i\phi)\partial_{\theta_k}(\Delta_i\phi)
\right].

Practical Implementation Notes

* Sample configurations from |\Psi_\theta(\sigma,t)|^2.
* Treat samples as fixed during autodiff.
* Use autodiff to compute the pathwise term \frac1N\sum_n \partial_{\theta_k}\ell_n.
* Add the covariance correction involving \partial_{\theta_k}\log p_n.
* Do not rely only on autodiff of the batch mean loss on fixed samples, because that misses the derivative of the sampling distribution and gives a biased VMC gradient.

Intended Project Structure

A reasonable first version of the project may include:

* wavefunction.py: NQS definition returning \log p and \phi
* sampler.py: MCMC sampler for |\Psi|^2
* hamiltonian.py: transverse-field Ising connectivity and diagonal terms
* loss.py: evaluation of A_n, B_n, and \hat L
* grad.py: pathwise gradient plus covariance correction
* train.py: optimization loop over time points or time windows
* TDVP.py: the main function of creating input parameters and running the simulation 

Minimal Objective

The minimal working goal is:

1. Generate samples \sigma_n.
2. Compute A_n, B_n, and \hat L.
3. Compute \partial_\theta \hat L using autodiff plus the VMC correction.
4. Update parameters with an optimizer such as Adam.
5. Verify the implementation on a small transverse-field Ising system.

Summary

This project is about implementing a VMC-compatible optimization procedure for a time-dependent neural quantum state. The central idea is to express the Schrödinger residual loss directly in terms of \log p and \phi, evaluate it with Monte Carlo samples, and compute its gradient as the sum of:

* a pathwise autodiff term
* a covariance correction from the parameter dependence of the sampling distribution

That gives a practical and theoretically consistent route to optimize the NQS within the VMC framework.


Programming Style

* use JAX and flax to forward and backward the neural network
* do not use legacy APIs in flax.linen, and rather use flax.nnx as in exisited code base
* write what has been added and changed in history.md and git commit.
