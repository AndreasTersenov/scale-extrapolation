# 2026-07-10 — Phase-1 kickoff + environment decision

## Hypothesis
The break-and-repair experiment (P4/P5/P6/P13/P-null) can be built as a
wavelet-factorized conditional flow-matching generator reusing `jax_flows`, climbing the
Karpathy ladder rung-by-rung, with `scaledrift` reused UNMODIFIED as the measurement
instrument.

## Environment decision (setup, not a PLAN deviation)
Two envs, clean boundary — nothing installed into the frozen instrument env:
- **`~/venvs/scale-extrap`** (`source env.sh`): `scaledrift` instrument + its Stop-hook
  gate. UNCHANGED — it is the frozen instrument (backpressure test d).
- **`~/wl-challenge-env`** (existing, user's): the JAX stack — jax 0.8.2, flax 0.12.5,
  optax 0.2.6, jaxtyping, tqdm, matplotlib, pytest 9.1.1. Used read-only for phase-1
  training/generation and the phase-1 JAX tests (`tests_wfm/`). `jax_flows` imports fine.

Non-obvious gotchas found:
- On the login node JAX aborts at CPU-client init: `pthread_create() failed` (EAGAIN) —
  XLA sizes its Eigen threadpool to all 192 cores and hits the shared process/thread
  limit. Fix: pin CPU affinity to a few cores BEFORE importing jax
  (`os.sched_setaffinity(0, {0,1,2,3})`, done in `tests_wfm/conftest.py`; `taskset -c 0-3`
  for ad-hoc runs). The `CUDA_ERROR_NO_DEVICE` line above it is benign (no GPU on login).
- `wl-challenge-env` has no `pywt`; so the generator carries its OWN JAX Haar transform
  (`wfm/haar.py`, round-trip gate-tested) and evaluation of generated fields is done in
  the scale-extrap env with `scaledrift`. Data bridges the two envs as `.npz`.
- diffrax on this cluster is broken vs jax≥0.9; use the euler/heun `lax.scan` samplers
  from `jax_flows` (already the default there).

## Plan (Karpathy ladder — each rung green + committed before the next)
(i) single-octave conditional FM overfits one field → (ii) two-octave recursion →
(iii) GRF end-to-end null (P-null) → (iv) arms A/B on gowerstreet (P4/P5/P6) →
(v) transfer (P13). GPU only via SLURM (def-lplevass CPU / rrg-lplevass GPU), each job
logged pre-submission with config hash + expected outcome.

## New package `wfm/` (wavelet flow matching) — separate from `scaledrift`
`haar.py` (JAX Haar DWT/iDWT), `model.py` (conditional velocity UNet: coarse channels +
optional 2-D scale-coord à la arm B), `cfm.py` (conditional CFM loss + sampler, reusing
`jax_flows.ot_interpolate`), `octaves.py` (per-octave coarse/detail pairs).

## Result / belief
- **Rung (i) GREEN + committed.** Single-octave conditional FM overfits one gowerstreet
  field: octave-2 detail (32²×3), tiny UNet (48,96), 2000 CFM steps (~40 s CPU/4-core),
  sampled-vs-true detail relative L2 = **0.081** (< 0.15 gate). Backpressure a/b/c green:
  Haar round-trip < 1e-5 (float32) + energy-preserving, sampler/recursion-step determinism
  bit-exact, overfit gate executable. `tests_wfm/` (8 tests, 57 s) run in wl-challenge-env;
  `scaledrift` suite untouched and green (backpressure d).
- Belief: the FM core + conditional UNet + JAX Haar recursion atoms all work. Proceed to
  rung (ii) two-octave recursion.
- **Rung (ii) GREEN + committed.** A SINGLE weight-tied model overfit on octaves 1 & 2 of
  one gowerstreet field; full field generated coarse-to-fine from its true coarsest coarse
  (sample detail|coarse → invert one Haar level → repeat). Recursive-vs-true field
  relative L2 = **0.069** (< 0.2 gate); recursion bit-exact deterministic under fixed seed.
  ~90 s CPU/4-core. `tests_wfm/` now 9 tests, ~126 s; scaledrift instrument still green.
- Belief: the recursion machinery (weight-tied multi-octave training + coarse-to-fine
  generation) is correct on one field. Next: rung (iii) GRF end-to-end null (P-null) —
  needs a multi-field trainer, extrapolated-resolution generation, and the scaledrift
  measurement bridge; that is the first rung wanting a GPU SLURM job.
