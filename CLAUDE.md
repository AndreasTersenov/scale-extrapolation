# CLAUDE.md — scale-extrapolation

**Read `HANDOFF.md` before doing anything else.** It carries the binding
state of the project, the operating rules, and the numbered task queue.
Then read `PROMPTS.md` if you were given a task ID (T1–T6).

This project is in **wind-down / low-token mode** (since 2026-08-08). The
experimental campaign is closed; the remaining work is certification,
writing, and one final pre-registered run. Optimize for doing ONE task
well and cheaply, not for exploring.

## Operating rules (binding — see HANDOFF.md §4 for the reasoning)

1. **One task per session**, taken from HANDOFF.md §5 by its ID. Do not
   plan a campaign, do not chain tasks, stop when the task is done.
2. **No subagents, no workflows, no parallel fan-outs.** Not a style
   preference — a cost rule.
3. **No new experiments.** If you think of one, append it to
   `IDEAS-PARKED.md` with a date and a paragraph. That is the whole
   protocol; it is not a lesser outcome.
4. **Numbers only by verbatim copy** from a committed artifact, carrying a
   `<!-- src: file -->` pointer. Never from memory, never recomputed in
   prose. `NUMBERS.md` holds the canonical ones for the paper.
5. **JUDGE-2 (persistent homology) is quarantined** — it is applied to NO
   generated map until the single blind shot (T3). This is absolute.
6. **Pre-registration still applies to T3 and T5 only** (anything that
   produces a paper claim from a new run). Writing and measurement on
   committed artifacts do not need it.

## Verification

```bash
./check.sh          # both test stacks, correct interpreters, cpu-pinned
```

Run it before committing anything that touches code. The Stop hook runs
the same gates automatically, but **only when executable files are
dirty** — writing turns skip it (deliberate: the unconditional version
burned minutes of login-node CPU per turn for nothing).

## Environment / cluster

- Two interpreter stacks, do not mix them:
  - `tests/`, `scripts_p2/` analysis → `source env.sh` (numpy/scipy/pywt)
  - `tests_wfm/`, `tests_p2/`, anything JAX → `~/wl-challenge-env/bin/python`
- **Login-node rule:** never run pytest, training, or map-wide numerics
  un-pinned. Use `taskset -c 0-3` (check.sh does this) or SLURM. Real
  compute goes through `sbatch`; MIG `h100_20gb` with `--time<=2:59` on
  `rrg-lplevass` starts fastest. CPU jobs → `def-lplevass`.
- Data: `/project/rrg-lplevass/shared/wl_chall_data/` (compute nodes only).
  Large intermediates → `$SCRATCH`; committed artifacts → `results_p2/`.
- Log SLURM jobs in `JOBS.md` before submitting (ID, config, expectation).

## Where things are

| I need… | Read |
|---|---|
| current state, rules, task queue | `HANDOFF.md` |
| the exact prompt for a task | `PROMPTS.md` |
| a number for the paper | `NUMBERS.md` (then its src artifact) |
| the science explained in prose | `PROJECT-EXPLAINER.md` |
| the paper draft | `paper/` (00→07 + `theory-w2.md` when written) |
| research directions, ranked | `BRIEF-foundations.md` |
| why a decision was made | `log/` — the binding ones are R43, R47, R48, R49 |
| historical campaign docs | `docs/archive/` |

`log/` holds ~50 dated rulings and readouts. You almost never need to read
more than the four binding ones listed above; HANDOFF.md §2 summarizes
what they concluded.

## Conventions that still hold

- Commit after every meaningful unit; push (remote is configured).
- Every scripted find/replace is followed by a `grep` verifying it landed.
- Tests-first for any NEW estimator (this still applies in T1/T5).
- Log entries: `log/YYYY-MM-DD-<slug>.md`, hypothesis → setup →
  expectation → result → what changed in our belief.
