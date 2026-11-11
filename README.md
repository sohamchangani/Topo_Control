# Shape the Attractor: Topological Regularization for Reliable Closed-Loop Synthesis

Code for the paper **"Shape the Attractor: Topological Regularization for Reliable Closed-Loop Synthesis"**
by Soham Changani\*, Yubaraj Neupane\*, Baris Coskunuzer† and Suat Gumussoy† (UT Dallas).
\* Equal contribution. † Co-senior authors.

We learn neural feedback controllers that bring trajectories from many initial conditions onto **one common
periodic orbit**, without prescribing that orbit. The closed loop is simulated from many initial conditions, the
transient is discarded, and the remaining states are pooled into one point cloud. A differentiable
persistent-homology regularizer rewards the widest range of scales over which this cloud forms **one connected
component with exactly one loop** (the *connected one-loop plateau*). The regularizer needs no reference orbit,
energy, period or Poincaré section, and it can be used alone or added to conventional control losses.

Persistent homology is used **only in training**. Success is judged by independent dynamical validators on
unseen initial conditions.

## Results at a glance

Each objective is trained with 10 independent seeds. A controller **passes** if all 32 screening initial
conditions settle onto one common periodic orbit, and is **perfect** if it also settles all 320 initial conditions
of 10 unseen retest batches.

| System | Task-only (perfect) | Task + topology (perfect) | Best combination |
|---|---|---|---|
| Double pendulum (Acrobot) | 2/30 | 7/30 | Topology alone, Topology + Range (3/10 each) |
| Kuramoto–Sivashinsky (L = 22) | 4/20 | 10/20 | Topology + Recurrence (7/10, all 2240 retest arms settled) |

The gains depend on the task loss and the training budget; see Section 4.4 and Appendix C.4 of the paper.

## Repository layout

```text
main_acrobot.py                  double pendulum: train, screen and retest one controller; summary table
main_ks.py                       Kuramoto–Sivashinsky: setup, train, validate, retest, search, summary
submit.sh                        Slurm job for the full KS experiment (all objectives)
topo_attractor/
  acrobot/
    plant.py                     dynamics, energy, initial conditions, RK4 closed-loop rollout
    controller.py                MLP policy and the shared energy-regulation warm start
    losses.py                    energy, range, soft Poincaré section, loop-only plateau (topology)
    training.py                  objectives and training loop
    validator.py                 hard Poincaré validator
    evaluation.py                screening on 32 held-out initial conditions, plots, CSV output
    retest.py                    retest of passing controllers, pass / perfect summary
  ks/
    config.py                    all settings (Cfg); every field is a command-line flag of main_ks.py
    plant.py                     ETDRK4 solver, actuators, sensors, closed-loop rollout
    policies.py                  MLP policy on point sensors; open-loop reference policies
    initial_conditions.py        attractor pools and low-mode fields; training / validation / retest sets
    losses.py                    connected one-loop plateau (topology), energy, recurrence
    validator.py                 return-error and Hausdorff validator, Lyapunov diagnostic, PH certificate
    reference.py                 reference values for the baselines (E*, v_ref, P*)
    experiment.py                training, validation, best checkpoint, retest, search, summary
    plots.py                     diagnostic plots
```

## How it works

1. **Simulate.** Roll out the current policy from `M = 8` training initial conditions.
2. **Discard the transient.** Drop the burn-in and map the remaining states to a fixed representation `z`:
   - Acrobot: `(cos q1, sin q1, cos q2, sin q2, q̇1/4, q̇2/4)` in R⁶.
   - KS: real and imaginary parts of the first 15 normalized Fourier modes, in R³⁰.
3. **Pool.** Subsample `N = 96` points from all trajectories into one cloud.
4. **Persistent homology.** Compute the Vietoris–Rips persistence diagrams with
   [Oineus](https://github.com/anigmetov/oineus) from the full pairwise-distance matrix. Gradients flow through the
   persistence endpoints back into the closed-loop simulation.
5. **Plateau loss.**
   - KS uses the connected plateau `R = max_i [e_i − max(s_i, r_conn)]` (one component *and* one loop), and keeps its
     signed value during training.
   - The Acrobot uses the loop-only plateau `R1 = max_i (e_i − s_i)`.

   The training loss is `L = λ_task·L_task + λ_topo·(−R) + λ_u·L_u`.

## Installation

```bash
git clone <this repository> && cd <repository>
pip install -r requirements.txt
```

Run every command from the repository root. All computations run on the **CPU** in double precision; no GPU is used.

## Double pendulum (Acrobot)

Two-link arm with torque at the elbow only, friction `b = 0.30`, `|u| ≤ 8`, RK4 with `Δt = 0.02`.

**Objectives** (`--mode`):
- `energy_only`, `range_only`, `poincare_only`
- `topology_only`
- `topology_energy`, `topology_range`, `topology_poincare`

**One run** trains a controller for 150 epochs on 8 fixed training initial conditions (600 training steps). It then
screens the final controller on 32 held-out initial conditions (1200 test steps). If all 32 settle, it retests the
controller, weights frozen, on 10 shared batches of 32 new initial conditions.

```bash
python main_acrobot.py --mode topology_only --seed 0
```

**Main experiment (Table 1).** Topology objectives use a burn-in of 300 training and 600 test steps (the defaults);
task-only objectives use none.

```bash
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for mode in topology_only topology_energy topology_range topology_poincare; do
    python main_acrobot.py --mode $mode --seed $seed
  done
  for mode in energy_only range_only poincare_only; do
    python main_acrobot.py --mode $mode --seed $seed --burn_in 0 --eval_burn_in 0
  done
done
python main_acrobot.py --summarize          # pass / perfect / settled retest arms -> runs_acrobot/summary.md
```

Each run writes `runs_acrobot/<mode>_b0.3_seed<k>/`:
- `controller.pt`: weights and all settings
- `summary.json`: screening results; `poincare_success_rate` = 1.0 means the controller passes
- `retest.json`: retest results, written only for passing controllers
- training history, initial conditions and plots

To retest a saved controller again, run `python main_acrobot.py --retest_only runs_acrobot/<run>`.
Retest batch `k` is drawn with generator seed `--retest_seed_base + k` (default 50000), so every controller is
retested on the same initial conditions.

**Ablations** (Appendix C.4): add one flag, and a separate `--outdir`, to the commands above.

| Ablation | Flag |
|---|---|
| Training epochs | `--epochs 300` |
| Training rollout length | `--steps 900` or `--steps 1200` |
| Number of training starts | `--n_initial 16` or `--n_initial 32` |
| Burn-in for task-only objectives | omit `--burn_in 0 --eval_burn_in 0` |

**Validator.** For every upward crossing of `q1 = 0 (mod 2π)` after the test burn-in, embed
`(cos q2, sin q2, q̇1/4, q̇2/4)`. A trajectory settles if its last five crossings have RMS spread ≤ 0.05 and their
mean lies within 0.10 of the median over all trajectories. Trajectories with fewer than five crossings fail.

## Kuramoto–Sivashinsky

`u_t + u u_x + u_xx + u_xxxx = Σ_j a_j(t) b_j(x)` on a periodic domain with `L = 22`, 64 Fourier modes, ETDRK4 with
`Δt = 0.05`. There are four mean-free Gaussian actuators with `|a_j| ≤ 1`, and the policy is an MLP on eight point
sensors, updated every 0.25 time units.

**Objectives** (`--loss`): `energy`, `recurrence`, `topology`, `topology_energy`, `topology_recurrence`.

**Protocol (as in the paper).**
- **Training:** 8 fixed starts per seed. Each epoch is 600 control steps: the first 300 run without gradient, the
  loss is computed on the last 300, and backpropagation is truncated every 20 steps.
- **Validation / screening:** each seed has 32 starts, drawn with seed `5001 + training seed`. With `--use_best_val`,
  the best validation checkpoint (checked every 50 epochs) is kept.
- **Retest:** controllers that pass are run on 10 fixed sets of 32 starts, shared by every controller.
- **Hyperparameter search** (`--hpsearch`): for each objective, epochs × learning rate × hidden width
  (`{300,…,1500} × {3e-4, 1e-3, 3e-3} × {32, 64, 128}`), on the seeds in `--hp_seeds`, scored on validation only.
- **Success test:** a start settles if its last 150 time units are periodic (return-error test) and lie within
  Hausdorff distance 0.15 × orbit diameter of the medoid orbit. Persistent homology is not used.

**Main experiment (Table 2).**

```bash
python main_ks.py --mode setup              # reference values + initial-condition pools (once)

for loss in topology energy recurrence topology_energy topology_recurrence; do
  python main_ks.py --mode pipeline --loss $loss --hpsearch --hp_seeds 0,1,2 --use_best_val
done

python main_ks.py --mode summarize          # -> runs_ks/summary.md
```

On a Slurm cluster, `sbatch submit.sh` runs all five objectives in parallel in one job (10 CPUs). Edit the
partition and environment lines first. If the job reaches its time limit, submit it again: finished runs are
reused.

Results are written to `runs_ks/`:
- `<loss>_bestval_hp_seed<k>/`: frozen controller (`policy.pt`), `validation.json`, `test.json` and plots
- `hpsearch/<loss>_bestval/hpsearch.md`: ranking of the hyperparameter grid
- `summary.md`: per-objective pass / perfect counts and per-run results

Seeds 0–2 inform the hyperparameter choice, so the paper also reports seeds 3–9 alone.

**Other modes:**

```bash
python main_ks.py --mode train --loss topology --seed 0                      # train + validate one seed
python main_ks.py --mode test --run_dir runs_ks/topology_seed0               # retest it, if it passed
python main_ks.py --mode scan                                                # open-loop forcing baseline
python main_ks.py --mode pipeline --loss topology --smoke --outdir runs_smoke  # tiny settings, checks the code
```

**Ablations**:

| Ablation | Command |
|---|---|
| No training burn-in for the task-only baselines | add `--burn_in 0 --outdir runs_ks_noburnin` to the pipeline command |
| Recurrence with a misspecified period 0.6 P* (no search, 1500 epochs, final weights) | `python main_ks.py --mode pipeline --loss recurrence --epochs 1500 --P_scale 0.6 --outdir runs_ks_period06` |
| Same, correct period | `python main_ks.py --mode pipeline --loss recurrence --epochs 1500 --outdir runs_ks_period10` |

## Acknowledgments

This work was partially supported by the National Science Foundation under grant DMS-2229417 and by the Simons
Foundation under grant SFM-21269.

## Requirements

- Python ≥ 3.9
- [PyTorch](https://pytorch.org) ≥ 2.0 (the CPU build is sufficient; all computations use float64)
- [NumPy](https://numpy.org)
- [Matplotlib](https://matplotlib.org) (plots)
- [Oineus](https://github.com/anigmetov/oineus) 0.9.38 (differentiable Vietoris–Rips persistence; the version used in
  the paper)

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
torch>=2.0
numpy
matplotlib
oineus==0.9.38
```
