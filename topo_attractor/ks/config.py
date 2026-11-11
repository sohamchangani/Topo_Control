"""KS experiment configuration: every setting has a default; main_ks.py exposes them as flags."""
from __future__ import annotations
from dataclasses import dataclass


LOSSES = ("topology", "energy", "recurrence", "topology_energy", "topology_recurrence")


@dataclass
class Cfg:
    seed: int = 0
    # --- KS plant
    L: float = 22.0
    N: int = 64
    dt: float = 0.05
    n_act: int = 4
    act_sigma: float = 0.4
    n_sens: int = 8
    ctrl_every: int = 5               # solver substeps per timestep (one timestep = 0.25 time units)
    c_max: float = 1.0
    # --- initial conditions
    ic_amp: float = 0.3               # random low-mode fields
    ic_amp_ood: float = 1.0           # optional out-of-distribution test set (--ood)
    pool_burn: float = 150.0
    pool_every: float = 5.0
    pool_size: int = 256              # states of the UNCONTROLLED attractor
    # --- loss
    loss: str = "topology"
    w_top: float = 1.0
    w_energy: float = 1.0
    w_rec: float = 1.0
    beta_effort: float = 0.05
    R_cap: float = 0.0                # 0 = off; otherwise min(R, R_cap) (guards against blow-up)
    cold_kappa: float = 0.0           # >0: when no (1,1) plateau exists use -kappa*l1 as cold-start push
    # --- training (Acrobot defaults; lengths in timesteps)
    n_initial: int = 8                # training starts, drawn once per seed and reused every epoch
    steps: int = 600                  # timesteps per training rollout
    burn_in: int = 300                # discarded; loss on the last steps - burn_in timesteps
    epochs: int = 150                 # one gradient update per epoch
    lr: float = 1e-3
    clip: float = 2.0                 # gradient-norm clip
    resample_initials: bool = False   # True: draw n_initial new starts every epoch
    use_best_val: bool = False        # True: freeze the best validation checkpoint instead of the final epoch
    val_every: int = 50               # epochs between validation checks (only with use_best_val)
    hidden: int = 64
    tbptt: int = 20                   # gradient truncation (timesteps) inside the scored window
    N_cloud: int = 96                 # farthest-point subsample used by the PH loss
    K_modes: int = 15                 # PH coordinates: Fourier modes 1..K (2K dims)
    # --- seeds
    n_seeds: int = 10                 # training seeds 0..n_seeds-1 (pipeline mode)
    val_seed_offset: int = 5001       # validation starts use generator val_seed_offset + training seed
    n_test_seeds: int = 10            # test seeds 1..n_test_seeds, shared by every controller
    # --- hyperparameter search (pipeline --hpsearch); scored on validation only
    hpsearch: bool = False
    hp_epochs: tuple = (300, 600, 900, 1200, 1500)
    hp_lrs: tuple = (3e-4, 1e-3, 3e-3)
    hp_hiddens: tuple = (32, 64, 128)
    hp_seeds: tuple = ()              # seeds used for the search; () = all training seeds
    # --- validation / testing (Acrobot defaults; lengths in timesteps)
    eval_initial: int = 32            # starts per validation / test set
    eval_steps: int = 1200
    eval_burn_in: int = 600           # success is judged on the last eval_steps - eval_burn_in
    # --- reference values handed to the baselines (filled by calibrate; 0 = not set)
    E_star: float = 0.0               # target mean energy 0.5*|z|^2 (energy baseline)
    v_ref: float = 0.0                # temporal variance of the reference loop
    P_ref: float = 0.0                # period of the reference loop (recurrence baseline)
    P_scale: float = 1.0              # recurrence loss uses P_scale * P_ref (misspecification ablation)
    # --- validator (success rate)
    lag_max: float = 30.0             # max return lag (time units)
    decor_thr: float = 0.7            # first lag with relative return error >= this is "decorrelated"
    tol_rec: float = 0.10             # periodic iff min relative return error <= tol_rec
    rms_floor_frac: float = 0.2       # start must oscillate with rms >= this * rms_ref (else "steady")
    rms_floor_abs: float = 0.05       # used when no reference is available
    delta_rel: float = 0.15           # common-orbit tolerance: Hausdorff <= delta_rel * orbit diameter
    n_lyap: int = 8
    T_lyap: float = 300.0
    lyap_discard: float = 20.0
    lam_tol: float = 0.02
    require_lyap: bool = False        # True: qualification also needs max Lyapunov <= lam_tol
    N_cert: int = 200
    cert_factor: float = 4.0          # certificate: R > cert_factor * eps_hat
    cert: bool = True
    ood: bool = False                 # True: also report an out-of-distribution set (not counted)
    # --- fixed (open loop) forcing
    amp: float = 1.0
    pattern: tuple = (1.0, -1.0, 1.0, -1.0)
    amps: tuple = (0.4, 0.65, 0.8, 1.0, 1.2)
    # --- io
    outdir: str = "runs_ks"
    threads: int = 0

    @property
    def dt_ctrl(self) -> float:
        return self.dt * self.ctrl_every

    def n_ctrl(self, T: float) -> int:
        return max(1, int(round(T / self.dt_ctrl)))


def check_cfg(c: Cfg):
    if not 0 <= c.burn_in < c.steps:
        raise ValueError("need 0 <= burn_in < steps")
    if not 0 <= c.eval_burn_in < c.eval_steps:
        raise ValueError("need 0 <= eval_burn_in < eval_steps")
    if c.n_initial < 1 or c.eval_initial < 2:
        raise ValueError("need n_initial >= 1 and eval_initial >= 2")
    if c.resample_initials and max(c.epochs, *c.hp_epochs) >= 10000:
        raise ValueError("--resample_initials supports at most 9999 epochs (per-epoch generator seeds)")
    if c.val_every < 1:
        raise ValueError("need val_every >= 1")
    if not c.hp_epochs or not c.hp_lrs or not c.hp_hiddens or min(c.hp_epochs) < 1:
        raise ValueError("hp_epochs / hp_lrs / hp_hiddens must be non-empty, epochs >= 1")
