# PPO-Driven Inter-Slice Resource Sharing for 5G V2X Network Slicing

Code for the paper **"A PPO-based Adaptive Inter-Slice Resource Sharing Mechanism for 5G V2X Network Slicing"** (F. Bendaoud, H. Djiroun, M. Abdennebi).

A single PPO agent decides how much bandwidth each network slice should lend to overloaded high-priority slices in a multi-cell V2X scenario. Lagrange multipliers protect the lending slices, and a multi-pass admission controller in ns-3 always serves handover (HO) flows before new calls (NC).

> The full method, experiments and discussion are in the paper: [`docs/paper.pdf`](docs/paper.pdf)

---

## The problem

5G network slicing lets several logical networks share one radio infrastructure. V2X traffic (teleoperated driving, autonomous driving, infotainment, sensor upload) needs both **efficiency** and **isolation**:

- Static slicing keeps slices isolated but wastes capacity (about 91-92% utilization in our tests) and drops many handovers under load.
- Free sharing raises utilization but lets one slice's surge hurt the others.

We study three cells, each with four slices:

| Slice | Service | Priority | Baseline share |
|------:|---------|----------|---------------:|
| S0 | Autonomous driving | High | 23% |
| S1 | Teleoperated driving | High | 38% |
| S2 | Infotainment | Low | 13% |
| S3 | Sensor upload | Low | 26% |

When a high-priority slice is overloaded, bandwidth is borrowed from the other slices. Dropping a handover ends a live vehicular session, so HO flows must be protected first, without starving the slices that lend bandwidth.

## Main ideas

1. **Continuous actions.** For each cell the agent outputs four weights in [0, 1], one "donation willingness" per slice. This gives finer control than the fixed-step discrete actions of DQL-based approaches.
2. **Constrained learning with Lagrange multipliers.** Three constraints are enforced online: lender NC blocking ≤ 7%, lender HO blocking ≤ 6%, borrower HO blocking ≤ 6%. Multipliers are updated every tick and are also visible to the agent in its observation.
3. **Structural handover priority.** A multi-pass admission controller in ns-3 serves HO flows first (with preemption of NC flows if needed), independently of the reward.
4. **One policy for all loads.** A load feature and λ-curriculum training (45-65 veh/min) let one trained policy work across traffic levels without per-load retraining.

## Results

Evaluated against Strict Slicing, SoftSlice, NBS1 and a reproduced DQL baseline at five arrival rates (see the paper for full figures):

| Metric | PPO (this work) | Best baseline |
|--------|-----------------|---------------|
| Resource utilization | up to **99.7%** (λ = 65) | DQL 98.7% |
| Borrower HO drop at λ = 65 | **6.7%** | DQL 8.0%, NBS1 7.3% |
| Lender HO drop (all loads) | **< 2%** | DQL 4.5% |
| Average reward at λ = 65 | **8.2** | DQL 0.5 |

The cost is higher new-call blocking in lender cells (7.8-11.5%), a deliberate trade-off to protect handovers.

**Ablation at λ = 60:**

| Variant | Util. (%) | Borrower HO drop (%) | Lender NC blk (%) | Reward |
|---------|----------:|---------------------:|------------------:|-------:|
| Full PPO | 99.4 | 5.8 | 10.8 | 8.9 |
| No Lagrange multipliers | 99.1 | 7.4 | 14.1 | 7.1 |
| No multi-pass controller | 97.8 | 9.2 | 9.5 | 6.4 |
| Discrete actions | 98.6 | 6.9 | 11.0 | 7.8 |

Result figures and training logs are in [`results/`](results/).

---

## Repository structure

```
.
├── ns3_scheduler/
│   ├── gnb_scheduler.cc      # ns-3 scheduler: admission, lending, Lagrange, reward, OpenGym env
│   └── CMakeLists.txt
├── ns3/
│   └── ns3_env.py            # Python wrapper around the ns-3 process (ZMQ / ns3-gym)
├── bridge/
│   └── sumo_bridge.py        # vehicle mobility and traffic generator (writes tick JSON files)
├── ppo/
│   └── ppo_agent.py            # PPO training, evaluation and plotting
├── results/                  # figures, logs, final checkpoint
└── docs/
    └── paper.pdf
```

`ppo_agent.py` imports `ns3_env` from `<repo root>/ns3`, so it must stay one folder below the repository root.

## Requirements

- Ubuntu (tested on WSL2, Ubuntu 22.04)
- [ns-3.40](https://www.nsnam.org/) with the [`opengym`](https://apps.nsnam.org/app/ns3-gym/) module
- Python 3.8+ with `numpy`, `torch`, `matplotlib`, `pyzmq`, `gym` and `ns3gym`
- Optional: `libzmq` / `cppzmq` to build the ns-3 side with `WITH_ZMQ`

## Setup

**1. Build the scheduler inside ns-3**

```bash
cd ~/ns-allinone-3.40/ns-3.40/scratch
ln -s ~/v2x_sim/ns3_scheduler gnb_scheduler     # or copy the folder
cd ..
./ns3 configure
./ns3 build scratch/gnb_scheduler/gnb_scheduler
```

**2. Fix the hard-coded paths** in `ns3/ns3_env.py` (`_NS3_WAF_DEFAULT` and the binary path in `_build_ns3_cmd()`) so they point to your ns-3 installation.

**3. Generate traffic**

```bash
python bridge/sumo_bridge.py --report_dir bridge/tick_reports_live
```

This writes tick files for λ = 45, 50, 55, 60, 65 veh/min (300 episodes × 200 ticks each). These files are generated data and are not stored in the repository.

## Usage

```bash
# train + evaluate
python ppo/ppo_agent.py --report_dir bridge/tick_reports_live \
                      --out_dir outputs --total_events 800000

# evaluate a trained model only
python ppo/ppo_agent.py --skip_train --eval_episodes 40

# resume training
python ppo/ppo_agent.py --resume_checkpoint outputs/ppo_ckpt_upd200.pt --resume_events 400000
```

| Option | Default | Description |
|--------|---------|-------------|
| `--report_dir` | `bridge/tick_reports_live` | Folder with generated tick files |
| `--out_dir` | `outputs` | Checkpoints and figures |
| `--total_events` | 800000 | Training budget |
| `--n_steps` | 200 | Ticks per episode |
| `--port` | 5561 | Base ZMQ port |
| `--eval_episodes` | 40 | Episodes per λ at evaluation |
| `--skip_train`, `--skip_eval` | off | Skip a phase |
| `--log_file` | none | Mirror stdout to a file |

Evaluation writes the figures `fig1`-`fig7` and `fig15` (utilization, HO/NC blocking in lender and borrower cells, reward, training convergence) to the output folder.

## Training setup

MLP 256-256-128 for actor and critic, learning rates 1e-4 / 3e-4, γ = 0.99, GAE λ = 0.95, clip 0.15, 6 epochs, batch size 256, entropy coefficient 0.05 → 0.01, 800,000 training events (about 200 updates).

## Troubleshooting

- **`unknown target 'scratch_gnb_scheduler_gnb_scheduler'`**: the symlink in `scratch/` is broken (check path case, e.g. `V2X_SIM` vs `v2x_sim`) or CMake was not re-run. Fix the link and run `./ns3 configure` from the ns-3 root.
- **`./ns3: No such file or directory`**: run ns-3 commands from the `ns-3.40/` folder.
- **Port already in use**: kill the stale `ns3.40-gnb_scheduler-default` process or change `--port`.

## Citation

```bibtex
@article{bendaoud2025ppo,
  title   = {A PPO-based Adaptive Inter-Slice Resource Sharing Mechanism for 5G V2X Network Slicing},
  author  = {Bendaoud, Fayssal and Djiroun, Hiba and Abdennebi, Marwen},
  year    = {2025},
  note    = {Manuscript}
}
```

## License

Add your license here.
