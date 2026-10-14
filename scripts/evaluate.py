# # #!/usr/bin/env python3
# # """
# # eval_online.py — TRUE online evaluation with real SUMO traffic
# # ═══════════════════════════════════════════════════════════════

# # Architecture:
# #     Python (this script) binds REP on :5555  ← ns-3 connects as REQ
# #     SUMO bridge         binds PUSH on :5556  ← ns-3 pulls from here

# # HOW TO RUN — ALL λ flows (3 terminals, repeat for each λ):
# # ═══════════════════════════════════════════════════════════

# #   Terminal 1 — agent (run ONCE, loops over all λ automatically):
# #     cd ~/v2x_sim
# #     python scripts/eval_online.py \
# #         --all_lambdas \
# #         --checkpoint_dir outputs_ns3_eval \
# #         --n_episodes 10 \
# #         --n_steps 200

# #   For each λ, restart ns-3 + SUMO bridge in the other terminals:

# #   Terminal 2 — SUMO bridge (restart for each λ, e.g. λ=55):
# #     python bridge/sumo_bridge.py --online --lambda_arrival 55 \
# #         --n_steps 2000 --push_port 5556 --sumo_bin sumo \
# #         --sumo_cfg sumo/v2x_sim.sumocfg

# #   Terminal 3 — ns-3 (restart for each λ):
# #     cd ~/ns-allinone-3.40/ns-3.40
# #     ./build/scratch/gnb_scheduler/ns3.40-gnb_scheduler-default \
# #         --simTime=2000 --openGymPort=5555 --zmqPull=1 --arrivalPort=5556

# # HOW TO RUN — Single λ:
# #     python scripts/eval_online.py \
# #         --checkpoint outputs_ns3_eval/ppo_v48_lam55.pt \
# #         --lambda_arrival 55 --n_episodes 10
# # """

# # import os, sys, argparse, warnings, json
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import matplotlib
# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt
# # from datetime import datetime
# # warnings.filterwarnings("ignore")

# # _ROOT = os.path.dirname(os.path.abspath(__file__))
# # sys.path.insert(0, _ROOT)

# # DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # N_CELLS  = 3
# # N_SLICES = 4


# # # ══════════════════════════════════════════════════════════════════════════
# # #  Actor
# # # ══════════════════════════════════════════════════════════════════════════
# # def _mlp(in_d, hidden, out_d, gain):
# #     layers, d = [], in_d
# #     for h in hidden:
# #         layers += [nn.Linear(d, h), nn.Tanh()]
# #         d = h
# #     layers.append(nn.Linear(d, out_d))
# #     net = nn.Sequential(*layers)
# #     for m in net.modules():
# #         if isinstance(m, nn.Linear):
# #             nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
# #             nn.init.zeros_(m.bias)
# #     nn.init.orthogonal_(net[-1].weight, gain=gain)
# #     nn.init.zeros_(net[-1].bias)
# #     return net


# # class Actor(nn.Module):
# #     def __init__(self, obs_dim, act_dim, hidden=(256, 256, 128)):
# #         super().__init__()
# #         self.net     = _mlp(obs_dim, hidden, act_dim, 0.01)
# #         self.log_std = nn.Parameter(torch.full((act_dim,), 0.2))

# #     @torch.no_grad()
# #     def mean_action(self, obs):
# #         return ((torch.tanh(self.net(obs)) + 1.0) * 0.5).clamp(0.0, 1.0)


# # def load_actor(path, obs_dim, act_dim, hidden=(256, 256, 128)):
# #     actor = Actor(obs_dim, act_dim, hidden).to(DEVICE)
# #     ckpt  = torch.load(path, map_location=DEVICE)
# #     actor.load_state_dict(ckpt["actor"])
# #     actor.eval()
# #     print(f"[eval] ✓ Loaded actor: {path}")
# #     return actor


# # # ══════════════════════════════════════════════════════════════════════════
# # #  OnlineNS3Env
# # # ══════════════════════════════════════════════════════════════════════════
# # class OnlineNS3Env:
# #     obs_dim = 15
# #     act_dim = N_SLICES
# #     N_CELLS = N_CELLS

# #     def __init__(self, port=5555, max_steps=200, seed=7777):
# #         self.port      = port
# #         self.max_steps = max_steps
# #         self.seed      = seed
# #         self._step     = 0
# #         self._gym      = None
# #         self._last_obs = None
# #         self._connect()

# #     def _connect(self):
# #         try:
# #             from ns3gym.ns3env import Ns3Env
# #         except ImportError:
# #             print("[eval] ✗ ns3gym not found.")
# #             print("       pip install -e ~/v2x_sim/ns3-gym/model/ns3gym/")
# #             sys.exit(1)

# #         print(f"[eval] Binding REP socket on port {self.port} …")
# #         print(f"[eval] Waiting for ns-3 to connect — start ns-3 now in another terminal.")
# #         self._gym = Ns3Env(port=self.port, startSim=False, simSeed=self.seed)
# #         print(f"[eval] ✓ ns-3 connected and ready.")

# #     def _parse_obs(self, raw):
# #         if raw is None:
# #             return np.zeros((N_CELLS, self.obs_dim), np.float32)
# #         flat  = np.asarray(raw, dtype=np.float32).flatten()
# #         total = N_CELLS * self.obs_dim
# #         if flat.size < total:
# #             flat = np.pad(flat, (0, total - flat.size))
# #         return flat[:total].reshape(N_CELLS, self.obs_dim)

# #     def _parse_info(self, ns3_info):
# #         raw = {}
# #         if isinstance(ns3_info, str) and ns3_info.strip():
# #             try:
# #                 raw = json.loads(ns3_info)
# #             except Exception:
# #                 pass
# #         elif isinstance(ns3_info, dict):
# #             raw = ns3_info

# #         cell_infos = []
# #         for c in range(N_CELLS):
# #             cd     = raw.get(f"c{c}", {})
# #             ho_acc = int(cd.get("ho_acc", 0))
# #             ho_rej = int(cd.get("ho_rej", 0))
# #             nc_acc = int(cd.get("nc_acc", 0))
# #             nc_rej = int(cd.get("nc_rej", 0))
# #             # ── NEW: per-cell util and prio_sat ───────────────────────────
# #             util     = float(cd.get("util_served",  0.0))
# #             prio_sat = float(cd.get("prio_sat",     1.0))
# #             ho_t = ho_acc + ho_rej
# #             nc_t = nc_acc + nc_rej
# #             rt   = ho_t + nc_t
# #             cell_infos.append({
# #                 "ho_accepted": ho_acc, "ho_rejected": ho_rej,
# #                 "nc_accepted": nc_acc, "nc_rejected": nc_rej,
# #                 "util":        util,
# #                 "prio_sat":    prio_sat,
# #                 "ho_rate":     ho_acc / ho_t if ho_t > 0 else 1.0,
# #                 "nc_rate":     nc_acc / nc_t if nc_t > 0 else 1.0,
# #                 "accept_rate": (ho_acc + nc_acc) / rt if rt > 0 else 1.0,
# #             })

# #         ho_acc_t = sum(ci["ho_accepted"] for ci in cell_infos)
# #         ho_rej_t = sum(ci["ho_rejected"] for ci in cell_infos)
# #         nc_acc_t = sum(ci["nc_accepted"] for ci in cell_infos)
# #         nc_rej_t = sum(ci["nc_rejected"] for ci in cell_infos)
# #         total_t  = ho_acc_t + ho_rej_t + nc_acc_t + nc_rej_t

# #         return {
# #             "cell_infos":  cell_infos,
# #             "ho_acc": ho_acc_t, "ho_rej": ho_rej_t,
# #             "nc_acc": nc_acc_t, "nc_rej": nc_rej_t,
# #             "ho_rate":     ho_acc_t / max(1, ho_acc_t + ho_rej_t),
# #             "nc_rate":     nc_acc_t / max(1, nc_acc_t + nc_rej_t),
# #             "accept_rate": (ho_acc_t + nc_acc_t) / max(1, total_t),
# #             # aggregated across cells
# #             "util":     float(np.mean([ci["util"]     for ci in cell_infos])),
# #             "prio_sat": float(np.mean([ci["prio_sat"] for ci in cell_infos])),
# #             # ── NEW: overloaded vs lender split ──────────────────────────
# #             # Convention: cell 0 is the overloaded cell (highest load),
# #             # cells 1 & 2 are the lender cells.
# #             # If your ns-3 marks cells differently, adjust here.
# #             "ho_acc_ovl": cell_infos[0]["ho_accepted"],
# #             "ho_rej_ovl": cell_infos[0]["ho_rejected"],
# #             "nc_acc_ovl": cell_infos[0]["nc_accepted"],
# #             "nc_rej_ovl": cell_infos[0]["nc_rejected"],
# #             "ho_acc_lnd": sum(cell_infos[c]["ho_accepted"] for c in range(1, N_CELLS)),
# #             "ho_rej_lnd": sum(cell_infos[c]["ho_rejected"] for c in range(1, N_CELLS)),
# #             "nc_acc_lnd": sum(cell_infos[c]["nc_accepted"] for c in range(1, N_CELLS)),
# #             "nc_rej_lnd": sum(cell_infos[c]["nc_rejected"] for c in range(1, N_CELLS)),
# #         }

# #     def reset(self):
# #         self._step = 0
# #         if self._last_obs is None:
# #             raw = self._gym.ns3ZmqBridge.obsData
# #             self._last_obs = self._parse_obs(raw)
# #         return self._last_obs

# #     def step(self, actions):
# #         self._step += 1
# #         flat = np.clip(actions, 0.0, 1.0).flatten().astype(np.float32)
# #         try:
# #             raw_obs, reward, done, ns3_info = self._gym.step(flat)
# #         except Exception as e:
# #             print(f"[eval] ⚠ ZMQ exception on step {self._step} "
# #                   f"(ns-3 closed after done) — ending episode. ({e})")
# #             obs = self._last_obs if self._last_obs is not None else \
# #                   np.zeros((N_CELLS, self.obs_dim), np.float32)
# #             return obs, np.zeros(N_CELLS, np.float32), True, self._parse_info(None)

# #         obs  = self._parse_obs(raw_obs)
# #         self._last_obs = obs
# #         info = self._parse_info(ns3_info)

# #         if self._step == 1:
# #             print(f"[eval] [debug] ns3_info type={type(ns3_info).__name__} "
# #                   f"repr={repr(ns3_info)[:120]}")

# #         done = bool(done) or (self._step >= self.max_steps)
# #         return obs, np.full(N_CELLS, float(reward), np.float32), done, info

# #     def get_event_mask(self):
# #         return np.ones(N_CELLS, dtype=bool)

# #     def close(self):
# #         try:
# #             self._gym.close()
# #         except Exception:
# #             pass


# # # ══════════════════════════════════════════════════════════════════════════
# # #  Eval loop
# # # ══════════════════════════════════════════════════════════════════════════
# # def run_episode(actor, env):
# #     obs_all  = env.reset()
# #     done     = False
# #     ep_r     = 0.0
# #     ep_steps = 0
# #     last_info = {}

# #     while not done:
# #         actions = np.zeros((N_CELLS, env.act_dim), np.float32)
# #         for c in range(N_CELLS):
# #             ot = torch.tensor(obs_all[c], dtype=torch.float32,
# #                               device=DEVICE).unsqueeze(0)
# #             actions[c] = actor.mean_action(ot).squeeze(0).cpu().numpy()

# #         obs_all, rewards, done, info = env.step(actions)
# #         ep_r     += float(np.mean(rewards))
# #         ep_steps += 1
# #         last_info = info

# #     ho_acc = int(last_info.get("ho_acc", 0))
# #     ho_rej = int(last_info.get("ho_rej", 0))
# #     nc_acc = int(last_info.get("nc_acc", 0))
# #     nc_rej = int(last_info.get("nc_rej", 0))
# #     ho_t   = ho_acc + ho_rej
# #     nc_t   = nc_acc + nc_rej
# #     all_t  = ho_t + nc_t

# #     return {
# #         # ── global ────────────────────────────────────────────────────────
# #         "ep":          ep_steps,
# #         "avg_reward":  ep_r / max(1, ep_steps),
# #         "n_steps":     ep_steps,
# #         "ho_rate":     ho_acc / ho_t  if ho_t  > 0 else 0.0,
# #         "nc_rate":     nc_acc / nc_t  if nc_t  > 0 else 0.0,
# #         "accept_rate": (ho_acc + nc_acc) / all_t if all_t > 0 else 0.0,
# #         "ho_acc": ho_acc, "ho_rej": ho_rej,
# #         "nc_acc": nc_acc, "nc_rej": nc_rej,
# #         # ── util / prio ───────────────────────────────────────────────────
# #         "util":     last_info.get("util",     0.0),
# #         "prio_sat": last_info.get("prio_sat", 1.0),
# #         # ── overloaded cell ───────────────────────────────────────────────
# #         "ho_acc_ovl": last_info.get("ho_acc_ovl", 0),
# #         "ho_rej_ovl": last_info.get("ho_rej_ovl", 0),
# #         "nc_acc_ovl": last_info.get("nc_acc_ovl", 0),
# #         "nc_rej_ovl": last_info.get("nc_rej_ovl", 0),
# #         # ── lender cells ──────────────────────────────────────────────────
# #         "ho_acc_lnd": last_info.get("ho_acc_lnd", 0),
# #         "ho_rej_lnd": last_info.get("ho_rej_lnd", 0),
# #         "nc_acc_lnd": last_info.get("nc_acc_lnd", 0),
# #         "nc_rej_lnd": last_info.get("nc_rej_lnd", 0),
# #     }


# # def run_eval(actor, env, n_episodes, lam, plot_dir, tag):
# #     print(f"\n[eval] ▶ λ={lam}  {n_episodes} episodes …")
# #     print(f"[eval]   ⚠ ns-3 must be launched with --simTime={200 * n_episodes}")
# #     all_ep = []
# #     for i in range(n_episodes):
# #         r = run_episode(actor, env)
# #         all_ep.append(r)
# #         print(f"  ep {i+1:>2}/{n_episodes} | "
# #               f"R={r['avg_reward']:+.3f} | "
# #               f"HO={r['ho_rate']*100:.1f}% ({r['ho_acc']}✓/{r['ho_rej']}✗) | "
# #               f"NC={r['nc_rate']*100:.1f}% ({r['nc_acc']}✓/{r['nc_rej']}✗) | "
# #               f"steps={r['n_steps']}")

# #     # ── save episode JSON so plot_eval_all.py can load it ─────────────────
# #     json_path = os.path.join(plot_dir, f"episodes_lam{lam}_{tag}.json")
# #     with open(json_path, "w") as f:
# #         json.dump(all_ep, f, indent=2)
# #     print(f"[eval] ✓ Episodes saved: {json_path}")

# #     scalar_keys = ["ho_rate", "nc_rate", "accept_rate", "avg_reward",
# #                    "n_steps", "util", "prio_sat"]
# #     summary = {k: float(np.mean([e[k] for e in all_ep])) for k in scalar_keys}
# #     summary["lambda"] = lam

# #     # derived drop/block rates (mean across episodes)
# #     def _rate(acc_key, rej_key):
# #         vals = []
# #         for e in all_ep:
# #             a, r = e[acc_key], e[rej_key]
# #             vals.append(r / max(1, a + r))
# #         return float(np.mean(vals))

# #     summary["ho_drop_ovl"]  = _rate("ho_acc_ovl", "ho_rej_ovl")
# #     summary["ho_drop_lnd"]  = _rate("ho_acc_lnd", "ho_rej_lnd")
# #     summary["nc_block_ovl"] = _rate("nc_acc_ovl", "nc_rej_ovl")
# #     summary["nc_block_lnd"] = _rate("nc_acc_lnd", "nc_rej_lnd")

# #     return summary, all_ep


# # # ══════════════════════════════════════════════════════════════════════════
# # #  Paper-style plots  (all λ on one axes, like the article)
# # # ══════════════════════════════════════════════════════════════════════════
# # plt.rcParams.update({
# #     "figure.facecolor": "white", "axes.facecolor": "white",
# #     "axes.edgecolor":   "#333",  "axes.labelcolor": "#111",
# #     "xtick.color":      "#333",  "ytick.color":     "#333",
# #     "grid.color":       "#ccc",  "grid.linewidth":  0.6,
# #     "grid.linestyle":   "--",    "font.size":       10,
# #     "legend.framealpha":0.9,     "legend.edgecolor":"#aaa",
# # })

# # # marker/colour for each policy
# # POL_STYLE = {
# #     "Strict Slicing": ("s-", "black"),
# #     "Softslice":      ("o-", "#e05050"),
# #     "NBS1":           ("^-", "#4488cc"),
# #     "PPO (ours)":     ("x-", "#1155aa"),
# # }

# # def _save_fig(fig, path):
# #     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
# #     fig.savefig(path, dpi=150, bbox_inches="tight")
# #     plt.close(fig)
# #     print(f"[eval] ✓ Saved → {path}")


# # def _make_line_fig(title, ylabel, lambdas, ppo_vals,
# #                    baseline_vals=None, ylim=None, zero_line=False):
# #     """
# #     lambdas      : list of λ values (x-axis)
# #     ppo_vals     : list of y values for PPO
# #     baseline_vals: dict  { policy_name: [y0, y1, …] }  or None
# #     """
# #     fig, ax = plt.subplots(figsize=(7, 4.5))
# #     ax.set_title(title)

# #     if baseline_vals:
# #         for pol, yvals in baseline_vals.items():
# #             mk, col = POL_STYLE[pol]
# #             ax.plot(lambdas, yvals, mk, color=col, lw=1.8, ms=6, label=pol)

# #     mk, col = POL_STYLE["PPO (ours)"]
# #     ax.plot(lambdas, ppo_vals, mk, color=col, lw=2.2, ms=8, label="PPO (ours)")

# #     if zero_line:
# #         ax.axhline(0, color="black", lw=0.8, ls=":", zorder=0)
# #     if ylim:
# #         ax.set_ylim(*ylim)

# #     ax.set_xticks(lambdas)
# #     ax.set_xlabel("Arrivals (Vehicles/min)")
# #     ax.set_ylabel(ylabel)
# #     ax.grid(True)
# #     ax.legend()
# #     return fig


# # def plot_paper_figures(all_summaries, plot_dir, tag):
# #     """
# #     all_summaries : list of summary dicts, one per λ (sorted by lambda)
# #     """
# #     all_summaries = sorted(all_summaries, key=lambda s: s["lambda"])
# #     lambdas = [s["lambda"] for s in all_summaries]

# #     def _get(key):
# #         return [s[key] for s in all_summaries]

# #     # ── Fig 1 — Resource Utilisation ──────────────────────────────────────
# #     fig = _make_line_fig(
# #         "Resource Utilisation vs. Arrival Rate",
# #         "Resource Utilisation (%)",
# #         lambdas, [v * 100 for v in _get("util")],
# #         ylim=(50, 100),
# #     )
# #     _save_fig(fig, os.path.join(plot_dir, f"fig1_resource_utilisation_{tag}.png"))

# #     # ── Fig 2 — HO Drop — Overloaded ─────────────────────────────────────
# #     fig = _make_line_fig(
# #         "Handover Drop Probability — Overloaded Slices",
# #         "HO Drop Probability",
# #         lambdas, _get("ho_drop_ovl"),
# #     )
# #     _save_fig(fig, os.path.join(plot_dir, f"fig2_ho_drop_overloaded_{tag}.png"))

# #     # ── Fig 3 — HO Drop — Lender ──────────────────────────────────────────
# #     fig = _make_line_fig(
# #         "Handover Drop Probability — Lender Slices",
# #         "HO Drop Probability",
# #         lambdas, _get("ho_drop_lnd"),
# #     )
# #     _save_fig(fig, os.path.join(plot_dir, f"fig3_ho_drop_lender_{tag}.png"))

# #     # ── Fig 4 — NC Block — Overloaded ────────────────────────────────────
# #     fig = _make_line_fig(
# #         "New-Call Block Probability — Overloaded Slices",
# #         "NC Block Probability",
# #         lambdas, _get("nc_block_ovl"),
# #     )
# #     _save_fig(fig, os.path.join(plot_dir, f"fig4_nc_block_overloaded_{tag}.png"))

# #     # ── Fig 5 — NC Block — Lender ────────────────────────────────────────
# #     fig = _make_line_fig(
# #         "New-Call Block Probability — Lender Slices",
# #         "NC Block Probability",
# #         lambdas, _get("nc_block_lnd"),
# #     )
# #     _save_fig(fig, os.path.join(plot_dir, f"fig5_nc_block_lender_{tag}.png"))

# #     # ── Fig 6 — Average Reward ────────────────────────────────────────────
# #     fig = _make_line_fig(
# #         "Average Reward vs. Arrival Rate",
# #         "Average Reward",
# #         lambdas, _get("avg_reward"),
# #         zero_line=True,
# #     )
# #     _save_fig(fig, os.path.join(plot_dir, f"fig6_average_reward_{tag}.png"))

# #     # ── Fig 7 — HO Accept Rate ───────────────────────────────────────────
# #     fig = _make_line_fig(
# #         "Handover Accept Rate vs. Arrival Rate",
# #         "HO Accept Rate (%)",
# #         lambdas, [v * 100 for v in _get("ho_rate")],
# #     )
# #     _save_fig(fig, os.path.join(plot_dir, f"fig7_ho_accept_rate_{tag}.png"))

# #     # ── Fig 8 — NC Accept Rate ───────────────────────────────────────────
# #     fig = _make_line_fig(
# #         "New-Call Accept Rate vs. Arrival Rate",
# #         "NC Accept Rate (%)",
# #         lambdas, [v * 100 for v in _get("nc_rate")],
# #     )
# #     _save_fig(fig, os.path.join(plot_dir, f"fig8_nc_accept_rate_{tag}.png"))

# #     # ── Fig 9 — Overall Accept Rate ──────────────────────────────────────
# #     fig = _make_line_fig(
# #         "Overall Accept Rate vs. Arrival Rate",
# #         "Accept Rate (%)",
# #         lambdas, [v * 100 for v in _get("accept_rate")],
# #     )
# #     _save_fig(fig, os.path.join(plot_dir, f"fig9_overall_accept_rate_{tag}.png"))

# #     print(f"\n[eval] ✅ Paper figures saved to {plot_dir}/")


# # # ══════════════════════════════════════════════════════════════════════════
# # #  Legacy single-lambda plot (kept for backward compatibility)
# # # ══════════════════════════════════════════════════════════════════════════
# # COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# # def plot_single(all_ep, lam, save_path):
# #     keys   = ["avg_reward", "ho_rate", "nc_rate", "accept_rate"]
# #     titles = ["Avg Step Reward", "HO Accept Rate",
# #               "NC Accept Rate",  "Overall Accept Rate"]
# #     n_ep = len(all_ep)
# #     x    = np.arange(1, n_ep + 1)

# #     fig, axes = plt.subplots(1, 4, figsize=(20, 4))
# #     fig.patch.set_facecolor("#0f1117")
# #     fig.suptitle(f"Online Eval (SUMO) — λ={lam} | {n_ep} episodes",
# #                  color="white", fontsize=13, fontweight="bold")

# #     for idx, (key, title) in enumerate(zip(keys, titles)):
# #         ax    = axes[idx]
# #         ax.set_facecolor("#1a1d27")
# #         for sp in ax.spines.values():
# #             sp.set_edgecolor("#2e3142")
# #         y     = np.array([e[key] for e in all_ep])
# #         color = COLORS[idx]
# #         ax.plot(x, y, "o-", color=color, linewidth=2, markersize=7)
# #         ax.axhline(y.mean(), color=color, linestyle="--",
# #                    linewidth=0.8, alpha=0.6)
# #         ax.set_title(title, color="white", fontsize=10)
# #         ax.set_xlabel("Episode", color="#8888aa", fontsize=8)
# #         ax.tick_params(colors="#8888aa", labelsize=8)
# #         ax.grid(alpha=0.15, color="#3a3f55")
# #         ax.text(0.97, 0.95, f"μ={y.mean():.3f}",
# #                 transform=ax.transAxes, color=color,
# #                 fontsize=8, ha="right", va="top")

# #     plt.tight_layout()
# #     os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
# #     fig.savefig(save_path, dpi=150, bbox_inches="tight",
# #                 facecolor=fig.get_facecolor())
# #     plt.close(fig)
# #     print(f"[eval] ✓ Per-λ plot: {save_path}")


# # # ══════════════════════════════════════════════════════════════════════════
# # #  Main
# # # ══════════════════════════════════════════════════════════════════════════
# # def parse_args():
# #     p = argparse.ArgumentParser()
# #     p.add_argument("--checkpoint",     type=str, default=None,
# #                    help="Single checkpoint path (single-λ mode)")
# #     p.add_argument("--checkpoint_dir", type=str, default="outputs_ns3_eval",
# #                    help="Dir with ppo_v48_lam{λ}.pt files (all-λ mode)")
# #     p.add_argument("--all_lambdas",    action="store_true",
# #                    help="Evaluate all λ values sequentially")
# #     p.add_argument("--lambdas",        type=int, nargs="+",
# #                    default=[45, 50, 55, 60, 65],
# #                    help="List of λ values when --all_lambdas is set")
# #     p.add_argument("--lambda_arrival", type=int, default=55,
# #                    help="Single λ value (single-λ mode)")
# #     p.add_argument("--n_episodes",     type=int, default=10)
# #     p.add_argument("--n_steps",        type=int, default=200)
# #     p.add_argument("--port",           type=int, default=5555)
# #     p.add_argument("--plot_dir",       type=str, default="eval_plots")
# #     p.add_argument("--seed",           type=int, default=7777)
# #     return p.parse_args()


# # def main():
# #     args = parse_args()
# #     os.makedirs(args.plot_dir, exist_ok=True)
# #     tag      = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     run_list = args.lambdas if args.all_lambdas else [args.lambda_arrival]

# #     all_summaries = []

# #     # ── For all-λ mode: connect ONCE, then loop ───────────────────────────
# #     # The env keeps the ZMQ socket open between λ runs.
# #     # For each λ you must restart ns-3 + SUMO bridge in the other terminals.
# #     env = OnlineNS3Env(port=args.port, max_steps=args.n_steps, seed=args.seed)

# #     for lam in run_list:
# #         # Resolve checkpoint path
# #         if args.checkpoint and not args.all_lambdas:
# #             ckpt_path = args.checkpoint
# #         else:
# #             ckpt_path = os.path.join(args.checkpoint_dir,
# #                                      f"ppo_v48_lam{lam}.pt")

# #         if not os.path.exists(ckpt_path):
# #             print(f"[eval] ⚠ Not found: {ckpt_path} — skipping λ={lam}")
# #             continue

# #         actor = load_actor(ckpt_path, obs_dim=env.obs_dim, act_dim=env.act_dim)

# #         # ── In all-λ mode: pause and ask user to restart ns-3 + SUMO ─────
# #         if args.all_lambdas and len(run_list) > 1:
# #             print(f"\n{'='*60}")
# #             print(f"  λ = {lam}  ready.  Please:")
# #             print(f"  1) Restart SUMO bridge with --lambda_arrival {lam}")
# #             print(f"  2) Restart ns-3 with --simTime={args.n_steps * args.n_episodes}")
# #             print(f"  Then press ENTER here to continue …")
# #             print(f"{'='*60}")
# #             input()

# #         summary, all_ep = run_eval(actor, env, args.n_episodes, lam,
# #                                    args.plot_dir, tag)
# #         all_summaries.append(summary)

# #         # Per-λ dark plot (as before)
# #         plot_single(all_ep, lam,
# #                     save_path=os.path.join(args.plot_dir,
# #                                            f"eval_lam{lam}_{tag}.png"))

# #         print(f"\n[eval] λ={lam} summary:")
# #         print(f"  Reward      : {summary['avg_reward']:+.4f}")
# #         print(f"  HO Accept   : {summary['ho_rate']*100:.1f}%")
# #         print(f"  NC Accept   : {summary['nc_rate']*100:.1f}%")
# #         print(f"  Accept Rate : {summary['accept_rate']*100:.1f}%")
# #         print(f"  Util        : {summary['util']*100:.1f}%")
# #         print(f"  HO Drop OVL : {summary['ho_drop_ovl']*100:.2f}%")
# #         print(f"  HO Drop LND : {summary['ho_drop_lnd']*100:.2f}%")
# #         print(f"  NC Blk  OVL : {summary['nc_block_ovl']*100:.2f}%")
# #         print(f"  NC Blk  LND : {summary['nc_block_lnd']*100:.2f}%")

# #     env.close()

# #     # ── Paper-style figures (only when we have ≥ 2 λ results) ────────────
# #     if len(all_summaries) >= 2:
# #         plot_paper_figures(all_summaries, args.plot_dir, tag)
# #     elif len(all_summaries) == 1:
# #         print("[eval] Only 1 λ evaluated — skipping multi-λ paper figures.")
# #         print("       Run with --all_lambdas to get all paper-style plots.")

# #     print(f"\n[eval] ✅ Done. All plots saved in {args.plot_dir}/")


# # if __name__ == "__main__":
# #     main()

# #!/usr/bin/env python3
# """
# eval_online.py — TRUE online evaluation with real SUMO traffic
# ═══════════════════════════════════════════════════════════════

# Architecture:
#     Python (this script) binds REP on :5555  ← ns-3 connects as REQ
#     SUMO bridge         binds PUSH on :5556  ← ns-3 pulls from here

# HOW TO RUN — ALL λ flows (1 terminal, fully automatic):
# ═══════════════════════════════════════════════════════

#   Terminal 1 — agent (run ONCE, loops over all λ automatically):
#     cd ~/v2x_sim
#     python scripts/eval_online.py \
#         --all_lambdas \
#         --checkpoint_dir outputs_ns3_eval \
#         --n_episodes 10 \
#         --n_steps 200

#   ns-3 and SUMO bridge are launched/killed automatically for each λ.

# HOW TO RUN — Single λ (manual 3-terminal mode):
#     Terminal 1:
#         cd ~/v2x_sim
#         python scripts/eval_online.py \
#             --checkpoint outputs_ns3_eval/ppo_v48_lam55.pt \
#             --lambda_arrival 55 --n_episodes 10

#     Terminal 2:
#         python bridge/sumo_bridge.py --online --lambda_arrival 55 \
#             --n_steps 2000 --push_port 5556 --sumo_bin sumo \
#             --sumo_cfg sumo/v2x_sim.sumocfg

#     Terminal 3:
#         cd ~/ns-allinone-3.40/ns-3.40
#         ./build/scratch/gnb_scheduler/ns3.40-gnb_scheduler-default \
#             --simTime=2000 --openGymPort=5555 --zmqPull=1 --arrivalPort=5556
# """

# import os, sys, argparse, warnings, json, subprocess, time
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from datetime import datetime
# warnings.filterwarnings("ignore")

# _ROOT = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, _ROOT)

# DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# N_CELLS  = 3
# N_SLICES = 4

# NS3_DIR  = os.path.expanduser("~/ns-allinone-3.40/ns-3.40")
# V2X_DIR  = os.path.expanduser("~/v2x_sim")


# # ══════════════════════════════════════════════════════════════════════════
# #  Backend process management (ns-3 + SUMO bridge)
# # ══════════════════════════════════════════════════════════════════════════

# def launch_backends(lam, n_steps, n_episodes):
#     """Launch ns-3 and SUMO bridge as background subprocesses."""
#     sim_time = n_steps * n_episodes

#     print(f"\n[eval] 🚀 Launching ns-3  (simTime={sim_time}) …")
#     ns3 = subprocess.Popen(
#         [
#             "./build/scratch/gnb_scheduler/ns3.40-gnb_scheduler-default",
#             f"--simTime={sim_time}",
#             "--openGymPort=5555",
#             "--zmqPull=1",
#             "--arrivalPort=5556",
#         ],
#         cwd=NS3_DIR,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#     )

#     print(f"[eval] 🚀 Launching SUMO bridge (λ={lam}) …")
#     sumo = subprocess.Popen(
#         [
#             sys.executable, "bridge/sumo_bridge.py",
#             "--online",
#             f"--lambda_arrival={lam}",
#             f"--n_steps={sim_time}",
#             "--push_port=5556",
#             "--sumo_bin=sumo",
#             "--sumo_cfg=sumo/v2x_sim.sumocfg",
#         ],
#         cwd=V2X_DIR,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#     )

#     print("[eval] ⏳ Waiting 3 s for backends to bind ports …")
#     time.sleep(3)
#     return ns3, sumo


# def kill_backends(ns3, sumo):
#     """Terminate ns-3 and SUMO bridge cleanly."""
#     print("[eval] 🛑 Stopping backends …")
#     for p in [ns3, sumo]:
#         if p is None:
#             continue
#         try:
#             p.terminate()
#             p.wait(timeout=5)
#         except Exception:
#             try:
#                 p.kill()
#             except Exception:
#                 pass
#     time.sleep(2)
#     print("[eval] ✓ Backends stopped.")


# # ══════════════════════════════════════════════════════════════════════════
# #  Actor
# # ══════════════════════════════════════════════════════════════════════════
# def _mlp(in_d, hidden, out_d, gain):
#     layers, d = [], in_d
#     for h in hidden:
#         layers += [nn.Linear(d, h), nn.Tanh()]
#         d = h
#     layers.append(nn.Linear(d, out_d))
#     net = nn.Sequential(*layers)
#     for m in net.modules():
#         if isinstance(m, nn.Linear):
#             nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
#             nn.init.zeros_(m.bias)
#     nn.init.orthogonal_(net[-1].weight, gain=gain)
#     nn.init.zeros_(net[-1].bias)
#     return net


# class Actor(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden=(256, 256, 128)):
#         super().__init__()
#         self.net     = _mlp(obs_dim, hidden, act_dim, 0.01)
#         self.log_std = nn.Parameter(torch.full((act_dim,), 0.2))

#     @torch.no_grad()
#     def mean_action(self, obs):
#         return ((torch.tanh(self.net(obs)) + 1.0) * 0.5).clamp(0.0, 1.0)


# def load_actor(path, obs_dim, act_dim, hidden=(256, 256, 128)):
#     actor = Actor(obs_dim, act_dim, hidden).to(DEVICE)
#     ckpt  = torch.load(path, map_location=DEVICE)
#     actor.load_state_dict(ckpt["actor"])
#     actor.eval()
#     print(f"[eval] ✓ Loaded actor: {path}")
#     return actor


# # ══════════════════════════════════════════════════════════════════════════
# #  OnlineNS3Env
# # ══════════════════════════════════════════════════════════════════════════
# class OnlineNS3Env:
#     obs_dim = 15
#     act_dim = N_SLICES
#     N_CELLS = N_CELLS

#     def __init__(self, port=5555, max_steps=200, seed=7777):
#         self.port      = port
#         self.max_steps = max_steps
#         self.seed      = seed
#         self._step     = 0
#         self._gym      = None
#         self._last_obs = None
#         self._connect()

#     def _connect(self):
#         try:
#             from ns3gym.ns3env import Ns3Env
#         except ImportError:
#             print("[eval] ✗ ns3gym not found.")
#             print("       pip install -e ~/v2x_sim/ns3-gym/model/ns3gym/")
#             sys.exit(1)

#         print(f"[eval] Binding REP socket on port {self.port} …")
#         print(f"[eval] Waiting for ns-3 to connect …")
#         self._gym = Ns3Env(port=self.port, startSim=False, simSeed=self.seed)
#         print(f"[eval] ✓ ns-3 connected and ready.")

#     def _parse_obs(self, raw):
#         if raw is None:
#             return np.zeros((N_CELLS, self.obs_dim), np.float32)
#         flat  = np.asarray(raw, dtype=np.float32).flatten()
#         total = N_CELLS * self.obs_dim
#         if flat.size < total:
#             flat = np.pad(flat, (0, total - flat.size))
#         return flat[:total].reshape(N_CELLS, self.obs_dim)

#     def _parse_info(self, ns3_info):
#         raw = {}
#         if isinstance(ns3_info, str) and ns3_info.strip():
#             try:
#                 raw = json.loads(ns3_info)
#             except Exception:
#                 pass
#         elif isinstance(ns3_info, dict):
#             raw = ns3_info

#         cell_infos = []
#         for c in range(N_CELLS):
#             cd     = raw.get(f"c{c}", {})
#             ho_acc = int(cd.get("ho_acc", 0))
#             ho_rej = int(cd.get("ho_rej", 0))
#             nc_acc = int(cd.get("nc_acc", 0))
#             nc_rej = int(cd.get("nc_rej", 0))
#             util     = float(cd.get("util_served",  0.0))
#             prio_sat = float(cd.get("prio_sat",     1.0))
#             ho_t = ho_acc + ho_rej
#             nc_t = nc_acc + nc_rej
#             rt   = ho_t + nc_t
#             cell_infos.append({
#                 "ho_accepted": ho_acc, "ho_rejected": ho_rej,
#                 "nc_accepted": nc_acc, "nc_rejected": nc_rej,
#                 "util":        util,
#                 "prio_sat":    prio_sat,
#                 "ho_rate":     ho_acc / ho_t if ho_t > 0 else 1.0,
#                 "nc_rate":     nc_acc / nc_t if nc_t > 0 else 1.0,
#                 "accept_rate": (ho_acc + nc_acc) / rt if rt > 0 else 1.0,
#             })

#         ho_acc_t = sum(ci["ho_accepted"] for ci in cell_infos)
#         ho_rej_t = sum(ci["ho_rejected"] for ci in cell_infos)
#         nc_acc_t = sum(ci["nc_accepted"] for ci in cell_infos)
#         nc_rej_t = sum(ci["nc_rejected"] for ci in cell_infos)
#         total_t  = ho_acc_t + ho_rej_t + nc_acc_t + nc_rej_t

#         return {
#             "cell_infos":  cell_infos,
#             "ho_acc": ho_acc_t, "ho_rej": ho_rej_t,
#             "nc_acc": nc_acc_t, "nc_rej": nc_rej_t,
#             "ho_rate":     ho_acc_t / max(1, ho_acc_t + ho_rej_t),
#             "nc_rate":     nc_acc_t / max(1, nc_acc_t + nc_rej_t),
#             "accept_rate": (ho_acc_t + nc_acc_t) / max(1, total_t),
#             "util":     float(np.mean([ci["util"]     for ci in cell_infos])),
#             "prio_sat": float(np.mean([ci["prio_sat"] for ci in cell_infos])),
#             "ho_acc_ovl": cell_infos[0]["ho_accepted"],
#             "ho_rej_ovl": cell_infos[0]["ho_rejected"],
#             "nc_acc_ovl": cell_infos[0]["nc_accepted"],
#             "nc_rej_ovl": cell_infos[0]["nc_rejected"],
#             "ho_acc_lnd": sum(cell_infos[c]["ho_accepted"] for c in range(1, N_CELLS)),
#             "ho_rej_lnd": sum(cell_infos[c]["ho_rejected"] for c in range(1, N_CELLS)),
#             "nc_acc_lnd": sum(cell_infos[c]["nc_accepted"] for c in range(1, N_CELLS)),
#             "nc_rej_lnd": sum(cell_infos[c]["nc_rejected"] for c in range(1, N_CELLS)),
#         }

#     def reset(self):
#         self._step = 0
#         if self._last_obs is None:
#             raw = self._gym.ns3ZmqBridge.obsData
#             self._last_obs = self._parse_obs(raw)
#         return self._last_obs

#     def step(self, actions):
#         self._step += 1
#         flat = np.clip(actions, 0.0, 1.0).flatten().astype(np.float32)
#         try:
#             raw_obs, reward, done, ns3_info = self._gym.step(flat)
#         except Exception as e:
#             print(f"[eval] ⚠ ZMQ exception on step {self._step} "
#                   f"(ns-3 closed after done) — ending episode. ({e})")
#             obs = self._last_obs if self._last_obs is not None else \
#                   np.zeros((N_CELLS, self.obs_dim), np.float32)
#             return obs, np.zeros(N_CELLS, np.float32), True, self._parse_info(None)

#         obs  = self._parse_obs(raw_obs)
#         self._last_obs = obs
#         info = self._parse_info(ns3_info)

#         if self._step == 1:
#             print(f"[eval] [debug] ns3_info type={type(ns3_info).__name__} "
#                   f"repr={repr(ns3_info)[:120]}")

#         done = bool(done) or (self._step >= self.max_steps)
#         return obs, np.full(N_CELLS, float(reward), np.float32), done, info

#     def get_event_mask(self):
#         return np.ones(N_CELLS, dtype=bool)

#     def close(self):
#         try:
#             self._gym.close()
#         except Exception:
#             pass


# # ══════════════════════════════════════════════════════════════════════════
# #  Eval loop
# # ══════════════════════════════════════════════════════════════════════════
# def run_episode(actor, env):
#     obs_all  = env.reset()
#     done     = False
#     ep_r     = 0.0
#     ep_steps = 0
#     last_info = {}

#     while not done:
#         actions = np.zeros((N_CELLS, env.act_dim), np.float32)
#         for c in range(N_CELLS):
#             ot = torch.tensor(obs_all[c], dtype=torch.float32,
#                               device=DEVICE).unsqueeze(0)
#             actions[c] = actor.mean_action(ot).squeeze(0).cpu().numpy()

#         obs_all, rewards, done, info = env.step(actions)
#         ep_r     += float(np.mean(rewards))
#         ep_steps += 1
#         last_info = info

#     ho_acc = int(last_info.get("ho_acc", 0))
#     ho_rej = int(last_info.get("ho_rej", 0))
#     nc_acc = int(last_info.get("nc_acc", 0))
#     nc_rej = int(last_info.get("nc_rej", 0))
#     ho_t   = ho_acc + ho_rej
#     nc_t   = nc_acc + nc_rej
#     all_t  = ho_t + nc_t

#     return {
#         "ep":          ep_steps,
#         "avg_reward":  ep_r / max(1, ep_steps),
#         "n_steps":     ep_steps,
#         "ho_rate":     ho_acc / ho_t  if ho_t  > 0 else 0.0,
#         "nc_rate":     nc_acc / nc_t  if nc_t  > 0 else 0.0,
#         "accept_rate": (ho_acc + nc_acc) / all_t if all_t > 0 else 0.0,
#         "ho_acc": ho_acc, "ho_rej": ho_rej,
#         "nc_acc": nc_acc, "nc_rej": nc_rej,
#         "util":     last_info.get("util",     0.0),
#         "prio_sat": last_info.get("prio_sat", 1.0),
#         "ho_acc_ovl": last_info.get("ho_acc_ovl", 0),
#         "ho_rej_ovl": last_info.get("ho_rej_ovl", 0),
#         "nc_acc_ovl": last_info.get("nc_acc_ovl", 0),
#         "nc_rej_ovl": last_info.get("nc_rej_ovl", 0),
#         "ho_acc_lnd": last_info.get("ho_acc_lnd", 0),
#         "ho_rej_lnd": last_info.get("ho_rej_lnd", 0),
#         "nc_acc_lnd": last_info.get("nc_acc_lnd", 0),
#         "nc_rej_lnd": last_info.get("nc_rej_lnd", 0),
#     }


# def run_eval(actor, env, n_episodes, lam, plot_dir, tag):
#     print(f"\n[eval] ▶ λ={lam}  {n_episodes} episodes …")
#     all_ep = []
#     for i in range(n_episodes):
#         r = run_episode(actor, env)
#         all_ep.append(r)
#         print(f"  ep {i+1:>2}/{n_episodes} | "
#               f"R={r['avg_reward']:+.3f} | "
#               f"HO={r['ho_rate']*100:.1f}% ({r['ho_acc']}✓/{r['ho_rej']}✗) | "
#               f"NC={r['nc_rate']*100:.1f}% ({r['nc_acc']}✓/{r['nc_rej']}✗) | "
#               f"steps={r['n_steps']}")

#     json_path = os.path.join(plot_dir, f"episodes_lam{lam}_{tag}.json")
#     with open(json_path, "w") as f:
#         json.dump(all_ep, f, indent=2)
#     print(f"[eval] ✓ Episodes saved: {json_path}")

#     scalar_keys = ["ho_rate", "nc_rate", "accept_rate", "avg_reward",
#                    "n_steps", "util", "prio_sat"]
#     summary = {k: float(np.mean([e[k] for e in all_ep])) for k in scalar_keys}
#     summary["lambda"] = lam

#     def _rate(acc_key, rej_key):
#         vals = []
#         for e in all_ep:
#             a, r = e[acc_key], e[rej_key]
#             vals.append(r / max(1, a + r))
#         return float(np.mean(vals))

#     summary["ho_drop_ovl"]  = _rate("ho_acc_ovl", "ho_rej_ovl")
#     summary["ho_drop_lnd"]  = _rate("ho_acc_lnd", "ho_rej_lnd")
#     summary["nc_block_ovl"] = _rate("nc_acc_ovl", "nc_rej_ovl")
#     summary["nc_block_lnd"] = _rate("nc_acc_lnd", "nc_rej_lnd")

#     return summary, all_ep


# # ══════════════════════════════════════════════════════════════════════════
# #  Paper-style plots
# # ══════════════════════════════════════════════════════════════════════════
# plt.rcParams.update({
#     "figure.facecolor": "white", "axes.facecolor": "white",
#     "axes.edgecolor":   "#333",  "axes.labelcolor": "#111",
#     "xtick.color":      "#333",  "ytick.color":     "#333",
#     "grid.color":       "#ccc",  "grid.linewidth":  0.6,
#     "grid.linestyle":   "--",    "font.size":       10,
#     "legend.framealpha":0.9,     "legend.edgecolor":"#aaa",
# })

# POL_STYLE = {
#     "Strict Slicing": ("s-", "black"),
#     "Softslice":      ("o-", "#e05050"),
#     "NBS1":           ("^-", "#4488cc"),
#     "PPO (ours)":     ("x-", "#1155aa"),
# }

# def _save_fig(fig, path):
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     fig.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     print(f"[eval] ✓ Saved → {path}")


# def _make_line_fig(title, ylabel, lambdas, ppo_vals,
#                    baseline_vals=None, ylim=None, zero_line=False):
#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     ax.set_title(title)

#     if baseline_vals:
#         for pol, yvals in baseline_vals.items():
#             mk, col = POL_STYLE[pol]
#             ax.plot(lambdas, yvals, mk, color=col, lw=1.8, ms=6, label=pol)

#     mk, col = POL_STYLE["PPO (ours)"]
#     ax.plot(lambdas, ppo_vals, mk, color=col, lw=2.2, ms=8, label="PPO (ours)")

#     if zero_line:
#         ax.axhline(0, color="black", lw=0.8, ls=":", zorder=0)
#     if ylim:
#         ax.set_ylim(*ylim)

#     ax.set_xticks(lambdas)
#     ax.set_xlabel("Arrivals (Vehicles/min)")
#     ax.set_ylabel(ylabel)
#     ax.grid(True)
#     ax.legend()
#     return fig


# def plot_paper_figures(all_summaries, plot_dir, tag):
#     all_summaries = sorted(all_summaries, key=lambda s: s["lambda"])
#     lambdas = [s["lambda"] for s in all_summaries]

#     def _get(key):
#         return [s[key] for s in all_summaries]

#     figures = [
#         ("fig1_resource_utilisation",   "Resource Utilisation vs. Arrival Rate",          "Resource Utilisation (%)",             [v * 100 for v in _get("util")],         dict(ylim=(50, 100))),
#         ("fig2_ho_drop_overloaded",     "Handover Drop Probability — Overloaded Slices",  "HO Drop Probability",                  _get("ho_drop_ovl"),                     {}),
#         ("fig3_ho_drop_lender",         "Handover Drop Probability — Lender Slices",      "HO Drop Probability",                  _get("ho_drop_lnd"),                     {}),
#         ("fig4_nc_block_overloaded",    "New-Call Block Probability — Overloaded Slices", "NC Block Probability",                 _get("nc_block_ovl"),                    {}),
#         ("fig5_nc_block_lender",        "New-Call Block Probability — Lender Slices",     "NC Block Probability",                 _get("nc_block_lnd"),                    {}),
#         ("fig6_average_reward",         "Average Reward vs. Arrival Rate",                "Average Reward",                       _get("avg_reward"),                      dict(zero_line=True)),
#         ("fig7_ho_accept_rate",         "Handover Accept Rate vs. Arrival Rate",          "HO Accept Rate (%)",                   [v * 100 for v in _get("ho_rate")],      {}),
#         ("fig8_nc_accept_rate",         "New-Call Accept Rate vs. Arrival Rate",          "NC Accept Rate (%)",                   [v * 100 for v in _get("nc_rate")],      {}),
#         ("fig9_overall_accept_rate",    "Overall Accept Rate vs. Arrival Rate",           "Accept Rate (%)",                      [v * 100 for v in _get("accept_rate")],  {}),
#     ]

#     for fname, title, ylabel, vals, kwargs in figures:
#         fig = _make_line_fig(title, ylabel, lambdas, vals, **kwargs)
#         _save_fig(fig, os.path.join(plot_dir, f"{fname}_{tag}.png"))

#     print(f"\n[eval] ✅ Paper figures saved to {plot_dir}/")


# # ══════════════════════════════════════════════════════════════════════════
# #  Legacy single-lambda plot
# # ══════════════════════════════════════════════════════════════════════════
# COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# def plot_single(all_ep, lam, save_path):
#     keys   = ["avg_reward", "ho_rate", "nc_rate", "accept_rate"]
#     titles = ["Avg Step Reward", "HO Accept Rate",
#               "NC Accept Rate",  "Overall Accept Rate"]
#     n_ep = len(all_ep)
#     x    = np.arange(1, n_ep + 1)

#     fig, axes = plt.subplots(1, 4, figsize=(20, 4))
#     fig.patch.set_facecolor("#0f1117")
#     fig.suptitle(f"Online Eval (SUMO) — λ={lam} | {n_ep} episodes",
#                  color="white", fontsize=13, fontweight="bold")

#     for idx, (key, title) in enumerate(zip(keys, titles)):
#         ax    = axes[idx]
#         ax.set_facecolor("#1a1d27")
#         for sp in ax.spines.values():
#             sp.set_edgecolor("#2e3142")
#         y     = np.array([e[key] for e in all_ep])
#         color = COLORS[idx]
#         ax.plot(x, y, "o-", color=color, linewidth=2, markersize=7)
#         ax.axhline(y.mean(), color=color, linestyle="--",
#                    linewidth=0.8, alpha=0.6)
#         ax.set_title(title, color="white", fontsize=10)
#         ax.set_xlabel("Episode", color="#8888aa", fontsize=8)
#         ax.tick_params(colors="#8888aa", labelsize=8)
#         ax.grid(alpha=0.15, color="#3a3f55")
#         ax.text(0.97, 0.95, f"μ={y.mean():.3f}",
#                 transform=ax.transAxes, color=color,
#                 fontsize=8, ha="right", va="top")

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
#     fig.savefig(save_path, dpi=150, bbox_inches="tight",
#                 facecolor=fig.get_facecolor())
#     plt.close(fig)
#     print(f"[eval] ✓ Per-λ plot: {save_path}")


# # ══════════════════════════════════════════════════════════════════════════
# #  Main
# # ══════════════════════════════════════════════════════════════════════════
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--checkpoint",     type=str, default=None,
#                    help="Single checkpoint path (single-λ mode)")
#     p.add_argument("--checkpoint_dir", type=str, default="outputs_ns3_eval",
#                    help="Dir with ppo_v48_lam{λ}.pt files (all-λ mode)")
#     p.add_argument("--all_lambdas",    action="store_true",
#                    help="Evaluate all λ values sequentially (auto-launches backends)")
#     p.add_argument("--lambdas",        type=int, nargs="+",
#                    default=[45, 50, 55, 60, 65],
#                    help="List of λ values when --all_lambdas is set")
#     p.add_argument("--lambda_arrival", type=int, default=55,
#                    help="Single λ value (single-λ mode)")
#     p.add_argument("--n_episodes",     type=int, default=10)
#     p.add_argument("--n_steps",        type=int, default=200)
#     p.add_argument("--port",           type=int, default=5555)
#     p.add_argument("--plot_dir",       type=str, default="eval_plots")
#     p.add_argument("--seed",           type=int, default=7777)
#     return p.parse_args()


# def main():
#     args = parse_args()
#     os.makedirs(args.plot_dir, exist_ok=True)
#     tag      = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_list = args.lambdas if args.all_lambdas else [args.lambda_arrival]

#     all_summaries = []
#     ns3_proc = None
#     sumo_proc = None

#     for lam in run_list:
#         # ── Resolve checkpoint ─────────────────────────────────────────────
#         if args.checkpoint and not args.all_lambdas:
#             ckpt_path = args.checkpoint
#         else:
#             ckpt_path = os.path.join(args.checkpoint_dir,
#                                      f"ppo_v48_lam{lam}.pt")

#         if not os.path.exists(ckpt_path):
#             print(f"[eval] ⚠ Not found: {ckpt_path} — skipping λ={lam}")
#             continue

#         # ── Kill previous backends (if any) ───────────────────────────────
#         if args.all_lambdas:
#             if ns3_proc is not None or sumo_proc is not None:
#                 kill_backends(ns3_proc, sumo_proc)
#             ns3_proc, sumo_proc = launch_backends(lam, args.n_steps, args.n_episodes)

#         # ── Load actor ────────────────────────────────────────────────────
#         actor = load_actor(ckpt_path, obs_dim=15, act_dim=N_SLICES)

#         # ── Connect env (fresh connection per lambda) ──────────────────────
#         print(f"\n[eval] {'='*55}")
#         print(f"[eval]  λ = {lam}")
#         print(f"[eval] {'='*55}")
#         env = OnlineNS3Env(port=args.port, max_steps=args.n_steps, seed=args.seed)

#         # ── Run evaluation ────────────────────────────────────────────────
#         summary, all_ep = run_eval(actor, env, args.n_episodes, lam,
#                                    args.plot_dir, tag)
#         env.close()
#         all_summaries.append(summary)

#         # ── Per-λ dark plot ───────────────────────────────────────────────
#         plot_single(all_ep, lam,
#                     save_path=os.path.join(args.plot_dir,
#                                            f"eval_lam{lam}_{tag}.png"))

#         # ── Print summary ─────────────────────────────────────────────────
#         print(f"\n[eval] λ={lam} summary:")
#         print(f"  Reward      : {summary['avg_reward']:+.4f}")
#         print(f"  HO Accept   : {summary['ho_rate']*100:.1f}%")
#         print(f"  NC Accept   : {summary['nc_rate']*100:.1f}%")
#         print(f"  Accept Rate : {summary['accept_rate']*100:.1f}%")
#         print(f"  Util        : {summary['util']*100:.1f}%")
#         print(f"  HO Drop OVL : {summary['ho_drop_ovl']*100:.2f}%")
#         print(f"  HO Drop LND : {summary['ho_drop_lnd']*100:.2f}%")
#         print(f"  NC Blk  OVL : {summary['nc_block_ovl']*100:.2f}%")
#         print(f"  NC Blk  LND : {summary['nc_block_lnd']*100:.2f}%")

#     # ── Kill final backends ────────────────────────────────────────────────
#     if args.all_lambdas and (ns3_proc is not None or sumo_proc is not None):
#         kill_backends(ns3_proc, sumo_proc)

#     # ── Paper-style figures ────────────────────────────────────────────────
#     if len(all_summaries) >= 2:
#         plot_paper_figures(all_summaries, args.plot_dir, tag)
#     elif len(all_summaries) == 1:
#         print("[eval] Only 1 λ evaluated — skipping multi-λ paper figures.")
#         print("       Run with --all_lambdas to get all paper-style plots.")

#     print(f"\n[eval] ✅ Done. All plots saved in {args.plot_dir}/")


# if __name__ == "__main__":
#     main()


# #!/usr/bin/env python3
# """
# bridge/sumo_bridge_v49.py  —  vehicle-mobility model with max_load lowered to 75–80%
# """

# import argparse
# import json
# import os
# import time
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple

# import numpy as np

# # ── Constants (must match gnb_scheduler CAPS[] and PPO slice mapping) ──
# N_CELLS   = 3
# N_SLICES  = 4
# CAPS      = [0.25, 0.40, 0.10, 0.25]   # per-slice RB fractions

# # Vehicle mobility
# SPEED_MS       = 13.89          # 50 km/h
# CELL_LENGTH_M  = 1000.0
# DT_S           = 1.0



# # Boost NC flow probabilities to balance traffic
# NC_FLOWS_PER_SLICE: Dict[int, List[Tuple[str, float]]] = {
#     0: [("AUT_video_s0", 0.85)],          # was 0.70
#     1: [("AUT_video_s1", 0.80), ("AUT_voip_s1", 0.75)],  # was 0.65, 0.60
#     2: [("AUT_voip_s2", 0.80), ("INFO_be_s2", 0.70)],    # was 0.65, 0.55
#     3: [("NC_data_s3", 0.85), ("INFO_be_s3", 0.72)],     # was 0.70, 0.58
# }

# # Longer flow lifetimes = more sustained load = higher util
# FLOW_DEFS = {
#     "TOD_video_s0": dict(bw=0.042, mean_life=120, slice_idx=0, is_ho=True),   # was 80
#     "AUT_video_s0": dict(bw=0.038, mean_life=120, slice_idx=0, is_ho=False),
#     "TOD_video_s1": dict(bw=0.060, mean_life=120, slice_idx=1, is_ho=True),
#     "TOD_data_s1":  dict(bw=0.035, mean_life=90,  slice_idx=1, is_ho=True),
#     "AUT_video_s1": dict(bw=0.055, mean_life=120, slice_idx=1, is_ho=False),
#     "AUT_voip_s1":  dict(bw=0.020, mean_life=160, slice_idx=1, is_ho=False),
#     "AUT_voip_s2":  dict(bw=0.013, mean_life=130, slice_idx=2, is_ho=False),
#     "INFO_be_s2":   dict(bw=0.009, mean_life=60,  slice_idx=2, is_ho=False),
#     "NC_data_s3":   dict(bw=0.048, mean_life=120, slice_idx=3, is_ho=False),
#     "INFO_be_s3":   dict(bw=0.012, mean_life=60,  slice_idx=3, is_ho=False),
# }

# HO_FLOW_PER_SLICE: Dict[int, str] = {
#     0: "TOD_video_s0",
#     1: "TOD_video_s1",
#     2: "AUT_voip_s2",
#     3: "INFO_be_s3",
# }

# LAMBDA_LEVELS = [45, 50, 55, 60, 65]

# # FIX 1: extra HO spawning probability (independent of mobility)
# HO_TRIGGER_PROB = 0.03   # was 0.15


# @dataclass
# class Flow:
#     flow_type: str
#     bw: float
#     slice_idx: int
#     is_ho: bool
#     remaining: int


# @dataclass
# class Vehicle:
#     veh_id: str
#     cell: int
#     pos_in_cell: float
#     flows: List[Flow]
#     active: bool = True

#     def to_arrival(self, f: Flow, is_ho: bool) -> dict:
#         return {
#             "veh_id":    self.veh_id,
#             "flow_type": f.flow_type,
#             "bw":        f.bw,
#             "is_ho":     is_ho,
#             "slice_idx": f.slice_idx,
#             "remaining": f.remaining,
#         }


# class VehicleMobilitySimulator:
#     def __init__(self, lambda_arrival: float, max_load: float,
#                  seed: int, n_steps: int):
#         self.lam      = lambda_arrival
#         self.lam_s    = lambda_arrival / 60.0
#         self.max_load = max_load
#         self.n_steps  = n_steps
#         self.rng      = np.random.default_rng(seed)
#         self._vehicles: Dict[str, Vehicle] = {}
#         self._veh_counter = 0
#         self._load_acc   = np.zeros((N_CELLS, N_SLICES), dtype=np.float64)
#         self._load_ticks = 0

#     def reset(self, seed: Optional[int] = None):
#         if seed is not None:
#             self.rng = np.random.default_rng(seed)
#         self._vehicles.clear()
#         self._veh_counter = 0
#         self._load_acc[:] = 0.0
#         self._load_ticks  = 0

#     def step(self, tick: int, episode: int) -> dict:
#         cell_arrivals:   List[List[dict]] = [[] for _ in range(N_CELLS)]
#         cell_ho_export:  List[List[dict]] = [[] for _ in range(N_CELLS)]

#         # 1. New vehicle arrivals
#         n_new = int(self.rng.poisson(self.lam_s * DT_S))
#         for _ in range(n_new):
#             veh = self._spawn_vehicle(tick)
#             for f in veh.flows:
#                 cell_arrivals[veh.cell].append(veh.to_arrival(f, is_ho=False))

#         # 2. Move existing vehicles & age flows
#         to_remove = []
#         for vid, veh in self._vehicles.items():
#             if not veh.active:
#                 to_remove.append(vid)
#                 continue

#             alive = []
#             for f in veh.flows:
#                 f.remaining -= 1
#                 if f.remaining > 0:
#                     alive.append(f)
#             veh.flows = alive

#             if not veh.flows:
#                 veh.active = False
#                 to_remove.append(vid)
#                 continue

#             veh.pos_in_cell += SPEED_MS * DT_S
#             if veh.pos_in_cell >= CELL_LENGTH_M:
#                 veh.pos_in_cell -= CELL_LENGTH_M
#                 new_cell = veh.cell + 1
#                 if new_cell >= N_CELLS:
#                     veh.active = False
#                     to_remove.append(vid)
#                     continue
#                 for f in veh.flows:
#                     ho_rec = self._make_ho_record(veh, f)
#                     cell_arrivals[new_cell].append(ho_rec)
#                     cell_ho_export[new_cell].append(ho_rec)
#                 veh.cell = new_cell

#         for vid in to_remove:
#             self._vehicles.pop(vid, None)

#         # 3. Extra HO spawning (independent of mobility)
#         for vid, veh in self._vehicles.items():
#             if not veh.active:
#                 continue
#             if self.rng.random() < HO_TRIGGER_PROB:
#                 s = int(self.rng.integers(0, N_SLICES))
#                 ho_type = HO_FLOW_PER_SLICE[s]
#                 fd = FLOW_DEFS[ho_type]
#                 cell_arrivals[veh.cell].append({
#                     "veh_id": vid,
#                     "flow_type": ho_type,
#                     "bw": fd["bw"],
#                     "is_ho": True,
#                     "slice_idx": s,
#                     "remaining": 10,
#                 })

#         cell_load = self._compute_cell_load()
#         self._load_acc  += cell_load
#         self._load_ticks += 1

#         cells_data = {}
#         for c in range(N_CELLS):
#             cells_data[str(c)] = {
#                 "arrivals":  cell_arrivals[c],
#                 "ho_export": cell_ho_export[c],
#                 "cell_load": cell_load[c].tolist(),
#             }

#         avg_load = cell_load.mean(axis=0).tolist()
#         return {
#             "tick":    tick,
#             "episode": episode,
#             "cells":   cells_data,
#             "load":    avg_load,
#         }

#     def _spawn_vehicle(self, tick: int) -> Vehicle:
#         vid = f"v{self._veh_counter:06d}_t{tick}"
#         self._veh_counter += 1
#         flows = self._sample_flows(vid)
#         veh = Vehicle(
#             veh_id=vid,
#             cell=int(self.rng.integers(0, N_CELLS)),
#             pos_in_cell=float(self.rng.uniform(0, CELL_LENGTH_M)),
#             flows=flows,
#         )
#         self._vehicles[vid] = veh
#         return veh

#     def _sample_flows(self, vid: str) -> List[Flow]:
#         flows = []
#         for s in range(N_SLICES):
#             for ftype, prob in NC_FLOWS_PER_SLICE[s]:
#                 if self.rng.random() < prob:
#                     fd = FLOW_DEFS[ftype]
#                     life = max(10, int(self.rng.geometric(1.0 / fd["mean_life"])))
#                     life = min(life, 200)
#                     flows.append(Flow(
#                         flow_type=ftype,
#                         bw=fd["bw"],
#                         slice_idx=fd["slice_idx"],
#                         is_ho=False,
#                         remaining=life,
#                     ))

#         for s in [0, 1]:
#             has_slice = any(f.slice_idx == s for f in flows)
#             if not has_slice:
#                 ftype, _ = NC_FLOWS_PER_SLICE[s][0]
#                 fd = FLOW_DEFS[ftype]
#                 life = max(10, int(self.rng.geometric(1.0 / fd["mean_life"])))
#                 life = min(life, 200)
#                 flows.append(Flow(ftype, fd["bw"], s, False, life))

#         if not flows:
#             fd = FLOW_DEFS["NC_data_s3"]
#             life = max(10, int(self.rng.geometric(1.0 / fd["mean_life"])))
#             life = min(life, 200)
#             flows.append(Flow("NC_data_s3", fd["bw"], 3, False, life))

#         return flows

#     def _make_ho_record(self, veh: Vehicle, f: Flow) -> dict:
#         ho_type = HO_FLOW_PER_SLICE.get(f.slice_idx, "TOD_video_s1")
#         fd = FLOW_DEFS[ho_type]
#         return {
#             "veh_id":    veh.veh_id,
#             "flow_type": ho_type,
#             "bw":        fd["bw"],
#             "is_ho":     True,
#             "slice_idx": f.slice_idx,
#             "remaining": max(1, min(f.remaining, 200)),
#         }

#     def _compute_cell_load(self) -> np.ndarray:
#         bw_used = np.zeros((N_CELLS, N_SLICES), dtype=np.float64)
#         for veh in self._vehicles.values():
#             if not veh.active:
#                 continue
#             for f in veh.flows:
#                 bw_used[veh.cell, f.slice_idx] += f.bw

#         # Per-slice max loads: priority slices near/over capacity,
#         # non-priority slices have spare to donate (matches paper mechanism)
#         load = np.zeros((N_CELLS, N_SLICES), dtype=np.float64)
#         for s in range(N_SLICES):
#             if CAPS[s] > 0:
                load[:, s] = bw_used[:, s] / CAPS[s]
#         return load.astype(np.float32)


# class SumoBridge:
#     def __init__(
#         self,
#         lambda_arrival: float,
#         n_steps:        int,
#         n_episodes:     int,
#         seed:           int,
#         max_load:       float,
#         report_dir:     str,
#     ):
#         self.lam        = lambda_arrival
#         self.n_steps    = n_steps
#         self.n_episodes = n_episodes
#         self.seed       = seed
#         self.max_load   = max_load
#         self.base_dir   = report_dir
#         self.report_dir = os.path.join(report_dir, f"lam{int(lambda_arrival)}")
#         os.makedirs(self.report_dir, exist_ok=True)

#         self.sim = VehicleMobilitySimulator(
#             lambda_arrival=lambda_arrival,
#             max_load=max_load,
#             seed=seed,
#             n_steps=n_steps,
#         )

#     def run_offline(self):
#         print(
#             f"[sumo_bridge_v49] OFFLINE  λ={self.lam}  "
#             f"episodes={self.n_episodes}  steps={self.n_steps}  "
#             f"→ {self.report_dir}"
#         )
#         total_files = 0

#         for ep in range(self.n_episodes):
#             ep_seed = self.seed + ep * 7919
#             self.sim.reset(seed=ep_seed)

#             ho_total = nc_total = 0
#             last_report = None

#             for tick in range(self.n_steps):
#                 report = self.sim.step(tick, ep)
#                 last_report = report

#                 for c in range(N_CELLS):
#                     for a in report["cells"][str(c)]["arrivals"]:
#                         if a["is_ho"]:
#                             ho_total += 1
#                         else:
#                             nc_total += 1

#                 fname = f"ep{ep:04d}_tick{tick:04d}.json"
#                 with open(os.path.join(self.report_dir, fname), "w") as f:
#                     json.dump(report, f)
#                 total_files += 1

#             if ep % 50 == 0 and last_report is not None:
#                 avg_load = last_report["load"]
#                 util_bw  = sum(avg_load[s] * CAPS[s] for s in range(N_SLICES))
#                 util_pct = util_bw / sum(CAPS) * 100

#                 n_veh = len(self.sim._vehicles)
#                 print(
#                     f"  ep={ep:4d}  HO={ho_total:5d}  NC={nc_total:5d}  "
#                     f"HO/NC={ho_total/(nc_total+1):.2f}  "
#                     f"active_veh={n_veh}  "
#                     f"slice_load={[f'{avg_load[s]:.2f}' for s in range(N_SLICES)]}  "
#                     f"util={util_pct:.1f}%"
#                 )
#                 ho_total = nc_total = 0

#         print(f"[sumo_bridge_v49] Done. {total_files} files → {self.report_dir}")


# def _sanity_check(lam: float = 55.0, n_steps: int = 200, seed: int = 42):
#     print(f"\n[sanity v49] λ={lam}  steps={n_steps}")
#     sim = VehicleMobilitySimulator(lam, max_load=2.0, seed=seed, n_steps=n_steps)
#     sim.reset(seed=seed)

#     ho_total = nc_total = 0
#     bw_nc_total = bw_ho_total = 0.0

#     for t in range(n_steps):
#         report = sim.step(t, 0)
#         for c in range(N_CELLS):
#             for a in report["cells"][str(c)]["arrivals"]:
#                 if a["is_ho"]:
#                     ho_total += 1; bw_ho_total += a["bw"]
#                 else:
#                     nc_total += 1; bw_nc_total += a["bw"]

#     print(f"  Total HO arrivals : {ho_total}   avg bw/step/cell: {bw_ho_total/(n_steps*N_CELLS):.4f}")
#     print(f"  Total NC arrivals : {nc_total}   avg bw/step/cell: {bw_nc_total/(n_steps*N_CELLS):.4f}")
#     print(f"  HO/NC ratio       : {ho_total/(nc_total+1):.3f}")

#     slice_bw_nc = [0.0] * N_SLICES
#     sim2 = VehicleMobilitySimulator(lam, max_load=2.0, seed=seed, n_steps=n_steps)
#     sim2.reset(seed=seed)
#     for t in range(n_steps):
#         report = sim2.step(t, 0)
#         for c in range(N_CELLS):
#             for a in report["cells"][str(c)]["arrivals"]:
#                 if not a["is_ho"]:
#                     slice_bw_nc[a["slice_idx"]] += a["bw"]
#     print("\n  NC bandwidth per slice (total across all steps & cells):")
#     for s in range(N_SLICES):
#         avg = slice_bw_nc[s] / (n_steps * N_CELLS)
#         pct = avg / CAPS[s] * 100 if CAPS[s] > 0 else 0
#         status = "✓" if 70 < pct < 110 else ("⚠ LOW" if pct < 70 else "⚠ SATURATED")
#         print(f"    Slice {s}: avg_bw={avg:.4f}  cap={CAPS[s]:.2f}  "
#               f"utilisation≈{pct:.1f}%  {status}")
#     print(f"\n  Active vehicles at end: {len(sim2._vehicles)}")


# def main():
#     ap = argparse.ArgumentParser(
#         description="v49 vehicle-mobility tick generator (targeting 75–80% utilisation)"
#     )
#     ap.add_argument("--lambda_arrival", type=float, default=None)
#     ap.add_argument("--n_steps",        type=int,   default=200)
#     ap.add_argument("--n_episodes",     type=int,   default=300)
#     ap.add_argument("--seed",           type=int,   default=42)
#     ap.add_argument("--max_load",       type=float, default=None)
#     ap.add_argument("--report_dir",     type=str,   default="bridge/tick_reports_live")
#     ap.add_argument("--sanity_check",   action="store_true")
#     args = ap.parse_args()

#     if args.sanity_check:
#         lam = args.lambda_arrival or 55.0
#         _sanity_check(lam=lam, n_steps=args.n_steps)
#         return

#     target_lams = (
#         [int(args.lambda_arrival)]
#         if args.lambda_arrival is not None
#         else LAMBDA_LEVELS
#     )

#     for lam in target_lams:
#         # FIX 1: lower target utilisation to 75–80% for all λ
#         max_load = args.max_load or min(0.80, 0.65 + 0.008 * (lam - 45))
#         bridge = SumoBridge(
#             lambda_arrival=float(lam),
#             n_steps=args.n_steps,
#             n_episodes=args.n_episodes,
#             seed=args.seed,
#             max_load=max_load,
#             report_dir=args.report_dir,
#         )
#         bridge.run_offline()


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
