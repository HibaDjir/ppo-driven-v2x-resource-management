#!/usr/bin/env python3
"""
eval_online.py — Real-time eval that drives sumo_bridge via ZMQ.

Saves 6 figures:
    fig1_resource_utilisation.png
    fig2_ho_drop_overloaded.png      ← HO drop for the most-loaded cell
    fig3_ho_drop_lender.png          ← HO drop averaged over lender cells
    fig4_nc_block_overloaded.png     ← NC block for the most-loaded cell
    fig5_nc_block_lender.png         ← NC block averaged over lender cells
    fig6_average_reward.png
"""

import argparse
import json
import os
import sys
import warnings
import collections
import glob

import numpy as np
import torch
import torch.nn as nn
import zmq

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Constants (must match sumo_bridge.py exactly) ──────────────────────────
N_CELLS = 3
N_SLICES = 4
CAPS = np.array([0.23, 0.38, 0.13, 0.26], dtype=np.float32)
LAMBDA_LEVELS = [45, 50, 55, 60, 65]
LAM_MIN = float(min(LAMBDA_LEVELS))
LAM_MAX = float(max(LAMBDA_LEVELS))

_BASE_OBS_PER_CELL = 22
OBS_PER_CELL = _BASE_OBS_PER_CELL + 1
ACT_DIM = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Obs builder ────────────────────────────────────────────────────────────
class CellObsBuilder:
    def __init__(self, alpha: float = 0.05):
        self.alpha  = alpha
        self.ho_blk = np.zeros((N_CELLS, N_SLICES), dtype=np.float32)
        self.nc_blk = np.zeros((N_CELLS, N_SLICES), dtype=np.float32)

    def reset(self):
        self.ho_blk[:] = 0.0
        self.nc_blk[:] = 0.0

    def update(self, tick_data: dict, actions: np.ndarray):
        cells = tick_data["cells"]
        obs   = np.zeros((N_CELLS, _BASE_OBS_PER_CELL), dtype=np.float32)

        for c in range(N_CELLS):
            cd = cells[str(c)]
            obs[c, 0:4] = np.array(cd["cell_load"], dtype=np.float32)

            ho_bw  = np.zeros(N_SLICES, dtype=np.float32)
            nc_bw  = np.zeros(N_SLICES, dtype=np.float32)

            for arr in cd["arrivals"]:
                s  = int(arr["slice_idx"])
                bw = float(arr["bw"])
                if arr["is_ho"]:
                    ho_bw[s] += bw
                else:
                    nc_bw[s] += bw

            obs[c, 4:8]  = ho_bw
            obs[c, 8:12] = nc_bw

            # Replicate C++ Admit(): spare = alloc[s] - used[s]
            alloc = actions[c] * CAPS
            used  = obs[c, 0:4] * CAPS  # cell_load * CAPS = absolute used BW

            ho_acc = np.zeros(N_SLICES)
            ho_rej = np.zeros(N_SLICES)
            nc_acc = np.zeros(N_SLICES)
            nc_rej = np.zeros(N_SLICES)

            sim_used = used.copy()
            for arr in cd["arrivals"]:
                s  = int(arr["slice_idx"])
                bw = float(arr["bw"])
                spare = alloc[s] - sim_used[s]
                if arr["is_ho"]:
                    if spare >= bw - 1e-8:
                        ho_acc[s] += 1; sim_used[s] += bw
                    else:
                        ho_rej[s] += 1
                else:
                    if spare >= bw - 1e-8:
                        nc_acc[s] += 1; sim_used[s] += bw
                    else:
                        nc_rej[s] += 1

            for s in range(N_SLICES):
                ht = ho_acc[s] + ho_rej[s]
                nt = nc_acc[s] + nc_rej[s]
                new_ho = (ho_rej[s] / ht) if ht > 0 else 0.0
                new_nc = (nc_rej[s] / nt) if nt > 0 else 0.0
                self.ho_blk[c, s] = (1 - self.alpha) * self.ho_blk[c, s] + self.alpha * new_ho
                self.nc_blk[c, s] = (1 - self.alpha) * self.nc_blk[c, s] + self.alpha * new_nc

        return obs


# ── ObsNormalizer ──────────────────────────────────────────────────────────
class ObsNormalizer:
    def __init__(self, obs_dim: int, clip: float = 5.0):
        self.clip = clip
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.full(obs_dim, 0.1, dtype=np.float64)
        self.count = 512.0
        self._steps = 512

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        std = np.sqrt(self.var + 1e-8).astype(np.float32)
        out = (obs.astype(np.float32) - self.mean.astype(np.float32)) / std
        out = np.clip(out, -self.clip, self.clip).astype(np.float32)
        out[..., -1] = obs[..., -1]
        return out

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]
        self._steps = int(d.get("steps", 512))


# ── Actor ──────────────────────────────────────────────────────────────────
def _mlp(in_d, hidden, out_d, gain):
    layers, d = [], in_d
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    out_layer = nn.Linear(d, out_d)
    nn.init.orthogonal_(out_layer.weight, gain=gain)
    nn.init.constant_(out_layer.bias, 0.0)
    layers.append(out_layer)
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear) and m is not out_layer:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    return net


class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_PER_CELL, act_dim=ACT_DIM,
                 hidden=(256, 256, 128)):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, act_dim, 0.01)

    @torch.no_grad()
    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.net(obs))
        return ((u + 1.0) * 0.5).clamp(0.0, 1.0)


# ── Checkpoint loader ──────────────────────────────────────────────────────
def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    actor = Actor().to(DEVICE)
    actor.load_state_dict(ckpt["actor"], strict=False)
    actor.eval()

    obs_norm = ObsNormalizer(OBS_PER_CELL)
    if "obs_norm" in ckpt:
        obs_norm.load_state_dict(ckpt["obs_norm"])
        print(f"[eval] ObsNormalizer loaded (steps={obs_norm._steps:,})")
    else:
        print("[eval] WARNING: no obs_norm in checkpoint")

    print(f"[eval] Actor loaded  →  {path}  (device={DEVICE})")
    return actor, obs_norm


# ── ZMQ sockets ───────────────────────────────────────────────────────────
def make_zmq_sockets(pull_port: int, ack_port: int, host: str = "localhost"):
    ctx = zmq.Context()
    pull = ctx.socket(zmq.PULL)
    pull.setsockopt(zmq.RCVTIMEO, 30_000)
    pull.connect(f"tcp://{host}:{pull_port}")

    ack = ctx.socket(zmq.PUSH)
    ack.setsockopt(zmq.SNDHWM, 1)
    ack.connect(f"tcp://{host}:{ack_port}")

    print(f"[eval] ZMQ PULL → {host}:{pull_port}   ACK PUSH → {host}:{ack_port}")
    return ctx, pull, ack


# ── Lambda helpers ─────────────────────────────────────────────────────────
def lam_norm(lam: float) -> float:
    return (lam - LAM_MIN) / max(LAM_MAX - LAM_MIN, 1.0)


def append_lam(obs: np.ndarray, lam: float) -> np.ndarray:
    ln = np.full((obs.shape[0], 1), lam_norm(lam), dtype=np.float32)
    return np.concatenate([obs, ln], axis=1)


# ── Single episode ─────────────────────────────────────────────────────────
def run_episode(actor, obs_norm, pull_sock, ack_sock,
                obs_builder, lam: float, n_steps: int, ep_idx: int) -> dict:

    obs_builder.reset()
    actions = np.ones((N_CELLS, ACT_DIM), dtype=np.float32) * 0.5

    ep_util_sum = 0.0
    # Per-cell accumulators: shape (N_CELLS, N_SLICES)
    ep_ho_blk_percell = np.zeros((N_CELLS, N_SLICES), dtype=np.float64)
    ep_nc_blk_percell = np.zeros((N_CELLS, N_SLICES), dtype=np.float64)
    ep_load_percell = np.zeros(N_CELLS, dtype=np.float64)
    ep_reward = 0.0
    n_ticks = 0

    for tick_i in range(n_steps):
        try:
            raw = pull_sock.recv_string()
        except zmq.Again:
            print(f"[eval] TIMEOUT waiting for tick {tick_i} in ep {ep_idx}")
            break

        tick_data = json.loads(raw)
        obs_base = obs_builder.update(tick_data, actions)
        obs_aug = append_lam(obs_base, lam)
        obs_n = obs_norm.normalize(obs_aug)

        new_actions = np.zeros((N_CELLS, ACT_DIM), dtype=np.float32)
        for c in range(N_CELLS):
            t = torch.tensor(obs_n[c], dtype=torch.float32,
                             device=DEVICE).unsqueeze(0)
            new_actions[c] = actor.mean_action(t).squeeze(0).cpu().numpy()
        actions = new_actions

        # Per-cell mean load (averaged over slices)
        cell_loads = np.array([
            np.mean(tick_data["cells"][str(c)]["cell_load"])
            for c in range(N_CELLS)
        ], dtype=np.float64)

        util = float(np.mean(cell_loads))
        ep_util_sum += util
        ep_load_percell += cell_loads
        ep_ho_blk_percell += obs_builder.ho_blk  # (N_CELLS, N_SLICES)
        ep_nc_blk_percell += obs_builder.nc_blk

        ho_blk_tick = obs_builder.ho_blk.mean(axis=0)
        nc_blk_tick = obs_builder.nc_blk.mean(axis=0)
        penalty = (ho_blk_tick.mean() * 2.0 + nc_blk_tick.mean()) * 0.5
        reward = util - penalty
        ep_reward += reward
        n_ticks += 1

        ack_sock.send(b'K')

    if n_ticks == 0:
        return {}

    # Average over ticks
    ho_blk_mean_percell = ep_ho_blk_percell / n_ticks  # (N_CELLS, N_SLICES)
    nc_blk_mean_percell = ep_nc_blk_percell / n_ticks
    load_mean_percell = ep_load_percell / n_ticks  # (N_CELLS,)

    # Identify overloaded cell (highest mean load) vs lender cells (the rest)
    overloaded_idx = int(np.argmax(load_mean_percell))
    lender_idxs = [c for c in range(N_CELLS) if c != overloaded_idx]

    ho_drop_overloaded = float(np.mean(ho_blk_mean_percell[overloaded_idx]))
    nc_block_overloaded = float(np.mean(nc_blk_mean_percell[overloaded_idx]))

    ho_drop_lender = float(np.mean(ho_blk_mean_percell[lender_idxs]))
    nc_block_lender = float(np.mean(nc_blk_mean_percell[lender_idxs]))

    # Overall (all cells) for backward compat
    ho_blk_mean = ho_blk_mean_percell.mean(axis=0)
    nc_blk_mean = nc_blk_mean_percell.mean(axis=0)

    return {
        "avg_reward": ep_reward / n_ticks,
        "util_served_pct": ep_util_sum / n_ticks * 100,
        "ho_blk": ho_blk_mean.tolist(),
        "nc_blk": nc_blk_mean.tolist(),
        "ho_rate": float(1.0 - np.mean(ho_blk_mean)),
        "nc_rate": float(1.0 - np.mean(nc_blk_mean)),
        "ho_drop": float(np.mean(ho_blk_mean)),
        "nc_block": float(np.mean(nc_blk_mean)),
        # ── NEW per-role fields ──────────────────────────────────────────
        "ho_drop_overloaded": ho_drop_overloaded,
        "nc_block_overloaded": nc_block_overloaded,
        "ho_drop_lender": ho_drop_lender,
        "nc_block_lender": nc_block_lender,
    }


# ── Evaluate one lambda ────────────────────────────────────────────────────
def evaluate(actor, obs_norm, args):
    ctx, pull_sock, ack_sock = make_zmq_sockets(
        args.pull_port, args.ack_port, args.host
    )
    obs_builder = CellObsBuilder()
    lam = float(args.lambda_arrival)
    results = []

    print(f"\n[eval] λ={lam}  episodes={args.n_episodes}  steps={args.n_steps}")
    print("[eval] Waiting for first tick from sumo_bridge …\n")

    for ep_i in range(args.n_episodes):
        r = run_episode(actor, obs_norm, pull_sock, ack_sock,
                        obs_builder, lam, args.n_steps, ep_i)
        if not r:
            print(f"[eval] ep {ep_i} empty — bridge may have stopped")
            break
        results.append(r)
        print(f"  ep {ep_i + 1:>3}/{args.n_episodes} | "
              f"util={r['util_served_pct']:5.1f}% | "
              f"HO_acc={r['ho_rate'] * 100:5.1f}% | "
              f"NC_acc={r['nc_rate'] * 100:5.1f}% | "
              f"R={r['avg_reward']:+.3f}")

    pull_sock.close()
    ack_sock.close()
    ctx.term()
    print("[eval] ZMQ closed.")
    return results


# ── Summary (per lambda) ───────────────────────────────────────────────────
def print_summary(results: list, lam: float):
    if not results:
        print("[eval] No results to summarise.")
        return
    keys = ("avg_reward", "util_served_pct", "ho_rate", "nc_rate",
            "ho_drop", "nc_block",
            "ho_drop_overloaded", "nc_block_overloaded",
            "ho_drop_lender", "nc_block_lender")
    avg = {k: float(np.mean([r[k] for r in results if k in r])) for k in keys}
    print(f"\n[eval] ══ SUMMARY  λ={lam} ══")
    print(f"  Utilisation      : {avg['util_served_pct']:.1f}%")
    print(f"  HO Accept        : {avg['ho_rate'] * 100:.1f}%")
    print(f"  NC Accept        : {avg['nc_rate'] * 100:.1f}%")
    print(f"  HO Drop          : {avg['ho_drop'] * 100:.1f}%")
    print(f"  NC Block         : {avg['nc_block'] * 100:.1f}%")
    print(f"  HO Drop (overld) : {avg['ho_drop_overloaded'] * 100:.1f}%")
    print(f"  NC Blk  (overld) : {avg['nc_block_overloaded'] * 100:.1f}%")
    print(f"  HO Drop (lender) : {avg['ho_drop_lender'] * 100:.1f}%")
    print(f"  NC Blk  (lender) : {avg['nc_block_lender'] * 100:.1f}%")
    print(f"  Avg Reward       : {avg['avg_reward']:+.3f}")


# ── Save per-lambda JSON ───────────────────────────────────────────────────
def save_json(results: list, lam: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"lam{int(lam)}_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Raw results → {path}")


# ── Load all saved JSONs from out_dir ──────────────────────────────────────
def load_all_results(out_dir: str) -> dict:
    all_res = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "lam*_results.json"))):
        fname = os.path.basename(path)
        lam = int(fname.replace("lam", "").replace("_results.json", ""))
        with open(path) as f:
            all_res[lam] = json.load(f)
    return all_res


# ── Plot helpers ───────────────────────────────────────────────────────────
def _single_plot(lambdas, means, title, ylabel, fname, out_dir,
                 color="#1155cc", is_pct=True):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lambdas, means,
            color=color, marker="o", markersize=7,
            linewidth=2.0, zorder=3)

    for lam, val in zip(lambdas, means):
        pct = "%" if is_pct else ""
        ax.annotate(f"{val:.2f}{pct}",
                    xy=(lam, val),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=color)

    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"λ={l}" for l in lambdas], fontsize=9)
    ax.set_xlabel("Traffic Load (λ arrivals/s)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.35, linewidth=0.6)
    fig.tight_layout()

    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[eval] Saved → {path}")


# ── 6 combined figures (one metric each) ──────────────────────────────────
def plot_combined_figures(all_results: dict, out_dir: str):
    """
    Saves the six figures with the exact filenames:

        fig1_resource_utilisation.png
        fig2_ho_drop_overloaded.png
        fig3_ho_drop_lender.png
        fig4_nc_block_overloaded.png
        fig5_nc_block_lender.png
        fig6_average_reward.png

    For fig2-fig5 the script tries the new per-role keys first.
    If the stored JSONs pre-date this fix (missing those keys), it falls
    back gracefully to the overall ho_drop / nc_block values.
    """
    os.makedirs(out_dir, exist_ok=True)
    lambdas = sorted(all_results.keys())

    def means(key, scale=1.0, fallback=None):
        vals = []
        for lam in lambdas:
            eps = all_results[lam]
            if any(key in r for r in eps):
                v = float(np.mean([r[key] * scale for r in eps if key in r]))
            elif fallback is not None:
                v = float(np.mean([r[fallback] * scale for r in eps if fallback in r]))
            else:
                v = 0.0
            vals.append(v)
        return vals

    # ── fig1: Resource Utilisation ─────────────────────────────────────────
    _single_plot(
        lambdas,
        means("util_served_pct"),
        title="Resource Utilisation vs Traffic Load",
        ylabel="Utilisation (%)",
        fname="fig1_resource_utilisation.png",
        out_dir=out_dir,
        color="#1155cc",
        is_pct=True,
    )

    # ── fig2: HO Drop — Overloaded Cell ───────────────────────────────────
    _single_plot(
        lambdas,
        means("ho_drop_overloaded", scale=100.0, fallback="ho_drop"),
        title="HO Drop Rate — Overloaded Cell vs Traffic Load",
        ylabel="HO drop rate, overloaded cell (%)",
        fname="fig2_ho_drop_overloaded.png",
        out_dir=out_dir,
        color="#cc2222",
        is_pct=True,
    )

    # ── fig3: HO Drop — Lender Cells ──────────────────────────────────────
    _single_plot(
        lambdas,
        means("ho_drop_lender", scale=100.0, fallback="ho_drop"),
        title="HO Drop Rate — Lender Cells vs Traffic Load",
        ylabel="HO drop rate, lender cells (%)",
        fname="fig3_ho_drop_lender.png",
        out_dir=out_dir,
        color="#e07020",
        is_pct=True,
    )

    # ── fig4: NC Block — Overloaded Cell ──────────────────────────────────
    _single_plot(
        lambdas,
        means("nc_block_overloaded", scale=100.0, fallback="nc_block"),
        title="NC Block Rate — Overloaded Cell vs Traffic Load",
        ylabel="NC block rate, overloaded cell (%)",
        fname="fig4_nc_block_overloaded.png",
        out_dir=out_dir,
        color="#7722cc",
        is_pct=True,
    )

    # ── fig5: NC Block — Lender Cells ─────────────────────────────────────
    _single_plot(
        lambdas,
        means("nc_block_lender", scale=100.0, fallback="nc_block"),
        title="NC Block Rate — Lender Cells vs Traffic Load",
        ylabel="NC block rate, lender cells (%)",
        fname="fig5_nc_block_lender.png",
        out_dir=out_dir,
        color="#229977",
        is_pct=True,
    )

    # ── fig6: Average Reward ──────────────────────────────────────────────
    _single_plot(
        lambdas,
        means("avg_reward"),
        title="Average Reward vs Traffic Load",
        ylabel="Avg reward",
        fname="fig6_average_reward.png",
        out_dir=out_dir,
        color="#1155cc",
        is_pct=False,
    )


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--lambda_arrival", type=float, default=55)
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--n_steps", type=int, default=200)
    ap.add_argument("--pull_port", type=int, default=5556)
    ap.add_argument("--ack_port", type=int, default=5557)
    ap.add_argument("--host", type=str, default="localhost")
    ap.add_argument("--out_dir", type=str, default="eval_zmq_results")
    ap.add_argument("--plots_only", action="store_true",
                    help="Skip ZMQ eval; regenerate combined figure from saved JSONs")
    args = ap.parse_args()

    if not args.plots_only:
        actor, obs_norm = load_checkpoint(args.checkpoint)
        results = evaluate(actor, obs_norm, args)
        print_summary(results, args.lambda_arrival)
        if results:
            save_json(results, args.lambda_arrival, args.out_dir)

    all_results = load_all_results(args.out_dir)
    present = sorted(all_results.keys())
    expected = sorted(LAMBDA_LEVELS)

    if present == expected:
        print(f"\n[eval] All λ levels present {present} → generating combined figure …")
        plot_combined_figures(all_results, args.out_dir)
    else:
        missing = sorted(set(expected) - set(present))
        print(f"\n[eval] Combined figure will be generated once all λ levels are done.")
        print(f"       Present : {present}")
        print(f"       Missing : {missing}")

    print("[eval] Done.")


   