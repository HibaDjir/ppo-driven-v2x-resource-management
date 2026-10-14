#!/usr/bin/env python3
"""
ppo_v43.py — Clean PPO trainer:
- Trains purely on C++ reward from gnb_scheduler (no Python-side shaping)
- OBS_PER_CELL=22 with Lagrange observations
- What you monitor is exactly what the agent optimizes
"""

import os, time, warnings
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "ns3"))

try:
    from ns3_env import NS3MultiCellEnv, UTIL_TARGET, UTIL_TARGET_LOW
except ImportError:
    from ns3_env import NS3MultiCellEnv
    UTIL_TARGET     = 0.99
    UTIL_TARGET_LOW = 0.98

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_LEVELS = [45, 50, 55, 60, 65]
LAM_MIN       = float(min(LAMBDA_LEVELS))
LAM_MAX       = float(max(LAMBDA_LEVELS))

# OBS_PER_CELL = 22 (19 base + 3 Lagrange) + 1 lambda feature = 23 total
_BASE_OBS_PER_CELL = 22
LAM_FEATURE        = 1
OBS_PER_CELL       = _BASE_OBS_PER_CELL + LAM_FEATURE   # = 23

HP = {
    "lr_actor":          1e-4,
    "lr_critic":         3e-4,
    "clip":              0.15,
    "gamma":             0.99,
    "gae_lam":           0.95,
    "target_kl":         0.08,
    "n_epochs":          6,
    "batch_size":        256,
    "vf_coef":           0.50,
    "max_grad":          0.50,
    "ent_coef_start":    0.05,
    "ent_coef_end":      0.01,
    "cycle_events":      200_000,
    "total_events":      800_000,
    "n_events":          32_000,
    "hidden":            (256, 256, 128),
    "actor_bias_init":   0.0,
    "actor_log_std":     -0.1,
    "obs_norm_clip":     5.0,
    "vf_clip":           20.0,
    "lam_switch_every":  8,
    "norm_warmup_steps": 0,
    "warmup_ticks":      3,
}

# Thresholds matching gnb_scheduler.cc
HO_THRESH      = 0.060
NC_THRESH      = 0.070
LEND_NC_TARGET = 0.070
LEND_HO_TARGET = 0.060

_IDX_USED  = 4
_IDX_ALLOC = 8


def _lam_norm(lam: float) -> float:
    return (lam - LAM_MIN) / max(LAM_MAX - LAM_MIN, 1.0)


def _append_lam(obs_cells: np.ndarray, lam: float) -> np.ndarray:
    n = obs_cells.shape[0]
    ln = np.full((n, 1), _lam_norm(lam), dtype=np.float32)
    out = np.concatenate([obs_cells, ln], axis=1)
    assert out.shape[1] == OBS_PER_CELL, \
        f"Expected {OBS_PER_CELL} obs dims, got {out.shape[1]}"
    return out


class ObsNormalizer:
    def __init__(self, obs_dim: int, clip: float = 5.0):
        self.clip   = clip
        self.mean   = np.zeros(obs_dim, dtype=np.float64)
        self.var    = np.full(obs_dim, 0.1, dtype=np.float64)
        self.count  = 512.0
        self._steps = 512

    def update(self, obs: np.ndarray):
        batch = obs.reshape(-1, obs.shape[-1]).astype(np.float64)
        if batch.shape[1] > 1:
            batch = batch[:, :-1]   # skip lambda feature
        n = batch.shape[0]
        bm = batch.mean(0); bv = batch.var(0)
        tot = self.count + n; d = bm - self.mean[:-1]
        self.mean[:-1] = self.mean[:-1] + d * n / tot
        self.var[:-1]  = (self.var[:-1] * self.count + bv * n +
                          d**2 * self.count * n / tot) / tot
        self.count = tot
        self._steps += n

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        if self._steps < HP["norm_warmup_steps"]:
            return obs.astype(np.float32)
        std = np.sqrt(self.var + 1e-8).astype(np.float32)
        out = (obs.astype(np.float32) - self.mean.astype(np.float32)) / std
        out = np.clip(out, -self.clip, self.clip).astype(np.float32)
        out[..., -1] = obs[..., -1]   # restore lambda feature unnormalised
        return out

    def state_dict(self):
        return {"mean": self.mean, "var": self.var,
                "count": self.count, "steps": self._steps}

    def load_state_dict(self, d):
        self.mean   = d["mean"]; self.var = d["var"]
        self.count  = d["count"]
        self._steps = int(d.get("steps", 512))


def _mlp(in_d, hidden, out_d, gain, bias_init=0.0):
    layers, d = [], in_d
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    out_layer = nn.Linear(d, out_d)
    nn.init.orthogonal_(out_layer.weight, gain=gain)
    nn.init.constant_(out_layer.bias, bias_init)
    layers.append(out_layer)
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear) and m is not out_layer:
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    return net


class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.5

    def __init__(self, obs_dim, act_dim, hidden=HP["hidden"]):
        super().__init__()
        self.net     = _mlp(obs_dim, hidden, act_dim, 0.01,
                            bias_init=HP["actor_bias_init"])
        self.log_std = nn.Parameter(torch.full((act_dim,), HP["actor_log_std"]))

    def _dist(self, obs):
        mu  = self.net(obs)
        std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return Normal(mu, std)

    def act(self, obs):
        d  = self._dist(obs); z = d.rsample(); u = torch.tanh(z)
        a  = (u + 1.0) * 0.5
        lp = d.log_prob(z) - torch.log(1.0 - u.pow(2) + 1e-6)
        return a, lp.sum(-1)

    def evaluate(self, obs, action):
        d  = self._dist(obs)
        u  = (action * 2.0 - 1.0).clamp(-1 + 1e-6, 1 - 1e-6)
        z  = torch.atanh(u)
        lp = d.log_prob(z) - torch.log(1.0 - u.pow(2) + 1e-6)
        return lp.sum(-1), d.entropy().sum(-1)

    @torch.no_grad()
    def mean_action(self, obs):
        u = torch.tanh(self.net(obs))
        return ((u + 1.0) * 0.5).clamp(0.0, 1.0)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=HP["hidden"]):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, 1, 1.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class RolloutBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.clear()

    def clear(self):
        self.obs=[]; self.act=[]; self.rew=[]; self.don=[]
        self.lp=[]; self.val=[]
        self.ptr = 0

    def add(self, o, a, r, d, lp, v):
        self.obs.append(o.copy()); self.act.append(a.copy())
        self.rew.append(float(r)); self.don.append(float(d))
        self.lp.append(float(lp)); self.val.append(float(v))
        self.ptr += 1

    def is_full(self): return self.ptr >= self.max_size

    def compute_gae(self, last_v, gamma=0.99, lam=0.95):
        n   = self.ptr
        obs = np.array(self.obs[:n], np.float32)
        act = np.array(self.act[:n], np.float32)
        rew = np.array(self.rew[:n], np.float32)
        don = np.array(self.don[:n], np.float32)
        lp  = np.array(self.lp[:n],  np.float32)
        val = np.array(self.val[:n], np.float32)
        adv = np.zeros(n, np.float32); g = 0.0
        for t in reversed(range(n)):
            nv  = last_v if t == n-1 else val[t+1]
            nnt = 1.0 - don[t]
            delta = rew[t] + gamma * nv * nnt - val[t]
            g = delta + gamma * lam * nnt * g
            adv[t] = g
        return obs, act, lp, adv, adv+val, val

    def batches(self, obs, act, lp_arr, adv, ret, old_val, bs):
        n   = obs.shape[0]; idx = np.random.permutation(n)
        for s in range(0, n, bs):
            b = idx[s:s+bs]
            yield (torch.tensor(obs[b],    device=DEVICE),
                   torch.tensor(act[b],    device=DEVICE),
                   torch.tensor(lp_arr[b], device=DEVICE),
                   torch.tensor(adv[b],    device=DEVICE),
                   torch.tensor(ret[b],    device=DEVICE),
                   torch.tensor(old_val[b],device=DEVICE))


class LambdaSampler:
    def __init__(self, levels, rng_seed=42):
        self.levels = levels
        self.rng    = np.random.default_rng(rng_seed)
        self.counts = np.zeros(len(levels), dtype=int)

    def sample(self) -> float:
        inv     = 1.0 / (self.counts + 1.0)
        weights = inv / inv.sum()
        idx     = int(self.rng.choice(len(self.levels), p=weights))
        self.counts[idx] += 1
        return float(self.levels[idx])


class UnifiedPPOTrainer:
    N_CELLS = 3

    def __init__(self, args, seed=42):
        torch.manual_seed(seed); np.random.seed(seed)
        self.args = args
        self.seed = seed
        self.act_dim = 4
        self.obs_dim = OBS_PER_CELL   # 23

        self.n_events    = HP["n_events"]
        self.n_epochs    = HP["n_epochs"]
        self.batch_size  = HP["batch_size"]
        self.gamma       = HP["gamma"]
        self.gae_lam     = HP["gae_lam"]
        self.clip        = HP["clip"]
        self.vf_coef     = HP["vf_coef"]
        self.vf_clip     = HP["vf_clip"]
        self.max_grad    = HP["max_grad"]
        self.target_kl   = HP["target_kl"]
        self.total_events   = args.total_events
        self.cycle_events   = HP["cycle_events"]
        self.ent_coef_start = HP["ent_coef_start"]
        self.ent_coef_end   = HP["ent_coef_end"]
        self.done_events    = 0

        self.actor  = Actor(self.obs_dim, self.act_dim, HP["hidden"]).to(DEVICE)
        self.critic = Critic(self.obs_dim, HP["hidden"]).to(DEVICE)
        self.opt_a  = optim.Adam(self.actor.parameters(),
                                 lr=HP["lr_actor"], eps=1e-5)
        self.opt_c  = optim.Adam(self.critic.parameters(),
                                 lr=HP["lr_critic"], eps=1e-5)

        anneal = lambda _: max(1.0 - self.done_events / self.total_events, 0.05)
        self.sch_a = optim.lr_scheduler.LambdaLR(self.opt_a, anneal)
        self.sch_c = optim.lr_scheduler.LambdaLR(self.opt_c, anneal)

        self.buf = [RolloutBuffer(self.n_events // self.N_CELLS)
                    for _ in range(self.N_CELLS)]
        self.obs_norm    = ObsNormalizer(self.obs_dim, clip=HP["obs_norm_clip"])
        self.lam_sampler = LambdaSampler(LAMBDA_LEVELS, rng_seed=seed)

        self._env      = None
        self._cur_lam  = float(LAMBDA_LEVELS[0])
        self._ep_count = 0

        self.reward_log_updates = []; self.reward_log_values = []
        self.ep_rewards=[]; self.ep_utils=[]; self.ep_ho=[]
        self.ep_nc=[]; self.ep_ps=[]
        self.ep_lend_nc=[]

        self._ep_r     = np.zeros(self.N_CELLS)
        self._ep_steps = np.zeros(self.N_CELLS, dtype=int)
        self._ep_keys  = ["util_served", "prio_sat", "ho_rate", "nc_rate",
                          "sf_covered", "accept_rate",
                          "teleop_ho_rate", "teleop_nc_rate",
                          "auto_ho_rate", "auto_nc_rate",
                          "lender_nc_blk", "lender_ho_blk"]
        self._ep = {k: [[] for _ in range(self.N_CELLS)]
                    for k in self._ep_keys}
        self._upd_count = 0

    def _ent_coef(self):
        p = min(1.0, self.done_events / self.total_events)
        return self.ent_coef_start + (self.ent_coef_end - self.ent_coef_start) * p

    def _make_env(self, lam, port):
        max_load = min(1.80, 1.60 + 0.008 * (lam - 45))
        return NS3MultiCellEnv(
            report_dir=self.args.report_dir, lambda_arrival=lam,
            max_steps=self.args.n_steps, seed=self.seed,
            max_load=max_load, port=port)

    def _open_env(self, lam, port_offset):
        if self._env is not None:
            try: self._env.close()
            except: pass
            self._env = None; time.sleep(2)
        port = self.args.port + port_offset
        self._cur_lam = lam
        self._env = self._make_env(lam, port)
        print(f"[PPO] λ={lam:.0f} port={port} events={self.done_events:,}")

    def _aug(self, obs_raw):
        return _append_lam(obs_raw, self._cur_lam)

    def _query(self, obs_norm):
        ot = torch.tensor(obs_norm, dtype=torch.float32,
                          device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            a, lp = self.actor.act(ot)
            v     = self.critic(ot)
        return a.squeeze(0).cpu().numpy(), float(lp.item()), float(v.item())

    def _collect(self, obs_all_aug):
        for b in self.buf:
            b.clear()
        self.actor.eval(); self.critic.eval()

        while not all(b.is_full() for b in self.buf):
            self.obs_norm.update(obs_all_aug)
            obs_norm = self.obs_norm.normalize(obs_all_aug)

            ev_pre  = self._env.get_event_mask()
            actions = np.zeros((self.N_CELLS, self.act_dim), np.float32)
            per_cell = {}
            for c in range(self.N_CELLS):
                a, lp_c, v_c = self._query(obs_norm[c])
                actions[c] = a
                per_cell[c] = (lp_c, v_c)

            next_obs_raw, r_ns3, done, info, ev_mask = self._env.step(actions)

            # ── Diagnostic print every 100 steps
            if self._ep_steps[0] % 100 == 0 and self._ep_steps[0] > 0:
                print("\n==== Diagnostic Admission Stats ====")
                for c in range(self.N_CELLS):
                    ci = info["cell_infos"][c]
                    alloc = ci['_alloc_obs']
                    used  = ci['_used_norm']
                    actual = [round(used[s]*alloc[s], 3) for s in range(4)]
                    lend_ho = round(float(ci.get('lender_ho_blk', -1.0)), 4)
                    lend_nc = round(float(ci.get('lender_nc_blk', -1.0)), 4)
                    is_lend = ci.get('is_lender', False)
                    is_borr = ci.get('is_borrower', False)
                    print(f"c{c}: ho_acc={ci['ho_acc']} ho_rej={ci['ho_rej']} "
                          f"nc_acc={ci['nc_acc']} nc_rej={ci['nc_rej']} "
                          f"alloc={[round(x,2) for x in alloc]} "
                          f"used={[round(x,2) for x in used]} "
                          f"actual_bw={actual} "
                          f"lender={is_lend} borrower={is_borr} "
                          f"lend_ho_blk={lend_ho} lend_nc_blk={lend_nc}")
                print("====================================\n")

            next_obs_aug = self._aug(next_obs_raw)
            n_ev = int(ev_mask.sum())

            for c in range(self.N_CELLS):
                ci = info["cell_infos"][c]
                self._ep_steps[c] += 1
                for key in self._ep_keys:
                    self._ep[key][c].append(float(ci.get(key, 0.0)))

                # ── No Python shaping: train purely on C++ reward
                # r_ns3 is already what the C++ scheduler computed via GetReward()
                if self._ep_steps[c] >= HP["warmup_ticks"]:
                    self._ep_r[c] += r_ns3

                if (self._ep_steps[c] >= HP["warmup_ticks"]
                        and not self.buf[c].is_full()):
                    lp_c, v_c = per_cell[c]
                    self.buf[c].add(obs_norm[c], actions[c], r_ns3,
                                    float(done), lp_c, v_c)

            self.done_events += max(1, n_ev)

            if done:
                self._flush_episode()
                self._ep_count += 1

                if self._ep_count % HP["lam_switch_every"] == 0:
                    new_lam    = self.lam_sampler.sample()
                    port_off   = LAMBDA_LEVELS.index(int(new_lam))
                    if new_lam != self._cur_lam:
                        for b in self.buf:
                            if b.ptr > 0:
                                b.don[-1] = 1.0
                        self._open_env(new_lam, port_off)
                        break

                next_obs_raw = self._env.reset()
                next_obs_aug = self._aug(next_obs_raw)
                self._reset_ep_accumulators()

            obs_all_aug = next_obs_aug

        obs_norm_last = self.obs_norm.normalize(obs_all_aug)
        last_vs = []
        for c in range(self.N_CELLS):
            ot = torch.tensor(obs_norm_last[c], dtype=torch.float32,
                              device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                last_vs.append(float(self.critic(ot).item()))
        self.actor.train(); self.critic.train()
        return obs_all_aug, last_vs

    def _update(self, last_vs):
        all_obs, all_act, all_lp, all_adv, all_ret, all_oval = \
            [], [], [], [], [], []
        for c in range(self.N_CELLS):
            o, a, lp, adv_c, ret_c, ov = self.buf[c].compute_gae(
                last_vs[c], self.gamma, self.gae_lam)
            all_obs.append(o);  all_act.append(a)
            all_lp.append(lp);  all_adv.append(adv_c)
            all_ret.append(ret_c); all_oval.append(ov)

        obs     = np.concatenate(all_obs)
        act     = np.concatenate(all_act)
        lp_arr  = np.concatenate(all_lp)
        adv     = np.concatenate(all_adv)
        ret     = np.concatenate(all_ret)
        old_val = np.concatenate(all_oval)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ec  = self._ent_coef(); approx_kl = 0.0

        for _ in range(self.n_epochs):
            for ob, ac, lp_old, adv_b, ret_b, ov_b in self.buf[0].batches(
                    obs, act, lp_arr, adv, ret, old_val, self.batch_size):
                new_lp, ent = self.actor.evaluate(ob, ac)
                ratio = torch.exp(new_lp - lp_old)
                with torch.no_grad():
                    approx_kl = float(
                        ((ratio - 1) - (new_lp - lp_old)).mean())
                p1 = -adv_b * ratio
                p2 = -adv_b * ratio.clamp(1 - self.clip, 1 + self.clip)
                pi_loss = torch.max(p1, p2).mean() - ec * ent.mean()
                self.opt_a.zero_grad(); pi_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),
                                         self.max_grad)
                self.opt_a.step()

                val_new     = self.critic(ob)
                val_clipped = ov_b + (val_new - ov_b).clamp(
                    -self.vf_clip, self.vf_clip)
                vl = self.vf_coef * 0.5 * torch.max(
                    (val_new  - ret_b)**2,
                    (val_clipped - ret_b)**2).mean()
                self.opt_c.zero_grad(); vl.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                         self.max_grad)
                self.opt_c.step()

            if approx_kl > self.target_kl:
                break

        self.sch_a.step(); self.sch_c.step()

    def _flush_episode(self):
        for c in range(self.N_CELLS):
            n_steps = max(1, self._ep_steps[c])
            self.ep_rewards.append(float(self._ep_r[c]) / n_steps)
            def _m(k): return float(np.mean(self._ep[k][c])) \
                if self._ep[k][c] else 0.0
            self.ep_utils.append(_m("util_served"))
            self.ep_ho.append(_m("ho_rate"))
            self.ep_nc.append(_m("nc_rate"))
            self.ep_ps.append(_m("prio_sat"))
            self.ep_lend_nc.append(_m("lender_nc_blk"))

    def _reset_ep_accumulators(self):
        self._ep_r     = np.zeros(self.N_CELLS)
        self._ep_steps = np.zeros(self.N_CELLS, dtype=int)
        self._ep = {k: [[] for _ in range(self.N_CELLS)]
                    for k in self._ep_keys}

    def train(self):
        self._open_env(float(LAMBDA_LEVELS[0]), 0)
        obs_raw     = self._env.reset()
        obs_all_aug = self._aug(obs_raw)
        upd = 0

        while self.done_events < self.total_events:
            obs_all_aug, last_vs = self._collect(obs_all_aug)
            upd += 1; self._upd_count = upd
            self._update(last_vs)

            n = len(self.ep_rewards); w = min(n, 20)
            r_val = float(np.mean(self.ep_rewards[-w:])) if n else 0.0
            self.reward_log_updates.append(upd)
            self.reward_log_values.append(r_val)

            if upd % 5 == 0:
                us  = float(np.mean(self.ep_utils[-w:]))    if n else 0.0
                ho  = float(np.mean(self.ep_ho[-w:]))       if n else 0.0
                nc  = float(np.mean(self.ep_nc[-w:]))       if n else 0.0
                ps  = float(np.mean(self.ep_ps[-w:]))       if n else 0.0
                lnc = float(np.mean(self.ep_lend_nc[-w:])) if n else 0.0
                print(f"Upd {upd:4d} | ev={self.done_events:>7,} | "
                      f"λ={self._cur_lam:.0f} | R={r_val:+6.3f} | "
                      f"util={us:.3f} | HO={ho:.3f} NC={nc:.3f} | "
                      f"LendNC_blk={lnc:.3f} | "
                      f"ps={ps:.3f} | H={self._ent_coef():.3f}")

            if upd % 50 == 0:
                ckpt = os.path.join(
                    self.args.out_dir, f"ppo_ckpt_upd{upd}.pt")
                torch.save({
                    "actor":    self.actor.state_dict(),
                    "critic":   self.critic.state_dict(),
                    "obs_norm": self.obs_norm.state_dict(),
                }, ckpt)
                print(f"Saved {ckpt}")

        if self._env: self._env.close()
        print("Training complete.")
        return self

    def load_weights(self, path, resume_events=None):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        if "obs_norm" in ckpt:
            self.obs_norm.load_state_dict(ckpt["obs_norm"])
        if resume_events is not None:
            self.done_events = resume_events
            print(f"Resuming with done_events = {self.done_events}")
        print(f"Loaded {path}")


def _safe(ci, key, default=0.0):
    return ci.get(key, default)


def _eval_episode_loop(actor, obs_norm, env, lam):
    obs_raw  = env.reset()
    done     = False
    ep_r     = 0.0
    ep_steps = 0
    last_info = None

    lend_ho_blk_sum = 0.0
    lend_nc_blk_sum = 0.0
    borr_ho_blk_sum = 0.0
    borr_nc_blk_sum = 0.0
    n_lender_ticks  = 0
    n_borr_ticks    = 0

    while not done:
        obs_aug = _append_lam(obs_raw, lam)
        obs_n   = obs_norm.normalize(obs_aug)
        ev_mask = env.get_event_mask()
        actions = np.zeros((env.N_CELLS, 4), np.float32)
        for c in range(env.N_CELLS):
            if ev_mask[c]:
                ot = torch.tensor(obs_n[c], dtype=torch.float32,
                                  device=DEVICE).unsqueeze(0)
                actions[c] = actor.mean_action(ot).squeeze(0).cpu().numpy()

        obs_raw, r_ns3, done, info, _ = env.step(actions)
        ep_r    += r_ns3
        ep_steps += 1
        last_info = info

        if ep_steps <= HP["warmup_ticks"]:
            continue

        for ci in info["cell_infos"]:
            if _safe(ci, "is_lender", False):
                lend_ho_blk_sum += float(ci.get("lender_ho_blk", 0.0))
                lend_nc_blk_sum += float(ci.get("lender_nc_blk", 0.0))
                n_lender_ticks  += 1
            if _safe(ci, "is_borrower", False):
                borr_ho_blk_sum += 1.0 - float(ci.get("ho_rate", 1.0))
                borr_nc_blk_sum += 1.0 - float(ci.get("nc_rate", 1.0))
                n_borr_ticks    += 1

    result = {
        "avg_reward":      ep_r / max(1, ep_steps),
        "util_served":     last_info.get("util_served", 0.0) if last_info else 0.0,
        "util_served_pct": (last_info.get("util_served", 0.0)*100) if last_info else 0.0,
        "ho_rate":         last_info.get("ho_rate", 0.0) if last_info else 0.0,
        "nc_rate":         last_info.get("nc_rate", 0.0) if last_info else 0.0,
        "lend_ho_blk":     lend_ho_blk_sum / max(1, n_lender_ticks),
        "lend_nc_blk":     lend_nc_blk_sum / max(1, n_lender_ticks),
        "borr_ho_blk":     borr_ho_blk_sum / max(1, n_borr_ticks),
        "borr_nc_blk":     borr_nc_blk_sum / max(1, n_borr_ticks),
    }
    return result


def evaluate_all_lambdas(actor, obs_norm, args, n_ep=40, seed=7777):
    results = {}
    for lam in LAMBDA_LEVELS:
        max_load = min(1.80, 1.60 + 0.008 * (lam - 45))
        env = NS3MultiCellEnv(
            report_dir=args.report_dir, lambda_arrival=float(lam),
            max_steps=args.n_steps, seed=seed, max_load=max_load,
            port=args.port + 99 + LAMBDA_LEVELS.index(lam))
        print(f"Evaluating λ={lam} ...", end=" ", flush=True)
        eps = [_eval_episode_loop(actor, obs_norm, env, float(lam))
               for _ in range(n_ep)]
        env.close(); time.sleep(3)
        avg = {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
        results[lam] = avg
        print(f"util={avg['util_served_pct']:.1f}%  "
              f"HO_acc={avg['ho_rate']:.3f} NC_acc={avg['nc_rate']:.3f}  "
              f"borr_HO_blk={avg['borr_ho_blk']*100:.1f}%  "
              f"borr_NC_blk={avg['borr_nc_blk']*100:.1f}%  "
              f"lend_HO_blk={avg['lend_ho_blk']*100:.1f}%  "
              f"lend_NC_blk={avg['lend_nc_blk']*100:.1f}%  "
              f"R={avg['avg_reward']:.3f}")
    return results


def plot_all(eval_results, trainer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lams = sorted(eval_results.keys())
    ppo  = [eval_results[l] for l in lams]
    def _v(m): return [ppo[i][m] for i in range(len(lams))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lams, _v("util_served_pct"), "o-", label="PPO")
    ax.axhline(UTIL_TARGET*100, ls="--", color="red", label="Target")
    ax.set_xlabel("λ (veh/s)"); ax.set_ylabel("Util (%)")
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "fig1_resource_utilisation.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lams, [100*b for b in _v("borr_ho_blk")], "o-",
            label="Borrower HO block rate")
    ax.axhline(6, ls="--", color="red", label="Target <6%")
    ax.set_xlabel("λ (veh/s)"); ax.set_ylabel("HO block rate (%)")
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "fig2_ho_drop_overloaded.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lams, [100*b for b in _v("lend_ho_blk")], "s-",
            label="Lender HO block rate")
    ax.set_xlabel("λ (veh/s)"); ax.set_ylabel("HO block rate (%)")
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "fig3_ho_drop_lender.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lams, [100*b for b in _v("borr_nc_blk")], "o-",
            label="Borrower NC block rate")
    ax.axhline(7, ls="--", color="red", label="Target <7%")
    ax.set_xlabel("λ (veh/s)"); ax.set_ylabel("NC block rate (%)")
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "fig4_nc_block_overloaded.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lams, [100*b for b in _v("lend_nc_blk")], "s-",
            label="Lender NC block rate")
    ax.axhline(7, ls="--", color="red", label="Target <7%")
    ax.set_xlabel("λ (veh/s)"); ax.set_ylabel("NC block rate (%)")
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "fig5_nc_block_lender.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lams, _v("avg_reward"), "o-", label="Average reward")
    ax.set_xlabel("λ (veh/s)"); ax.set_ylabel("Mean episode reward")
    ax.legend(); ax.grid(True)
    fig.savefig(os.path.join(out_dir, "fig6_average_reward.png"), dpi=150)
    plt.close()

    if trainer.ep_lend_nc:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(trainer.ep_lend_nc, alpha=0.4, color="gray", label="raw")
        w = 20
        if len(trainer.ep_lend_nc) >= w:
            smooth = np.convolve(trainer.ep_lend_nc,
                                 np.ones(w)/w, mode="valid")
            ax.plot(range(w-1, len(trainer.ep_lend_nc)),
                    smooth, color="black", linewidth=2, label="smoothed")
        ax.axhline(LEND_NC_TARGET, ls="--", color="red",
                   label=f"Target {LEND_NC_TARGET:.0%}")
        ax.set_xlabel("Episode"); ax.set_ylabel("Lender NC block rate")
        ax.set_title("Lender NC Blocking — Training Progress")
        ax.legend(); ax.grid(True)
        fig.savefig(os.path.join(out_dir,
                    "fig7_lender_nc_convergence.png"), dpi=150)
        plt.close()

    if trainer.reward_log_updates:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(trainer.reward_log_updates, trainer.reward_log_values)
        ax.set_xlabel("Update"); ax.set_ylabel("Episode reward (C++ only)")
        ax.set_title("Training convergence — pure C++ reward signal")
        fig.savefig(os.path.join(out_dir,
                    "fig15_training_convergence.png"), dpi=150)
        plt.close()

    print(f"Figures saved to {out_dir}")


class StdoutRedirector:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_dir",    default="bridge/tick_reports_live")
    ap.add_argument("--total_events",  type=int, default=HP["total_events"])
    ap.add_argument("--cycle_events",  type=int, default=HP["cycle_events"])
    ap.add_argument("--n_steps",       type=int, default=200)
    ap.add_argument("--out_dir",       default="outputs_v43")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--eval_episodes", type=int, default=40)
    ap.add_argument("--port",          type=int, default=5561)
    ap.add_argument("--skip_train",    action="store_true")
    ap.add_argument("--skip_eval",     action="store_true")
    ap.add_argument("--resume_checkpoint", type=str, default=None)
    ap.add_argument("--resume_events",     type=int, default=None)
    ap.add_argument("--log_file",          type=str, default=None)
    args = ap.parse_args()

    original_stdout = sys.stdout
    log_redirector  = None
    if args.log_file:
        log_redirector = StdoutRedirector(args.log_file)
        sys.stdout     = log_redirector
        print(f"Logging to {args.log_file}")
        print("Command line:", " ".join(sys.argv))

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, "ppo_final.pt")

    trainer = UnifiedPPOTrainer(args, seed=args.seed)

    if args.resume_checkpoint:
        trainer.load_weights(args.resume_checkpoint,
                             resume_events=args.resume_events)

    if not args.skip_train:
        trainer.train()
        torch.save({
            "actor":    trainer.actor.state_dict(),
            "critic":   trainer.critic.state_dict(),
            "obs_norm": trainer.obs_norm.state_dict(),
        }, ckpt_path)
    else:
        if not args.resume_checkpoint:
            trainer.load_weights(ckpt_path)

    if not args.skip_eval:
        eval_results = evaluate_all_lambdas(
            trainer.actor, trainer.obs_norm, args,
            n_ep=args.eval_episodes)
        plot_all(eval_results, trainer, args.out_dir)

    if log_redirector is not None:
        log_redirector.close()
        sys.stdout = original_stdout