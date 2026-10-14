from __future__ import annotations

import json
import os
import subprocess
import time
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    from ns3gym import ns3env as _ns3env_mod
    _NS3GYM_BACKEND = "zmq"
    _NS3GYM_AVAILABLE = True
except ImportError:
    try:
        from gym_http_client import Client as _GymHttpClient
        _NS3GYM_BACKEND = "http"
        _NS3GYM_AVAILABLE = True
    except ImportError:
        _NS3GYM_BACKEND = None
        _NS3GYM_AVAILABLE = False
        warnings.warn(
            "[ns3_env] Neither ns3gym nor gym_http_client found.",
            ImportWarning, stacklevel=2,
        )

N_CELLS      = 3
N_SLICES     = 4


OBS_PER_CELL = 22           # was 19
OBS_DIM      = N_CELLS * OBS_PER_CELL   # = 66
ACT_PER_CELL = 4
ACT_DIM      = N_CELLS * ACT_PER_CELL

CAPS        = np.array([0.23, 0.38, 0.13, 0.26], dtype=np.float32)
SLICE_ALPHA = np.array([0.50, 1.00, 0.0,  0.0 ], dtype=np.float32)

P_H = 1.0
P_N = 0.6

_ALPHA_SUM          = float(SLICE_ALPHA[0] + SLICE_ALPHA[1])
BORROW_SHARE_TELEOP = SLICE_ALPHA[1] / _ALPHA_SUM
BORROW_SHARE_AUTO   = SLICE_ALPHA[0] / _ALPHA_SUM

IS_PRIORITY_SLICE = [True, True, False, False]

_IDX_LOAD     = 0
_IDX_USED     = 4
_IDX_ALLOC    = 8
_IDX_SINR     = 12
_IDX_HO_ARR   = 13
_IDX_OVERLOAD = 14
_IDX_HO_EMA   = 15
_IDX_NC_EMA   = 16
# NEW Lagrange obs indices
_IDX_LAG_LEND_NC = 19
_IDX_LAG_BORR_HO = 20
_IDX_LEND_NC_EMA = 21

UTIL_TARGET      = 0.99
UTIL_TARGET_LOW  = 0.98

_NS3_WAF_DEFAULT = "/home/hiba/ns-allinone-3.40/ns-3.40/ns3"


class RunningNorm:
    def __init__(self, alpha: float = 0.005):
        self.mean  = 0.0
        self.std   = 1.0
        self.alpha = alpha

    def __call__(self, r: float) -> float:
        self.mean += self.alpha * (r - self.mean)
        self.std   = max(0.5, self.std * (1 - self.alpha)
                         + self.alpha * abs(r - self.mean))
        return (r - self.mean) / self.std


class NS3MultiCellEnv:
    N_CELLS = N_CELLS
    obs_dim = OBS_PER_CELL
    act_dim = ACT_PER_CELL

    def __init__(
        self,
        report_dir:      str   = "bridge/tick_reports",
        lambda_arrival:  float = 55.0,
        max_steps:       int   = 200,
        seed:            int   = 42,
        max_load:        float = 1.25,
        port:            int   = 5555,
        ns3_waf:         str   = "",
        ns3_scratch_dir: str   = "",
        startSim:        bool  = True,
    ):
        if not _NS3GYM_AVAILABLE:
            raise RuntimeError("[ns3_env] ns3-gym is not installed.")

        self.lambda_arrival = float(lambda_arrival)
        self.max_steps      = int(max_steps)
        self.seed           = int(seed)
        self.max_load       = float(max_load)
        self.port           = int(port)
        self.startSim       = startSim

        base = os.path.abspath(report_dir)
        lam_subdir = os.path.join(base, f"lam{int(lambda_arrival)}")
        self.report_dir = lam_subdir if os.path.isdir(lam_subdir) else base

        self.ns3_waf = ns3_waf or os.environ.get("NS3_WAF", _NS3_WAF_DEFAULT)
        self.ns3_scratch_dir = (
            ns3_scratch_dir or
            os.path.dirname(os.path.abspath(self.ns3_waf))
        )

        self._proc:     Optional[subprocess.Popen] = None
        self._gym:      Any  = None
        self._log_file: Any  = None
        self._episode       = 0
        self._step          = 0
        self._last_obs = np.zeros((N_CELLS, OBS_PER_CELL), np.float32)

        if self.startSim:
            self._launch()

    def reset(self) -> np.ndarray:
        self._episode += 1
        self._step    = 0
        self._launch()
        time.sleep(0.3)
        raw = self._gym.ns3ZmqBridge.obsData
        if raw is None:
            raw = self._gym.ns3ZmqBridge.get_obs()
        obs = self._parse_obs(raw)
        self._last_obs = obs
        return obs

    def step(self, actions: np.ndarray):
        self._step += 1
        flat_act = np.asarray(actions, dtype=np.float32).flatten()
        if flat_act.size != ACT_DIM:
            raise ValueError(f"Expected {ACT_DIM} actions, got {flat_act.size}")
        flat_act = np.clip(flat_act, 0.0, 1.0)

        raw_obs, ns3_reward, ns3_done, ns3_info = self._gym.step(flat_act)
        next_obs = self._parse_obs(raw_obs)
        self._last_obs = next_obs

        cell_infos, ev_mask = self._parse_info(ns3_info, next_obs)
        info = self._build_info(cell_infos, next_obs)

        done = bool(ns3_done) or (self._step >= self.max_steps)
        return next_obs, float(ns3_reward), done, info, ev_mask

    def get_event_mask(self) -> np.ndarray:
        mask = np.zeros(N_CELLS, dtype=bool)
        for c in range(N_CELLS):
            obs = self._last_obs[c]
            mask[c] = obs[_IDX_OVERLOAD] > 0.5
        return mask

    def close(self):
        self._teardown()

    def __del__(self):
        try:
            self._teardown()
        except Exception:
            pass

    def _build_ns3_cmd(self) -> list:
        binary = (
            "/home/hiba/ns-allinone-3.40/ns-3.40/build/scratch/"
            "gnb_scheduler/ns3.40-gnb_scheduler-default"
        )
        return [
            binary,
            f"--simTime={self.max_steps}",
            f"--openGymPort={self.port}",
            f"--tickDir={self.report_dir}",
            f"--episode={self._episode % 300}",
            f"--lambdaArrival={self.lambda_arrival}",
        ]

    def _launch(self):
        self._teardown()
        import zmq as _zmq
        time.sleep(0.2)
        self._zmq_context = _zmq.Context()
        self._zmq_socket  = self._zmq_context.socket(_zmq.REP)
        self._zmq_socket.setsockopt(_zmq.LINGER, 0)
        self._zmq_socket.bind(f"tcp://*:{self.port}")
        print(f"[ns3_env] ZMQ REP bound on port {self.port}")
        cmd = self._build_ns3_cmd()
        log_dir  = os.path.join(os.path.dirname(self.report_dir), "ns3_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir,
            f"ns3_lam{int(self.lambda_arrival)}_ep{self._episode:04d}.log"
        )
        print(f"[ns3_env] Launching NS-3 ep={self._episode}: {' '.join(cmd)}")
        self._log_file = open(log_path, "w")
        self._proc = subprocess.Popen(
            cmd, cwd=self.ns3_scratch_dir,
            stdout=self._log_file, stderr=subprocess.STDOUT)
        self._connect()

    def _connect(self):
        import zmq as _zmq
        from ns3gym.ns3env import Ns3ZmqBridge
        import ns3gym.ns3env as _ne

        bridge = object.__new__(Ns3ZmqBridge)
        bridge.socket              = self._zmq_socket
        bridge.port                = self.port
        bridge.startSim            = False
        bridge.simSeed             = self.seed
        bridge.simArgs             = {}
        bridge.envStopped          = False
        bridge.simPid              = None
        bridge.wafPid              = None
        bridge.ns3Process          = None
        bridge.forceEnvStop        = False
        bridge.obsData             = None
        bridge.reward              = 0
        bridge.gameOver            = False
        bridge.gameOverReason      = None
        bridge.extraInfo           = None
        bridge.newStateRx          = False
        bridge._action_space       = None
        bridge._observation_space  = None

        print("[ns3_env] Waiting for NS-3 init …")
        bridge.initialize_env(1.0)
        bridge.rx_env_state()
        print("[ns3_env] Ready.")

        env = object.__new__(_ne.Ns3Env)
        env.ns3ZmqBridge          = bridge
        env.stepTime               = 1.0
        env.startSim               = False
        env.action_space           = bridge.get_action_space()
        env.observation_space      = bridge.get_observation_space()
        env.reward_range           = (-float("inf"), float("inf"))
        env.envDirty               = False
        env.viewer                 = None
        env.state                  = None
        env.steps_beyond_done      = None
        try:
            from gym.utils import seeding
            env.np_random, _ = seeding.np_random(self.seed)
        except Exception:
            env.np_random = None
        self._gym = env

    def _teardown(self):
        if self._gym is not None:
            try: self._gym.close()
            except Exception: pass
            self._gym = None
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=8)
            except Exception:
                try: self._proc.kill()
                except Exception: pass
            self._proc = None
        if self._log_file is not None:
            try: self._log_file.close()
            except Exception: pass
            self._log_file = None

    def _parse_obs(self, raw) -> np.ndarray:
        if raw is None:
            return np.zeros((N_CELLS, OBS_PER_CELL), np.float32)
        flat = np.asarray(raw, dtype=np.float32).flatten()
        if flat.size < OBS_DIM:
            flat = np.pad(flat, (0, OBS_DIM - flat.size))
        return flat[:OBS_DIM].reshape(N_CELLS, OBS_PER_CELL)

    def _parse_info(
        self, ns3_info: Any, obs: np.ndarray
    ) -> Tuple[List[dict], np.ndarray]:
        raw: dict = {}
        if isinstance(ns3_info, str) and ns3_info.strip():
            try:
                raw = json.loads(ns3_info)
            except json.JSONDecodeError:
                pass
        elif isinstance(ns3_info, dict):
            raw = ns3_info

        if not raw:
            cell_infos = []
            for c in range(N_CELLS):
                cell_infos.append(self._empty_cell_info())
            return cell_infos, self.get_event_mask()

        cell_infos = []
        for c in range(N_CELLS):
            cd = raw.get(f"c{c}", {})

            ho_acc = int(cd.get("ho_acc", 0))
            ho_rej = int(cd.get("ho_rej", 0))
            nc_acc = int(cd.get("nc_acc", 0))
            nc_rej = int(cd.get("nc_rej", 0))
            ho_t   = ho_acc + ho_rej
            nc_t   = nc_acc + nc_rej
            rt     = ho_t + nc_t

            ho_rate = float(cd.get("ho_rate",
                            ho_acc / ho_t if ho_t > 0 else 1.0))
            nc_rate = float(cd.get("nc_rate",
                            nc_acc / nc_t if nc_t > 0 else 1.0))

            util_served     = float(cd.get("util_served",  0.0))
            sf_covered      = float(cd.get("sf_covered",   0.0))
            is_lender       = bool(cd.get("is_lender",     False))
            is_borrower     = bool(cd.get("is_borrower",   False))
            ovl             = bool(cd.get("ovl",           False))
            donation_fracs  = [float(cd.get(f"don{s}", 0.0))
                               for s in range(N_SLICES)]

            teleop_ho_rate = float(cd.get("teleop_ho_rate", ho_rate))
            teleop_nc_rate = float(cd.get("teleop_nc_rate", nc_rate))
            auto_ho_rate   = float(cd.get("auto_ho_rate",   ho_rate))
            auto_nc_rate   = float(cd.get("auto_nc_rate",   nc_rate))

            prio_sat = (
                SLICE_ALPHA[1] * (P_H * teleop_ho_rate + P_N * teleop_nc_rate)
              + SLICE_ALPHA[0] * (P_H * auto_ho_rate   + P_N * auto_nc_rate)
            ) / (_ALPHA_SUM * (P_H + P_N))

            tot_donated  = sum(donation_fracs)
            borrow_teleop = BORROW_SHARE_TELEOP * tot_donated
            borrow_auto   = BORROW_SHARE_AUTO   * tot_donated

            ho_acc_ps = [int(cd.get(f"sl{s}_ho_acc", 0)) for s in range(N_SLICES)]
            ho_rej_ps = [int(cd.get(f"sl{s}_ho_rej", 0)) for s in range(N_SLICES)]
            nc_acc_ps = [int(cd.get(f"sl{s}_nc_acc", 0)) for s in range(N_SLICES)]
            nc_rej_ps = [int(cd.get(f"sl{s}_nc_rej", 0)) for s in range(N_SLICES)]

            lender_ho_blk    = float(cd.get("lender_ho_blk",   0.0))
            lender_nc_blk    = float(cd.get("lender_nc_blk",   0.0))
            ho_drop_borrower = float(cd.get("ho_drop_borrower",0.0))
            nc_blk_lender    = float(cd.get("nc_blk_lender",   0.0))

            at  = ho_acc + nc_acc
            acc = at / rt if rt > 0 else 1.0

            ci = {
                "ho_acc": ho_acc, "ho_rej": ho_rej,
                "nc_acc": nc_acc, "nc_rej": nc_rej,
                "ho_acc_ps": ho_acc_ps, "ho_rej_ps": ho_rej_ps,
                "nc_acc_ps": nc_acc_ps, "nc_rej_ps": nc_rej_ps,
                "ho_rate": ho_rate, "nc_rate": nc_rate,
                "accept_rate": acc,
                "util_served": util_served,
                "teleop_ho_rate": teleop_ho_rate,
                "teleop_nc_rate": teleop_nc_rate,
                "auto_ho_rate":   auto_ho_rate,
                "auto_nc_rate":   auto_nc_rate,
                "prio_sat": prio_sat,
                "sf_covered": sf_covered,
                "donation_fracs": donation_fracs,
                "tot_donated": tot_donated,
                "borrow_eff": float(np.mean(donation_fracs)),
                "borrow_teleop": borrow_teleop,
                "borrow_auto":   borrow_auto,
                "is_lender":   is_lender,
                "is_borrower": is_borrower,
                "_ovl": ovl,
                "_used_norm":  obs[c][_IDX_USED  : _IDX_USED  + N_SLICES].tolist(),
                "_alloc_obs":  obs[c][_IDX_ALLOC : _IDX_ALLOC + N_SLICES].tolist(),
                "lender_ho_blk":    lender_ho_blk,
                "lender_nc_blk":    lender_nc_blk,
                "ho_drop_borrower": ho_drop_borrower,
                "nc_blk_lender":    nc_blk_lender,
            }
            cell_infos.append(ci)

        return cell_infos, self.get_event_mask()

    def _empty_cell_info(self) -> dict:
        return {
            "ho_acc": 0, "ho_rej": 0, "nc_acc": 0, "nc_rej": 0,
            "ho_acc_ps": [0]*N_SLICES, "ho_rej_ps": [0]*N_SLICES,
            "nc_acc_ps": [0]*N_SLICES, "nc_rej_ps": [0]*N_SLICES,
            "ho_rate": 1.0, "nc_rate": 1.0, "accept_rate": 1.0,
            "util_served": 0.0,
            "teleop_ho_rate": 1.0, "teleop_nc_rate": 1.0,
            "auto_ho_rate":   1.0, "auto_nc_rate":   1.0,
            "prio_sat": 1.0, "sf_covered": 0.0,
            "donation_fracs": [0.0]*N_SLICES,
            "tot_donated": 0.0, "borrow_eff": 0.0,
            "borrow_teleop": 0.0, "borrow_auto": 0.0,
            "is_lender": False, "is_borrower": False, "_ovl": False,
            "_used_norm":  [0.0]*N_SLICES,
            "_alloc_obs":  CAPS.tolist(),
            "lender_ho_blk": 0.0, "lender_nc_blk": 0.0,
            "ho_drop_borrower": 0.0, "nc_blk_lender": 0.0,
        }

    def _build_info(self, cell_infos: List[dict],
                    obs: np.ndarray) -> dict:
        ho_acc_t = sum(ci["ho_acc"] for ci in cell_infos)
        ho_rej_t = sum(ci["ho_rej"] for ci in cell_infos)
        nc_acc_t = sum(ci["nc_acc"] for ci in cell_infos)
        nc_rej_t = sum(ci["nc_rej"] for ci in cell_infos)
        total_t  = ho_acc_t + ho_rej_t + nc_acc_t + nc_rej_t

        util_w = float(np.mean([
            sum(ci["_used_norm"][s] * ci["_alloc_obs"][s]
                for s in range(N_SLICES))
            for ci in cell_infos
        ]))

        return {
            "cell_infos":    cell_infos,
            "util_served":   util_w,
            "prio_sat":      float(np.mean([ci["prio_sat"]  for ci in cell_infos])),
            "sf_covered":    float(np.mean([ci["sf_covered"] for ci in cell_infos])),
            "borrow_eff":    float(np.mean([ci["borrow_eff"] for ci in cell_infos])),
            "teleop_ho_rate":float(np.mean([ci["teleop_ho_rate"] for ci in cell_infos])),
            "teleop_nc_rate":float(np.mean([ci["teleop_nc_rate"] for ci in cell_infos])),
            "auto_ho_rate":  float(np.mean([ci["auto_ho_rate"]   for ci in cell_infos])),
            "auto_nc_rate":  float(np.mean([ci["auto_nc_rate"]   for ci in cell_infos])),
            "borrow_teleop": float(np.mean([ci["borrow_teleop"]  for ci in cell_infos])),
            "borrow_auto":   float(np.mean([ci["borrow_auto"]    for ci in cell_infos])),
            "ho_acc": ho_acc_t, "ho_rej": ho_rej_t,
            "nc_acc": nc_acc_t, "nc_rej": nc_rej_t,
            "accept_rate": float((ho_acc_t + nc_acc_t) / max(1, total_t)),
            "ho_rate": float(ho_acc_t / max(1, ho_acc_t + ho_rej_t)),
            "nc_rate": float(nc_acc_t / max(1, nc_acc_t + nc_rej_t)),
            # aggregated lender metrics
            "lender_nc_blk": float(np.mean(
                [ci["lender_nc_blk"] for ci in cell_infos
                 if ci["is_lender"]] or [0.0])),
            "lender_ho_blk": float(np.mean(
                [ci["lender_ho_blk"] for ci in cell_infos
                 if ci["is_lender"]] or [0.0])),
        }


def _safe(d, key, default=0.0):
    return d.get(key, default)