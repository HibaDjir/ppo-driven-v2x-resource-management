import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

N_CELLS   = 3
N_SLICES  = 4
CAPS      = [0.23, 0.38, 0.13, 0.26]

SPEED_MS       = 11.11
CELL_LENGTH_M  = 1500.0      
DT_S           = 1.0

NC_FLOWS_PER_SLICE: Dict[int, List[Tuple[str, float]]] = {
    0: [("AUT_video_s0", 0.70)],
    1: [("AUT_video_s1", 0.40), ("AUT_voip_s1", 0.35)],
    2: [("AUT_voip_s2", 0.85), ("INFO_be_s2", 0.75)],
    3: [("NC_data_s3", 0.60), ("INFO_be_s3", 0.50)],
}

FLOW_DEFS = {
    "TOD_video_s0": dict(bw=0.015, mean_life=30, slice_idx=0, is_ho=True),
    "AUT_video_s0": dict(bw=0.015, mean_life=30, slice_idx=0, is_ho=False),
    "TOD_video_s1": dict(bw=0.020, mean_life=30, slice_idx=1, is_ho=True),
    "TOD_data_s1":  dict(bw=0.020, mean_life=28, slice_idx=1, is_ho=True),
    "AUT_video_s1": dict(bw=0.020, mean_life=30, slice_idx=1, is_ho=False),
    "AUT_voip_s1":  dict(bw=0.015, mean_life=30, slice_idx=1, is_ho=False),
    "AUT_voip_s2":  dict(bw=0.007, mean_life=25, slice_idx=2, is_ho=False),
    "INFO_be_s2":   dict(bw=0.007, mean_life=25, slice_idx=2, is_ho=False),
    "NC_data_s3":   dict(bw=0.010, mean_life=45, slice_idx=3, is_ho=False),
    "INFO_be_s3":   dict(bw=0.010, mean_life=45, slice_idx=3, is_ho=False),
}
HO_FLOW_PER_SLICE: Dict[int, str] = {
    0: "TOD_video_s0",
    1: "TOD_video_s1",
    2: "AUT_voip_s2",
    3: "INFO_be_s3",
}

LAMBDA_LEVELS = [45, 50, 55, 60, 65]


@dataclass
class Flow:
    flow_type: str
    bw: float
    slice_idx: int
    is_ho: bool
    remaining: int


@dataclass
class Vehicle:
    veh_id: str
    cell: int
    pos_in_cell: float
    flows: List[Flow]
    active: bool = True

    def to_arrival(self, f: Flow, is_ho: bool) -> dict:
        return {
            "veh_id":    self.veh_id,
            "flow_type": f.flow_type,
            "bw":        f.bw,
            "is_ho":     is_ho,
            "slice_idx": f.slice_idx,
            "remaining": f.remaining,
        }


class VehicleMobilitySimulator:
    def __init__(self, lambda_arrival: float, max_load: float,
                 seed: int, n_steps: int):
        self.lam      = lambda_arrival
        self.lam_s    = lambda_arrival / 60.0
        self.max_load = max_load
        self.n_steps  = n_steps
        self.rng      = np.random.default_rng(seed)
        self._vehicles: Dict[str, Vehicle] = {}
        self._veh_counter = 0
        self._load_acc   = np.zeros((N_CELLS, N_SLICES), dtype=np.float64)
        self._load_ticks = 0
        self.report_dir = None

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._load_acc[:] = 0.0
        self._load_ticks  = 0

    def _spawn_stable_vehicle(self, tick: int) -> Vehicle:
        vid = f"v{self._veh_counter:06d}_t{tick}"
        self._veh_counter += 1
        flows = []
        for s in range(N_SLICES):
            for ftype, prob in NC_FLOWS_PER_SLICE[s]:
                if self.rng.random() < prob:
                    fd = FLOW_DEFS[ftype]
                    life = max(50, int(fd["mean_life"] * 1.3))
                    flows.append(Flow(
                        flow_type=ftype,
                        bw=fd["bw"],
                        slice_idx=fd["slice_idx"],
                        is_ho=False,
                        remaining=life,
                    ))
        for s in [0, 1]:
            if not any(f.slice_idx == s for f in flows):
                ftype, _ = NC_FLOWS_PER_SLICE[s][0]
                fd = FLOW_DEFS[ftype]
                flows.append(Flow(ftype, fd["bw"], s, False,
                                  max(50, int(fd["mean_life"] * 1.3))))
        if not flows:
            fd = FLOW_DEFS["NC_data_s3"]
            flows.append(Flow("NC_data_s3", fd["bw"], 3, False, 50))
        veh = Vehicle(
            veh_id=vid,
            cell=self._veh_counter % N_CELLS,
            pos_in_cell=float(self.rng.uniform(0, CELL_LENGTH_M)),
            flows=flows,
        )
        self._vehicles[vid] = veh
        return veh

    def step(self, tick: int, episode: int) -> dict:
        cell_arrivals:  List[List[dict]] = [[] for _ in range(N_CELLS)]
        cell_ho_export: List[List[dict]] = [[] for _ in range(N_CELLS)]

        n_new = int(self.rng.poisson(self.lam_s * DT_S))
        for _ in range(n_new):
            veh = self._spawn_vehicle(tick)
            for f in veh.flows:
                cell_arrivals[veh.cell].append(veh.to_arrival(f, is_ho=False))

        to_remove = []
        for vid, veh in self._vehicles.items():
            if not veh.active:
                to_remove.append(vid)
                continue

            alive = []
            for f in veh.flows:
                f.remaining -= 1
                if f.remaining > 0:
                    alive.append(f)
            veh.flows = alive
            if not veh.flows:
                veh.active = False
                to_remove.append(vid)
                continue

            veh.pos_in_cell += SPEED_MS * DT_S
            if veh.pos_in_cell >= CELL_LENGTH_M:
                veh.pos_in_cell -= CELL_LENGTH_M
                new_cell = veh.cell + 1
                if new_cell >= N_CELLS:
                    veh.active = False
                    to_remove.append(vid)
                    continue
                for f in veh.flows:
                    ho_rec = self._make_ho_record(veh, f)
                    cell_arrivals[new_cell].append(ho_rec)
                    cell_ho_export[new_cell].append(ho_rec)
                veh.cell = new_cell

        for vid in to_remove:
            self._vehicles.pop(vid, None)

        active_vehicles = sum(1 for v in self._vehicles.values() if v.active)

        # FIX: was max(70, int(lam_s * 85))
        # At lam=45: max(70, 63) = 70
        # At lam=50: max(70, 70) = 70  ← SAME as lam=45, causing util to
        #   oscillate and dip below lam=45 levels when Poisson bursts end.
        # New formula max(65, int(lam_s * 95)) gives strictly increasing
        # floors: 71 → 79 → 87 → 95 → 102 across lam=45..65, so each
        # lambda level always has more stable background vehicles than the
        # one below it and util rises monotonically with lambda.
        min_vehicles = max(65, int(self.lam_s * 95))

        while active_vehicles < min_vehicles:
            veh = self._spawn_stable_vehicle(tick)
            for f in veh.flows:
                cell_arrivals[veh.cell].append(veh.to_arrival(f, is_ho=False))
            active_vehicles += 1

        cell_load = self._compute_cell_load()
        self._load_acc   += cell_load
        self._load_ticks += 1

        cells_data = {}
        for c in range(N_CELLS):
            cells_data[str(c)] = {
                "arrivals":  cell_arrivals[c],
                "ho_export": cell_ho_export[c],
                "cell_load": cell_load[c].tolist(),
            }

        avg_load = cell_load.mean(axis=0).tolist()
        result = {
            "tick":    tick,
            "episode": episode,
            "cells":   cells_data,
            "load":    avg_load,
        }

        if tick >= 0 and episode >= 0:
            fname = f"ep{episode:04d}_tick{tick:04d}.json"
            with open(os.path.join(self.report_dir, fname), "w") as fh:
                json.dump(result, fh)

        return result

    def _spawn_vehicle(self, tick: int) -> Vehicle:
        vid = f"v{self._veh_counter:06d}_t{tick}"
        self._veh_counter += 1
        flows = self._sample_flows(vid)
        veh = Vehicle(
            veh_id=vid,
            cell=int(self.rng.integers(0, N_CELLS)),
            pos_in_cell=float(self.rng.uniform(0, CELL_LENGTH_M)),
            flows=flows,
        )
        self._vehicles[vid] = veh
        return veh

    def _sample_flows(self, vid: str) -> List[Flow]:
        flows = []
        for s in range(N_SLICES):
            for ftype, prob in NC_FLOWS_PER_SLICE[s]:
                if self.rng.random() < prob:
                    fd = FLOW_DEFS[ftype]
                    life = int(self.rng.uniform(
                        fd["mean_life"] * 0.7,
                        fd["mean_life"] * 1.3
                    ))
                    life = max(5, min(life, 60))
                    flows.append(Flow(
                        flow_type=ftype,
                        bw=fd["bw"],
                        slice_idx=fd["slice_idx"],
                        is_ho=False,
                        remaining=life,
                    ))
        for s in [0, 1]:
            has_slice = any(f.slice_idx == s for f in flows)
            if not has_slice:
                ftype, _ = NC_FLOWS_PER_SLICE[s][0]
                fd = FLOW_DEFS[ftype]
                life = int(self.rng.uniform(fd["mean_life"] * 0.7, fd["mean_life"] * 1.3))
                life = max(5, min(life, 60))
                flows.append(Flow(ftype, fd["bw"], s, False, life))
        if not flows:
            fd = FLOW_DEFS["NC_data_s3"]
            life = int(self.rng.uniform(fd["mean_life"] * 0.7, fd["mean_life"] * 1.3))
            life = max(5, min(life, 60))
            flows.append(Flow("NC_data_s3", fd["bw"], 3, False, life))
        return flows

    def _make_ho_record(self, veh: Vehicle, f: Flow) -> dict:
        ho_type = HO_FLOW_PER_SLICE.get(f.slice_idx, "TOD_video_s1")
        fd = FLOW_DEFS[ho_type]
        life = int(self.rng.uniform(fd["mean_life"] * 0.7, fd["mean_life"] * 1.3))
        life = max(5, min(life, 60))
        return {
            "veh_id":    veh.veh_id,
            "flow_type": ho_type,
            "bw":        fd["bw"],
            "is_ho":     True,
            "slice_idx": f.slice_idx,
            "remaining": life,
        }

    def _compute_cell_load(self) -> np.ndarray:
        bw_used = np.zeros((N_CELLS, N_SLICES), dtype=np.float64)
        for veh in self._vehicles.values():
            if not veh.active:
                continue
            for f in veh.flows:
                if veh.cell < N_CELLS:
                    bw_used[veh.cell, f.slice_idx] += f.bw
        caps = np.array(CAPS, dtype=np.float64)
        load = np.clip(bw_used / (caps + 1e-9), 0.0, 1.0)
        return load.astype(np.float32)


class SumoBridge:
    def __init__(self, lambda_arrival: float, n_steps: int, n_episodes: int,
                 seed: int, max_load: float, report_dir: str):
        self.lam        = lambda_arrival
        self.n_steps    = n_steps
        self.n_episodes = n_episodes
        self.seed       = seed
        self.max_load   = max_load
        self.base_dir   = report_dir
        self.report_dir = os.path.join(report_dir, f"lam{int(lambda_arrival)}")
        os.makedirs(self.report_dir, exist_ok=True)
        self.sim = VehicleMobilitySimulator(
            lambda_arrival=lambda_arrival,
            max_load=max_load,
            seed=seed,
            n_steps=n_steps,
        )
        self.sim.report_dir = self.report_dir

    def run_offline(self):
        print(f"[sumo_bridge] OFFLINE  lambda={self.lam}  episodes={300}"
              f"  steps={self.n_steps}  -> {self.report_dir}")
        total_files = 0

        self.sim.reset(seed=self.seed)
        print(f"  Warming up for 600 ticks to reach steady state ...")
        for t in range(600):
            self.sim.step(tick=-1, episode=-1)
        print(f"  Warmup complete. Current vehicles: {len(self.sim._vehicles)}")

        for ep in range(self.n_episodes):
            self.sim._load_acc[:] = 0.0
            self.sim._load_ticks  = 0

            ho_total = nc_total = 0
            last_report = None
            for tick in range(self.n_steps):
                report = self.sim.step(tick, ep)
                last_report = report
                for c in range(N_CELLS):
                    for a in report["cells"][str(c)]["arrivals"]:
                        if a["is_ho"]:
                            ho_total += 1
                        else:
                            nc_total += 1
                total_files += 1
            if ep % 50 == 0 and last_report is not None:
                avg_load = last_report["load"]
                util_bw  = sum(avg_load[s] * CAPS[s] for s in range(N_SLICES))
                util_pct = util_bw / sum(CAPS) * 100
                n_veh = len(self.sim._vehicles)
                print(f"  ep={ep:4d}  HO={ho_total:5d}  NC={nc_total:5d}"
                      f"  HO/NC={ho_total/(nc_total+1):.2f}"
                      f"  active_veh={n_veh}"
                      f"  util={util_pct:.1f}%")
        print(f"[sumo_bridge] Done. {total_files} files -> {self.report_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda_arrival", type=float, default=None)
    ap.add_argument("--n_steps",        type=int,   default=200)
    ap.add_argument("--n_episodes",     type=int,   default=300)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--max_load",       type=float, default=None)
    ap.add_argument("--report_dir",     type=str,   default="bridge/tick_reports_live")
    args = ap.parse_args()

    target_lams = (
        [int(args.lambda_arrival)]
        if args.lambda_arrival is not None
        else LAMBDA_LEVELS
    )
    for lam in target_lams:
        max_load = args.max_load or min(1.80, 1.60 + 0.008 * (lam - 45))
        bridge = SumoBridge(
            lambda_arrival=float(lam),
            n_steps=args.n_steps,
            n_episodes=args.n_episodes,
            seed=args.seed,
            max_load=max_load,
            report_dir=args.report_dir,
        )
        bridge.run_offline()


if __name__ == "__main__":
    main()


