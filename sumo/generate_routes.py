#!/usr/bin/env python3
"""
generate_routes.py  — v45 (ALL-4-SLICES-PER-CELL FIX)
──────────────────────────────────────────────────────────────────────────
Generates a SUMO .rou.xml file whose per-step vehicle arrival counts
exactly reproduce the Poisson arrival model used in sumo_bridge.py v45.

ROOT CAUSE FIX (v44 → v45)
────────────────────────────
Previously, SLICE_TO_EDGE mapped each slice to a single road segment,
implying one slice per cell.  The correct model is:

    Every cell (seg_A, seg_B, seg_C) carries ALL 4 slices:
        Slice 0 — 25%  Tele-operated driving  (high priority)
        Slice 1 — 40%  Autonomous driving      (highest priority)
        Slice 2 — 25%  VoIP                    (medium priority)
        Slice 3 — 10%  Best effort             (low priority)

Routes are now generated per (cell, slice) combination (3 cells × 4 slices).
Each vehicle carries a "cell_id" param so the bridge knows which cell's
arrival it belongs to.

Run:
    python generate_routes.py --lambda_arrival 55 --n_steps 200 --seed 42
"""

import argparse
import os
import numpy as np
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

# ── Match PPO/bridge constants exactly ──────────────────────────────────────
FLOW_TYPES = {
    #              bw      mean_life  is_ho   vtype
    "HO_video"      : (0.0090, 12, True,  "vt_ho_video"),   # was 0.0180
    "HO_data"       : (0.0060,  8, True,  "vt_ho_data"),    # was 0.0120
    "NC_video"      : (0.0075, 10, False, "vt_nc_video"),   # was 0.0150
    "NC_voip"       : (0.0030, 15, False, "vt_nc_voip"),    # was 0.0060
    "NC_best_effort": (0.0020,  6, False, "vt_nc_be"),      # was 0.0040
}

# All 4 slices present in every cell
SLICE_FLOWS = {
    0: ["HO_video",  "NC_video"],
    1: ["HO_video",  "HO_data", "NC_video", "NC_voip"],
    2: ["NC_voip",   "NC_best_effort"],
    3: ["NC_best_effort"],
}

SLICE_CAPS = [0.25, 0.40, 0.25, 0.10]

# Road segment per cell (used for SUMO routing only)
CELL_TO_EDGE = {0: "seg_A", 1: "seg_B", 2: "seg_C"}

N_CELLS         = 3
N_SLICES        = 4
BASE_ARRIVAL    = 0.68
NC_ARRIVAL      = 0.63
MAX_HO_PER_STEP = 5    # per-cell cap
MAX_NC_PER_STEP = 6    # per-cell cap

STEP_DURATION = 1.0
VEH_LENGTH    = 7.5


def _arrival_rate(slice_idx, ftype, load, max_load, lf):
    bw, mean_life, is_ho, vtype = FLOW_TYPES[ftype]
    base = BASE_ARRIVAL if is_ho else NC_ARRIVAL
    return float(load[slice_idx]) * max_load * base * lf


def generate_routes(lambda_arrival: float,
                    n_steps: int,
                    seed: int,
                    max_load: float,
                    out_path: str):
    """
    Generate .rou.xml matching sumo_bridge v45:
      — For each tick, for each cell, generate arrivals for all 4 slices.
      — Each cell has an independent RNG (seed + cell * 1337).
      — Per-cell HO/NC caps applied after generation.
    """
    lf      = lambda_arrival / 20.0          # matches CellRNG.lf
    lam_off = (lambda_arrival - 45.0) / 5.0

    root = Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/routes_file.xsd")

    veh_id = 0

    # One RNG and load vector per cell (mirrors CellRNG)
    cell_rngs = [np.random.default_rng(seed + c * 1337) for c in range(N_CELLS)]
    cell_loads = []
    for c in range(N_CELLS):
        rng    = cell_rngs[c]
        lo     = 0.88 + 0.02 * lam_off
        hi     = min(max_load, lo + 0.30 * lf)
        hi     = max(lo + 0.04, hi)
        hi_a   = float(rng.uniform(lo + 0.02, hi))
        load   = rng.uniform(lo, hi_a, N_SLICES).astype(np.float32)
        load[1]= float(rng.uniform(max(lo, hi_a * 0.96),
                                    min(max_load, hi_a * 1.02)))
        for i in [0, 1]:
            load[i] = max(float(load[i]), 0.92)
        cell_loads.append(load)

    for step in range(n_steps):
        depart_time = step * STEP_DURATION

        for c in range(N_CELLS):
            rng  = cell_rngs[c]
            load = cell_loads[c]
            edge = CELL_TO_EDGE[c]

            ho_arrivals = []
            nc_arrivals = []

            # Generate arrivals for all 4 slices in this cell
            for s in range(N_SLICES):
                for ftype in SLICE_FLOWS[s]:
                    bw, mean_life, is_ho, vtype = FLOW_TYPES[ftype]
                    lam = _arrival_rate(s, ftype, load, max_load, lf)
                    n   = int(rng.poisson(lam))
                    for _ in range(n):
                        life = max(1, int(rng.geometric(1.0 / mean_life)))
                        rec  = dict(
                            vtype=vtype, slice_idx=s,
                            cell_id=c, edge=edge,
                            life=life, bw=bw,
                            is_ho=is_ho, ftype=ftype,
                        )
                        if is_ho:
                            ho_arrivals.append(rec)
                        else:
                            nc_arrivals.append(rec)

            # Apply per-cell caps (matches CellRNG.gen_arrivals_for_cell)
            if len(ho_arrivals) > MAX_HO_PER_STEP:
                idx = rng.choice(len(ho_arrivals), MAX_HO_PER_STEP, replace=False)
                ho_arrivals = [ho_arrivals[i] for i in idx]
            if len(nc_arrivals) > MAX_NC_PER_STEP:
                idx = rng.choice(len(nc_arrivals), MAX_NC_PER_STEP, replace=False)
                nc_arrivals = [nc_arrivals[i] for i in idx]

            for rec in ho_arrivals + nc_arrivals:
                v = SubElement(root, "vehicle")
                v.set("id",           f"veh_{veh_id}")
                v.set("type",         rec["vtype"])
                v.set("depart",       f"{depart_time:.1f}")
                v.set("departLane",   "best")
                v.set("departSpeed",  "max")

                # Encode PPO/bridge metadata as SUMO params
                def add_param(key, value):
                    p = SubElement(v, "param")
                    p.set("key",   key)
                    p.set("value", str(value))

                add_param("cell_id",   rec["cell_id"])
                add_param("slice_idx", rec["slice_idx"])
                add_param("bw_demand", rec["bw"])
                add_param("is_ho",     "1" if rec["is_ho"] else "0")
                add_param("mean_life", rec["life"])
                add_param("flow_type", rec["ftype"])

                route = SubElement(v, "route")
                if edge == "seg_A":
                    route.set("edges", "seg_A seg_B seg_C")
                elif edge == "seg_B":
                    route.set("edges", "seg_B seg_C")
                else:
                    route.set("edges", "seg_C")

                veh_id += 1

            # Load drift per cell (mirrors CellRNG.drift)
            delta = rng.normal(0.0, 0.018, N_SLICES).astype(np.float32)
            spike = ((rng.random(N_SLICES) < 0.04) *
                     rng.uniform(0.04, 0.12, N_SLICES))
            cell_loads[c] = np.clip(load + delta + spike,
                                    0.40, max_load).astype(np.float32)
            for i in [0, 1]:
                cell_loads[c][i] = max(float(cell_loads[c][i]), 0.88)

    try:
        indent(root, space="  ")
    except AttributeError:
        pass  # Python < 3.9

    tree = ElementTree(root)
    tree.write(out_path, encoding="unicode", xml_declaration=True)
    print(f"[generate_routes] Wrote {veh_id} vehicles over {n_steps} steps "
          f"({N_CELLS} cells × {N_SLICES} slices each) → {out_path}")
    return veh_id


def main():
    ap = argparse.ArgumentParser(
        description="Generate SUMO routes matching sumo_bridge v45 (all 4 slices per cell)"
    )
    ap.add_argument("--lambda_arrival", type=float, default=55.0)
    ap.add_argument("--n_steps",        type=int,   default=200)
    ap.add_argument("--seed",           type=int,   default=42)
    ap.add_argument("--max_load",       type=float, default=1.25)
    ap.add_argument("--out_dir",        type=str,   default=".")
    args = ap.parse_args()

    out_path = os.path.join(
        args.out_dir, f"routes_lam{int(args.lambda_arrival)}.rou.xml")
    generate_routes(args.lambda_arrival, args.n_steps,
                    args.seed, args.max_load, out_path)


if __name__ == "__main__":
    main()