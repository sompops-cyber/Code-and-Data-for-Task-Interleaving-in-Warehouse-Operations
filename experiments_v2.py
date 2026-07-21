# Copyright (c) 2026 Sompop Saengphueng
# Replication Code for: "Task Interleaving in Warehouse Operations"
# Licensed under the MIT License. See LICENSE file for full details.
"""Experimental grid v2 -- at CALIBRATED January demand (740/831) and tau* = 1.5."""
import json, statistics, sys
sys.path.insert(0, "/home/claude/paper")
from sim_interleaving import Params, run_once, ci95

POLICIES = ["SINGLE", "INTERLEAVE", "NEAREST"]
REPS = 30
BASE = dict(putaway_pallets_per_day=740, pick_pallets_per_day=831, pick_hold_min=1.5)

def experiment(p):
    out = {}
    for pol in POLICIES:
        runs = [run_once(pol, p, p.seed + r) for r in range(REPS)]
        out[pol] = {m: ci95([r[m] for r in runs]) for m in runs[0]}
    return out

res = {}
res["fleet"] = {}
for fleet in range(6, 13):
    res["fleet"][fleet] = experiment(Params(n_reach_trucks=fleet, **BASE))
    print("fleet", fleet, "ok")
res["hold"] = {}
for tau in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]:
    b = dict(BASE); b["pick_hold_min"] = tau
    res["hold"][tau] = experiment(Params(n_reach_trucks=9, **b))
    print("hold", tau, "ok")
res["demand"] = {}
for s in [0.8, 1.0, 1.2]:
    b = dict(BASE); b["putaway_pallets_per_day"] = 740*s; b["pick_pallets_per_day"] = 831*s
    res["demand"][s] = experiment(Params(n_reach_trucks=9, **b))
    print("demand", s, "ok")
json.dump(res, open("exp_results_v2.json","w"), indent=1)
print("saved")
