# Copyright (c) 2026 Sompop Saengphueng
# Replication Code for: "Task Interleaving in Warehouse Operations"
# Licensed under the MIT License. See LICENSE file for full details.
"""
Discrete-event simulation of warehouse task interleaving (dual-command) policies.
=================================================================================
Skeleton model parameterized with the empirical work-measurement data from:
"Assessment of Task Interleaving to Enhance Material Handling Equipment
Efficiency in Warehouse Operations" (case warehouse, 1,134 storage locations,
Reach Trucks, 8-hour shifts, 2025 throughput data).

Three dispatching policies are compared:
  1. SINGLE      - single-command cycles (current operation, baseline)
  2. INTERLEAVE  - opportunistic interleaving: after a put-away, take the
                   first waiting pick (FCFS) on the return trip if one exists
  3. NEAREST     - interleaving with nearest-compatible-task selection:
                   take the waiting pick whose location minimizes travel-between

Outputs per policy (mean and 95% CI over replications):
  - pallets handled per day (put-away + picking)
  - empty-travel share of total RT travel time
  - RT utilization
  - mean task waiting time
  - mean pick-order cycle time

Author notes / TODOs for the full study are marked with  # TODO(research):
"""

import random
import statistics
from dataclasses import dataclass, field

import simpy

# ----------------------------------------------------------------------------
# 1. PARAMETERS  (all times in minutes, distances in meters)
# ----------------------------------------------------------------------------

@dataclass
class Params:
    # --- shift & demand (Table 14, January 2025 as default scenario) ---
    shift_minutes: float = 480.0          # 8-hour shift
    putaway_pallets_per_day: float = 307  # avg daily put-away throughput
    pick_pallets_per_day: float = 366     # avg daily picking throughput
    pallets_per_cycle: int = 2            # RTs handle 2 pallets per cycle (Tables 1-2)

    # --- fleet ---
    n_reach_trucks: int = 9               # baseline fleet (Table 15 average)

    # --- travel model (Section 2.2.2) ---
    # RT speed calibrated from the paper: 48.1 m in 0.48 min  ->  ~99.2 m/min.
    rt_speed_m_per_min: float = 99.2
    # Storage-location distances are sampled so that the mean one-way travel
    # time reproduces the paper's 1.06 min average over 1,134 locations
    # (mean distance ~105 m).  TODO(research): replace with the actual
    # 1,134-location distance table used in Section 2.2.2.
    dist_min_m: float = 20.0
    dist_max_m: float = 190.0
    cross_aisle_penalty_m: float = 10.0   # extra distance for travel-between

    # --- element-time distributions (min) ---
    # Triangular(min, mode, max) fitted to the 10-cycle observations of
    # Table 4 (put-away) and the select times of Table 9 (picking).
    # TODO(research): refit with full observation sheets (e.g., lognormal via MLE,
    # Kolmogorov-Smirnov goodness of fit) for the journal version.
    putaway_fixed: tuple = (               # non-travel elements, one full cycle (2 pallets)
        (0.12, 0.16, 0.20),   # pick up two pallets
        (0.14, 0.17, 0.21),   # RF scan product + location
        (0.17, 0.18, 0.19),   # place pallets on floor
        (0.14, 0.20, 0.27),   # lift pallet 1
        (0.11, 0.15, 0.20),   # store pallet 1 on rack
        (0.14, 0.19, 0.30),   # lower forks to travel position
        (0.17, 0.29, 0.33),   # lift pallet 2
        (0.18, 0.24, 0.27),   # store pallet 2 on rack
    )
    picking_fixed: tuple = (               # non-travel elements, one full cycle (2 pallets)
        (0.15, 0.21, 0.27),   # position forks at rack 1
        (0.10, 0.14, 0.18),   # retrieve pallet 1
        (0.15, 0.21, 0.27),   # lower forks
        (0.10, 0.14, 0.18),   # RF scan 1
        (0.10, 0.14, 0.18),   # place pallet 1 at drop-off
        (0.15, 0.21, 0.27),   # position forks at rack 2
        (0.10, 0.14, 0.18),   # retrieve pallet 2
        (0.15, 0.21, 0.27),   # lower forks
        (0.10, 0.14, 0.18),   # RF scan 2
        (0.11, 0.15, 0.19),   # place pallet 2 at drop-off
    )

    # --- WMS task-holding rule (what creates interleaving opportunities) ---
    # Under the interleaving policies, an *idle* truck may only take a pick that
    # has already waited at least `pick_hold_min` minutes; returning trucks may
    # take any waiting pick immediately.  This reproduces the task-holding logic
    # of commercial WMS interleaving.  TODO(research): calibrate/hold-time
    # sensitivity analysis (0-10 min) -- it trades pick response time against
    # empty-travel reduction.
    pick_hold_min: float = 2.0

    # --- experiment design ---
    warmup_minutes: float = 60.0
    n_replications: int = 20
    seed: int = 42


# ----------------------------------------------------------------------------
# 2. MODEL
# ----------------------------------------------------------------------------

@dataclass
class Task:
    kind: str           # "putaway" | "pick"
    location_m: float   # one-way distance from staging (m)
    created: float      # sim time of arrival
    started: float = None
    finished: float = None


@dataclass
class Stats:
    pallets_done: int = 0
    travel_loaded: float = 0.0
    travel_empty: float = 0.0
    handling: float = 0.0
    busy: float = 0.0
    waits: list = field(default_factory=list)
    pick_cycle_times: list = field(default_factory=list)
    putaway_cycles: int = 0
    interleaved_cycles: int = 0


class WarehouseSim:
    def __init__(self, env: simpy.Environment, p: Params, policy: str, rng: random.Random):
        self.env, self.p, self.policy, self.rng = env, p, policy, rng
        self.putaway_q: list[Task] = []
        self.pick_q: list[Task] = []
        self.task_event = env.event()       # signals arrival of any task
        self.stats = Stats()
        env.process(self.arrivals("putaway", p.putaway_pallets_per_day))
        env.process(self.arrivals("pick", p.pick_pallets_per_day))
        for _ in range(p.n_reach_trucks):
            env.process(self.reach_truck())

    # ---- helpers -----------------------------------------------------------
    def tri(self, spec):
        lo, mode, hi = spec
        return self.rng.triangular(lo, hi, mode)

    def draw_location(self) -> float:
        return self.rng.uniform(self.p.dist_min_m, self.p.dist_max_m)

    def t_travel(self, meters: float) -> float:
        return meters / self.p.rt_speed_m_per_min

    def signal(self):
        if not self.task_event.triggered:
            self.task_event.succeed()
        self.task_event = self.env.event()

    # ---- arrival processes -------------------------------------------------
    def arrivals(self, kind: str, pallets_per_day: float):
        """Poisson arrivals of *cycles* (2 pallets each).
        TODO(research): replace stationary Poisson with the empirical intra-day
        arrival profile (e.g., inbound waves in the morning) from WMS logs."""
        rate = (pallets_per_day / self.p.pallets_per_cycle) / self.p.shift_minutes
        while True:
            yield self.env.timeout(self.rng.expovariate(rate))
            task = Task(kind, self.draw_location(), self.env.now)
            (self.putaway_q if kind == "putaway" else self.pick_q).append(task)
            self.signal()

    # ---- task selection per policy ----------------------------------------
    def next_single(self):
        """Task selection for an idle truck.
        SINGLE: FCFS across both queues.
        INTERLEAVE/NEAREST: put-away first; picks only after `pick_hold_min`
        (younger picks are reserved for trucks returning from put-aways)."""
        if self.policy == "SINGLE":
            pools = [q for q in (self.putaway_q, self.pick_q) if q]
            if not pools:
                return None
            q = min(pools, key=lambda q: q[0].created)
            return q.pop(0)
        if self.putaway_q:
            return self.putaway_q.pop(0)
        if self.pick_q and (self.env.now - self.pick_q[0].created) >= self.p.pick_hold_min:
            return self.pick_q.pop(0)
        return None

    def compatible_pick(self, from_location: float):
        """Pick to append after a put-away, according to the policy."""
        if not self.pick_q:
            return None
        if self.policy == "INTERLEAVE":                     # FCFS pick
            return self.pick_q.pop(0)
        if self.policy == "NEAREST":                        # min travel-between
            best = min(self.pick_q, key=lambda t: abs(t.location_m - from_location))
            self.pick_q.remove(best)
            return best
        return None

    # ---- reach-truck process ------------------------------------------------
    def reach_truck(self):
        p = self.p
        while True:
            s = self.stats            # re-bind each cycle (stats object is reset after warm-up)
            task = self.next_single()
            if task is None:
                yield self.task_event | self.env.timeout(0.5)   # idle; recheck held picks
                continue
            task.started = self.env.now
            s.waits.append(task.started - task.created)
            t0 = self.env.now

            if task.kind == "putaway":
                # loaded travel out, store 2 pallets
                out = self.t_travel(task.location_m)
                yield self.env.timeout(out); s.travel_loaded += out
                h = sum(self.tri(e) for e in p.putaway_fixed)
                yield self.env.timeout(h); s.handling += h
                s.pallets_done += p.pallets_per_cycle
                s.putaway_cycles += 1

                nxt = None
                if self.policy in ("INTERLEAVE", "NEAREST"):
                    nxt = self.compatible_pick(task.location_m)

                if nxt is not None:                          # dual-command cycle
                    nxt.started = self.env.now
                    s.waits.append(nxt.started - nxt.created)
                    between_m = abs(nxt.location_m - task.location_m) + p.cross_aisle_penalty_m
                    tb = self.t_travel(between_m)
                    yield self.env.timeout(tb); s.travel_empty += tb   # travel-between (empty)
                    h = sum(self.tri(e) for e in p.picking_fixed)
                    yield self.env.timeout(h); s.handling += h
                    back = self.t_travel(nxt.location_m)
                    yield self.env.timeout(back); s.travel_loaded += back
                    s.pallets_done += p.pallets_per_cycle
                    nxt.finished = self.env.now
                    s.pick_cycle_times.append(nxt.finished - nxt.created)
                    s.interleaved_cycles += 1
                else:                                        # empty return
                    back = self.t_travel(task.location_m)
                    yield self.env.timeout(back); s.travel_empty += back

            else:  # pick task executed as single command
                out = self.t_travel(task.location_m)
                yield self.env.timeout(out); s.travel_empty += out     # empty travel out
                h = sum(self.tri(e) for e in p.picking_fixed)
                yield self.env.timeout(h); s.handling += h
                back = self.t_travel(task.location_m)
                yield self.env.timeout(back); s.travel_loaded += back
                s.pallets_done += p.pallets_per_cycle
                task.finished = self.env.now
                s.pick_cycle_times.append(task.finished - task.created)

            task.finished = task.finished or self.env.now
            s.busy += self.env.now - t0


# ----------------------------------------------------------------------------
# 3. EXPERIMENT
# ----------------------------------------------------------------------------

def run_once(policy: str, p: Params, seed: int) -> dict:
    rng = random.Random(seed)
    env = simpy.Environment()
    sim = WarehouseSim(env, p, policy, rng)
    env.run(until=p.warmup_minutes)
    sim.stats = Stats()                                     # discard warm-up statistics
    env.run(until=p.warmup_minutes + p.shift_minutes)
    s, dur = sim.stats, p.shift_minutes
    travel = s.travel_loaded + s.travel_empty
    return {
        "pallets/day": s.pallets_done,
        "empty travel %": 100 * s.travel_empty / travel if travel else 0.0,
        "RT utilization %": 100 * s.busy / (p.n_reach_trucks * dur),
        "mean wait (min)": statistics.mean(s.waits) if s.waits else 0.0,
        "pick cycle (min)": statistics.mean(s.pick_cycle_times) if s.pick_cycle_times else 0.0,
        "interleave rate %": 100 * s.interleaved_cycles / s.putaway_cycles if s.putaway_cycles else 0.0,
    }


def ci95(xs):
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    h = 1.96 * statistics.stdev(xs) / (len(xs) ** 0.5)
    return m, h


def main():
    policies = ["SINGLE", "INTERLEAVE", "NEAREST"]
    for fleet in (9, 6):
        p = Params(n_reach_trucks=fleet)
        print(f"\nScenario: {p.putaway_pallets_per_day:.0f} put-away + "
              f"{p.pick_pallets_per_day:.0f} pick pallets/day, "
              f"{p.n_reach_trucks} RTs, {p.n_replications} replications")
        header = f"{'metric':22s}" + "".join(f"{pol:>24s}" for pol in policies)
        print(header); print("-" * len(header))
        results = {pol: [run_once(pol, p, p.seed + r) for r in range(p.n_replications)]
                   for pol in policies}
        for metric in results["SINGLE"][0]:
            row = f"{metric:22s}"
            for pol in policies:
                m, h = ci95([r[metric] for r in results[pol]])
                row += f"{m:14.1f} \u00B1{h:6.1f}  "
            print(row)
    # TODO(research): add scenarios (demand \u00B120%, fleet size 6..10, overlap ratio),
    # common random numbers with paired-t tests, and monthly validation runs
    # against the 2025 observed throughput (Tables 11 and 13).


if __name__ == "__main__":
    main()
