"""January 2025 calibration using the observed DAILY records.
Step 1: recalibrate demand  - real daily means are 740 put-away / 831 picking
        pallets (not Table 14's 307/366) -> rerun baseline utilization check.
Step 2: calibrate tau       - find the holding time at which the simulated
        interleave rate reproduces the observed 61.2% (13,579 / 22,186).
Step 3: day-level validation - at tau*, simulate each January day with its own
        observed rates and compare simulated vs observed interleaved pallets."""
import csv, random, statistics, sys
import simpy
sys.path.insert(0, "/home/claude/paper")
import sim_interleaving as si

DAYS = [(r['date'], int(r['putaway']), int(r['picking']), int(r['interleave']))
        for r in csv.DictReader(open('/home/claude/paper/jan_daily.csv'))]
PA_MEAN, PK_MEAN = 740, 831          # real January daily means
OBS_IL_RATE = 61.2                   # observed interleave / put-away volume (%)

def run(policy, pa, pk, tau, fleet=9, seed=1, reps=8):
    il_pallets, busy, ilrate = [], [], []
    for rep in range(reps):
        rng = random.Random(seed + rep)
        env = simpy.Environment()
        p = si.Params(putaway_pallets_per_day=pa, pick_pallets_per_day=pk,
                      pick_hold_min=tau, n_reach_trucks=fleet)
        sim = si.WarehouseSim(env, p, policy, rng)
        env.run(until=p.warmup_minutes); sim.stats = si.Stats()
        env.run(until=p.warmup_minutes + p.shift_minutes)
        s = sim.stats
        il_pallets.append(s.interleaved_cycles * p.pallets_per_cycle)
        busy.append(100 * s.busy / (fleet * p.shift_minutes))
        ilrate.append(100 * s.interleaved_cycles / s.putaway_cycles if s.putaway_cycles else 0)
    return (statistics.mean(il_pallets), statistics.mean(busy), statistics.mean(ilrate))

print("=== Step 1: baseline utilization at REAL January demand (SINGLE, 9 RTs) ===")
_, util, _ = run("SINGLE", PA_MEAN, PK_MEAN, 0)
print(f"implied fleet utilization: {util:.1f}%   (was ~29% with Table 14 demand -- now realistic)")

print("\n=== Step 2: tau calibration at real demand (NEAREST, 9 RTs) ===")
print(f"{'tau (min)':>10s}{'sim interleave rate %':>24s}   target = {OBS_IL_RATE}%")
best = None
for tau in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
    _, _, ir = run("NEAREST", PA_MEAN, PK_MEAN, tau)
    print(f"{tau:>10.1f}{ir:>24.1f}")
    if best is None or abs(ir - OBS_IL_RATE) < abs(best[1] - OBS_IL_RATE):
        best = (tau, ir)
tau_star = best[0]
print(f"--> tau* = {tau_star} min (sim {best[1]:.1f}% vs observed {OBS_IL_RATE}%)")

print(f"\n=== Step 3: day-level validation at tau* = {tau_star} min ===")
sim_il, obs_il = [], []
for date, pa, pk, il in DAYS:
    if pa == 0:
        continue
    s_il, _, _ = run("NEAREST", pa, pk, tau_star, reps=5)
    sim_il.append(s_il); obs_il.append(il)
n = len(sim_il)
mad = statistics.mean(abs(a-b) for a,b in zip(sim_il, obs_il))
mo, ms = statistics.mean(obs_il), statistics.mean(sim_il)
cov = sum((a-ms)*(b-mo) for a,b in zip(sim_il,obs_il))/n
corr = cov/(statistics.pstdev(sim_il)*statistics.pstdev(obs_il))
print(f"days compared: {n} | mean observed {mo:.0f} vs simulated {ms:.0f} interleaved pallets/day")
print(f"mean absolute deviation: {mad:.0f} pallets/day ({100*mad/mo:.1f}% of observed mean)")
print(f"day-level correlation:   r = {corr:.3f}")
import json
json.dump({"tau_star": tau_star, "sim_daily": sim_il, "obs_daily": obs_il,
           "dates": [d for d,pa,_,_ in DAYS if pa>0]}, open("calib_results.json","w"))
