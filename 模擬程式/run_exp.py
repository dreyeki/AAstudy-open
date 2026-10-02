# -*- coding: utf-8 -*-
import json, sys, time
from sim import replicate

REPS = 20
POLICY = dict(test_win=900)   # 建議營運政策：測試區加開中班（裝卸/判讀）
LAY = {"A": dict(layout="A", cells="hall"), "B": dict(layout="B", cells="line"),
       "C": dict(layout="C", cells="cells"), "F": dict(layout="F", cells="cells")}
KEEP = ["thr_day", "thr_1U", "thr_2U", "thr_GPU", "lt", "lt_1U", "lt_2U", "lt_GPU", "lt90_1U", "lt90_2U", "lt90_GPU", "wip",
        "days_met", "u_amr", "u_hand", "w_amr", "w_hand", "w_amr_p95", "u_kit", "u_gteam", "u_fct", "u_fqa", "u_rep", "u_asm",
        "asm_down", "slot_run", "slot_idle_occ", "bay_run", "bay_idle_occ", "dist_day"]

def slim(a, extra=()):
    o = {k: a[k] for k in KEEP if k in a}
    for e in extra:
        o[e] = a[e]
    return o

def run(name, cfg, extra=()):
    t = time.time()
    a = replicate(cfg, reps=REPS)
    print(name, round(time.time() - t, 1), "s", round(a["thr_day"]["mean"], 2), round(a["thr_GPU"]["mean"], 2), flush=True)
    return slim(a, extra)

part = sys.argv[1]
R = {}
if part == "1":
    R["E1_F_1shift"] = run("E1", dict(LAY["F"]), extra=("slot_tl", "bay_tl", "amr_hourly", "hand_hourly", "daily_all"))
    R["E1_F_2shift"] = run("E1b", dict(LAY["F"], **POLICY), extra=("slot_tl", "bay_tl", "amr_hourly", "hand_hourly", "daily_all"))
    for w in (450, 900):
        for b in range(5, 13):
            R[f"E2_bay_{w}_{b}"] = run(f"bay{w}-{b}", dict(LAY["F"], test_win=w, gpu_bays=b))
        for sl in (16, 20, 24, 28, 32):
            R[f"E2_slot_{w}_{sl}"] = run(f"slot{w}-{sl}", dict(LAY["F"], test_win=w, slots=sl))
elif part == "2":
    for k in "ABCF":
        R[f"E3_{k}"] = run("E3" + k, dict(LAY[k], **POLICY), extra=("amr_hourly", "hand_hourly", "daily_all", "thr_runs"))
    for k in "AF":
        for n in (1, 2, 3):
            R[f"E4_amr_{k}_{n}"] = run(f"amr{k}{n}", dict(LAY[k], n_amr=n, **POLICY))
        for n in (2, 3, 4):
            R[f"E4_hand_{k}_{n}"] = run(f"hand{k}{n}", dict(LAY[k], n_handler=n, **POLICY))
elif part == "3":
    for lvl, (mtbf, mttr) in {"none": (None, None), "mild": (1200, 30), "severe": (480, 45)}.items():
        for k in "BF":
            R[f"E5_{k}_{lvl}"] = run(f"brk{k}{lvl}", dict(LAY[k], brk_mtbf=mtbf, brk_mttr=mttr, **POLICY), extra=("daily_all", "thr_runs"))
    R["E6_now"] = run("g20now", dict(LAY["F"], gpu_per_day=20, **POLICY))
    R["E6_teams"] = run("g20teams", dict(LAY["F"], gpu_per_day=20, gpu_teams=4, pack_g=2, **POLICY))
    R["E6_full"] = run("g20full", dict(LAY["F"], gpu_per_day=20, gpu_teams=4, pack_g=2, gpu_bays=12, **POLICY))
json.dump(R, open(f"exp{part}.json", "w"), ensure_ascii=False, default=float)
print("done", part)
