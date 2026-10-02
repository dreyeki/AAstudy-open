# -*- coding: utf-8 -*-
"""Discrete-event simulation of the AI Server assembly plant (HW2 supplement).
Time unit: minute. Each day = 1440 min; manual work window [0, WIN) where WIN = 450.
"""
import json, random, math, statistics as st
from collections import defaultdict
import simpy

DATA = json.load(open(__file__.replace("sim.py", "data.json")))
DAY = 1440
WIN = 450  # effective working minutes (8 h minus breaks)

STD = {"1U": dict(kit=6, asm=35, fct=20, bi=240, fqa=6, pack=8, fpy=0.95),
       "2U": dict(kit=8, asm=50, fct=25, bi=360, fqa=8, pack=10, fpy=0.95),
       "GPU": dict(kit=18, asm=150, fct=45, bi=480, fqa=20, pack=25, fpy=0.90)}
SEQ = ["GPU", "1U", "2U", "1U", "1U", "2U"]  # x10 per day -> 10 GPU, 30 1U, 20 2U

def tri(rng, m, lo=0.85, hi=1.35):
    return rng.triangular(m * lo, m * hi, m)

def dist_table(key):
    return {(r["a"], r["b"]): r["dist"] for r in DATA["ev"][key]["routes"]}

DEFAULT = dict(
    layout="F", n_amr=2, n_handler=3, slots=32, gpu_bays=7, fct=4, gpu_teams=2, test_win=WIN,
    cells="cells",           # "cells" (2 x 3 persons) | "line" (paced single line) | "hall" (6 benches)
    brk_mtbf=None, brk_mttr=None,  # station-level MTBF/MTTR (working minutes)
    gpu_per_day=10, days=25, warm=5, seed=1, pack_g=1, n_fqa=2,
    inbay=None,              # None -> follow layout (F = in-bay FQA/debug)
)

class Busy:
    """wraps a resource to accumulate busy time (unit-minutes) and hourly profile."""
    def __init__(self, env, cap, name):
        self.env = env; self.res = simpy.Resource(env, cap); self.cap = cap; self.name = name
        self.busy = 0.0; self.waits = []; self.hourly = defaultdict(float)
    def use(self, dur, n=1, prio_rec=True, win=None):
        env = self.env
        t0 = env.now
        while True:
            reqs = [self.res.request() for _ in range(n)]
            yield simpy.AllOf(env, reqs)
            t = env.now % DAY
            if win is None or t + dur <= win or (dur > win and t < 1):
                break
            for r in reqs:
                self.res.release(r)
            yield env.timeout(DAY - t)
        self.waits.append(env.now - t0)
        s = env.now
        yield env.timeout(dur)
        self._acc(s, env.now, n)
        for r in reqs:
            self.res.release(r)
    def _acc(self, s, e, n):
        self.busy += (e - s) * n
        t = s
        while t < e:
            h = int(t // 60); nxt = min(e, (h + 1) * 60)
            self.hourly[h] += (nxt - t) * n; t = nxt

class Plant:
    def __init__(self, cfg):
        self.cfg = c = dict(DEFAULT, **cfg)
        self.rng = random.Random(c["seed"])
        self.env = env = simpy.Environment()
        self.D = dist_table(c["layout"])
        self.inbay = (c["layout"] == "F") if c["inbay"] is None else c["inbay"]
        self.amr = Busy(env, c["n_amr"], "AMR")
        self.hand = Busy(env, c["n_handler"], "搬運人員")
        self.kit = Busy(env, 2, "Kitting")
        self.gteam = Busy(env, c["gpu_teams"], "GPU 組裝")
        self.fct = Busy(env, c["fct"], "FCT")
        self.ttech = Busy(env, 3, "測試技術員")
        self.gtech = Busy(env, 2, "GPU 測試技術員")
        self.fqa = Busy(env, c["n_fqa"], "FQA")
        self.pack12 = Busy(env, 2, "包裝 1U/2U")
        self.packG = Busy(env, c["pack_g"], "包裝 GPU")
        self.rep = Busy(env, 2, "維修")
        self.slots = simpy.Resource(env, c["slots"]); self.bays = simpy.Resource(env, c["gpu_bays"])
        self.slot_run = 0; self.slot_wait = 0; self.bay_run = 0; self.bay_wait = 0
        self.slot_busy = 0.0; self.bay_busy = 0.0
        # assembly
        self.cell_mode = c["cells"]
        if c["cells"] == "line":
            self.asm_units = [simpy.PreemptiveResource(env, 1)]
        elif c["cells"] == "hall":
            self.asm_units = [simpy.PreemptiveResource(env, 1) for _ in range(6)]
        else:
            self.asm_units = [simpy.PreemptiveResource(env, 1) for _ in range(2)]
        self.asm_busy = 0.0; self.asm_down = 0.0
        # stats
        self.done = []; self.wip = 0; self.wip_samples = []; self.slot_samples = []; self.bay_samples = []
        self.dist_total = 0.0; self.moves = 0
        self.batches = {}
        env.process(self.release())
        env.process(self.background())
        env.process(self.monitor())
        if c["brk_mtbf"]:
            for k, u in enumerate(self.asm_units):
                n_st = {"line": 6, "cells": 3, "hall": 1}[c["cells"]]
                env.process(self.breakdown(u, c["brk_mtbf"] / n_st, c["brk_mttr"]))

    # ---------- time windows
    def wait_window(self, dur, win=WIN):
        env = self.env
        while True:
            t = env.now % DAY
            if t + dur <= win or (dur > win and t < 1):
                return
            nxt = (env.now // DAY + 1) * DAY if t >= 0 else env.now
            if t + dur > win:
                yield env.timeout(nxt - env.now)

    def work(self, busy, dur, n=1, win=WIN):
        yield from self.wait_window(dur, win)
        yield from busy.use(dur, n, win=win)

    # ---------- transport
    def move(self, a, b, mode, n_person=1):
        """mode: 'amr' | 'man'. unit arrives after one-way; resource held for round trip."""
        env = self.env
        d = self.D.get((a, b))
        if d is None:
            d = self.D.get((b, a), 20)
        self.dist_total += 2 * d; self.moves += 1
        if mode == "amr":
            one = d / 48.0 + 1.0; res = self.amr
        else:
            one = d / 40.0 + 1.5; res = self.hand
        arrived = env.event()
        def cyc():
            yield from self.wait_window(2 * one)
            t0 = env.now
            while True:
                reqs = [res.res.request() for _ in range(n_person)]
                yield simpy.AllOf(env, reqs)
                t = env.now % DAY
                if t + 2 * one <= WIN:
                    break
                for r in reqs:
                    res.res.release(r)
                yield env.timeout(DAY - t)
            res.waits.append(env.now - t0)
            s = env.now
            yield env.timeout(one)
            arrived.succeed()
            yield env.timeout(one)
            res._acc(s, env.now, n_person)
            for r in reqs:
                res.res.release(r)
        env.process(cyc())
        return arrived

    def batch_move(self, key, a, b, mode, size, unit_evt_list, timeout=30):
        """put a waiting event into batch; dispatch when full or timeout."""
        env = self.env
        B = self.batches.setdefault(key, {"items": [], "timer": None})
        ev = env.event(); B["items"].append(ev)
        if len(B["items"]) >= size:
            self._dispatch(key, a, b, mode)
        elif len(B["items"]) == 1:
            def timer(first=ev):
                yield env.timeout(timeout)
                if B["items"] and B["items"][0] is first:
                    self._dispatch(key, a, b, mode)
            env.process(timer())
        return ev

    def _dispatch(self, key, a, b, mode):
        B = self.batches[key]; items = B["items"]; B["items"] = []
        arr = self.move(a, b, mode)
        def fire():
            yield arr
            for e in items:
                e.succeed()
        self.env.process(fire())

    # ---------- generators
    def release(self):
        env = self.env; c = self.cfg
        day = 0
        while True:
            g = c["gpu_per_day"]
            if g == 10:
                seq = SEQ * 10
            else:
                others = (["1U", "2U", "1U", "1U", "2U"] * 10)
                seq = []; k = 0
                for i in range(g):
                    seq.append("GPU")
                    take = round((i + 1) * 50 / g) - round(i * 50 / g)
                    seq += others[k:k + take]; k += take
            for i, f in enumerate(seq):
                env.process(self.unit(f, day, i))
            day += 1
            yield env.timeout(DAY)

    def background(self):
        env = self.env; F = self.cfg["layout"] == "F"
        jobs = [("REC", "IQC", "man", 24), ("IQC", "WH", "man", 23), ("IQC", "REC", "man", 1),
                ("WH", "KIT", "amr", 30), ("WH", "KIT", "man", 8), ("KIT", "REC", "amr", 4),
                ("REP", "WH", "man", 2), ("WH", "REP", "man", 2)]
        jobs.append(("SHIP", "PACK", "man", 4) if F else ("REC", "PACK", "man", 4))
        if F:
            jobs += [("WH", "GPUT", "man", 1), ("GPUT", "WH", "man", 1)]
        while True:
            base = env.now
            for a, b, mode, n in jobs:
                for _ in range(n):
                    t = self.rng.uniform(0, WIN - 30)
                    env.process(self._later(t, a, b, mode))
            yield env.timeout(DAY)

    def _later(self, t, a, b, mode):
        yield self.env.timeout(t)
        yield self.move(a, b, mode)

    def monitor(self):
        env = self.env
        while True:
            self.wip_samples.append((env.now, self.wip))
            self.slot_samples.append((env.now, self.slot_run, self.slot_wait))
            self.bay_samples.append((env.now, self.bay_run, self.bay_wait))
            yield env.timeout(10)

    def advance_work(self, tau):
        """advance clock by tau minutes of *working-window* time."""
        env = self.env
        while tau > 1e-9:
            t = env.now % DAY
            if t >= WIN:
                yield env.timeout(DAY - t); continue
            step = min(tau, WIN - t)
            yield env.timeout(step); tau -= step

    def breakdown(self, unit, mtbf, mttr):
        env = self.env; rng = self.rng
        while True:
            yield from self.advance_work(rng.expovariate(1.0 / mtbf))
            with unit.request(priority=-1, preempt=True) as r:
                yield r
                d = rng.expovariate(1.0 / mttr)
                self.asm_down += d
                yield from self.advance_work(d)

    # ---------- assembly
    def assemble12(self, f):
        env = self.env; c = self.cfg
        if self.cell_mode == "line":
            dur = 50.0 / 6  # paced by slowest (2U) content per station
            flow = 5 * dur
        elif self.cell_mode == "hall":
            dur = tri(self.rng, STD[f]["asm"]); flow = 0
        else:
            dur = tri(self.rng, STD[f]["asm"] / 3.0); flow = 0
        unit = min(self.asm_units, key=lambda u: len(u.queue) + u.count)
        remaining = dur
        yield from self.wait_window(remaining)
        while remaining > 1e-6:
            yield from self.wait_window(remaining)
            with unit.request(priority=0) as req:
                yield req
                if env.now % DAY + remaining > WIN:
                    continue
                s = env.now
                try:
                    yield env.timeout(remaining)
                    self.asm_busy += remaining; remaining = 0
                except simpy.Interrupt:
                    self.asm_busy += env.now - s; remaining -= env.now - s
        if flow:
            yield env.timeout(flow)

    # ---------- unit life
    def unit(self, f, day, idx):
        env = self.env; rng = self.rng; s = STD[f]; t_rel = env.now
        self.wip += 1
        gpu = f == "GPU"
        # kitting
        yield from self.work(self.kit, tri(rng, s["kit"]))
        if gpu:
            e1 = self.move("KIT", "GPUA", "amr"); e2 = self.move("KIT", "GPUA", "amr")
            yield simpy.AllOf(env, [e1, e2])
            yield from self.work(self.gteam, tri(rng, s["asm"] / 2.0))
            yield self.move("GPUA", "GPUT", "man", 2)
            yield from self.gpu_test(f)
        else:
            yield self.batch_move("KA", "KIT", "ASM", "amr", 2, None, timeout=20)
            yield from self.assemble12(f)
            yield self.batch_move("AT", "ASM", "TST", "amr", 4, None, timeout=30)
            yield from self.test12(f)
            yield self.batch_move("TF", "TST", "FQA", "amr", 4, None, timeout=30)
            yield from self.work(self.fqa, tri(rng, s["fqa"]))
            if rng.random() < 0.02:
                yield self.move("FQA", "REP", "man")
                yield from self.work(self.rep, rng.lognormvariate(math.log(45), 0.5))
                yield self.move("REP", "FQA", "man") if ("REP", "FQA") in self.D else self.move("FQA", "REP", "man")
                yield from self.work(self.fqa, tri(rng, s["fqa"]))
            yield self.batch_move("FP", "FQA", "PACK", "amr", 4, None, timeout=30)
        # pack
        yield from self.work(self.packG if gpu else self.pack12, tri(rng, s["pack"]))
        yield self.batch_move("PS" + ("G" if gpu else ""), "PACK", "SHIP", "man", 1 if gpu else 6, None, timeout=60)
        self.wip -= 1
        self.done.append(dict(f=f, rel=t_rel, fin=env.now, day=int(env.now // DAY), rday=day))

    def test12(self, f):
        env = self.env; rng = self.rng; s = STD[f]
        while True:
            yield from self.work(self.fct, tri(rng, s["fct"]))
            req = self.slots.request(); yield req
            yield from self.work(self.ttech, 3, win=self.cfg["test_win"])
            self.slot_run += 1; t0 = env.now
            yield env.timeout(s["bi"])
            self.slot_run -= 1; self.slot_wait += 1
            yield from self.work(self.ttech, 3, win=self.cfg["test_win"])
            self.slot_wait -= 1; self.slot_busy += env.now - t0
            self.slots.release(req)
            if rng.random() < s["fpy"]:
                return
            yield self.move("TST", "REP", "man")
            yield from self.work(self.rep, rng.lognormvariate(math.log(50), 0.5))
            yield self.move("REP", "TST", "man")
            # retest: FCT only
            yield from self.work(self.fct, tri(rng, s["fct"]))
            return

    def gpu_test(self, f):
        env = self.env; rng = self.rng; s = STD[f]; tw = self.cfg["test_win"]
        req = self.bays.request(); yield req
        t0 = env.now
        yield from self.work(self.gtech, 8, win=tw)          # dock, cable, coolant
        self.bay_run += 1
        yield env.timeout(tri(rng, s["fct"]) + s["bi"])      # FCT + burn-in (machine)
        self.bay_run -= 1; self.bay_wait += 1
        yield from self.work(self.gtech, 8, win=tw)          # read result / undock
        self.bay_wait -= 1
        fail = rng.random() > s["fpy"]
        if self.inbay:
            # 測畢移至測試區內暫存位（不需搬運設備），釋放灣位；除錯與 FQA 於區內原位執行
            self.bay_busy += env.now - t0
            self.bays.release(req)
            if fail:
                yield from self.work(self.rep, rng.lognormvariate(math.log(60), 0.5))
                req = self.bays.request(); yield req
                yield from self.work(self.gtech, 8, win=tw)
                self.bay_run += 1; yield env.timeout(tri(rng, s["fct"])); self.bay_run -= 1
                yield from self.work(self.gtech, 5, win=tw)
                self.bays.release(req)
            yield from self.work(self.fqa, tri(rng, s["fqa"]))   # 區內原位 FQA
            yield self.move("GPUT", "PACK", "man", 2)
        else:
            self.bay_busy += env.now - t0
            self.bays.release(req)
            if fail:
                yield self.move("GPUT", "REP", "man", 2)
                yield from self.work(self.rep, rng.lognormvariate(math.log(60), 0.5))
                yield self.move("REP", "GPUT", "man", 2)
                req = self.bays.request(); yield req
                yield from self.work(self.gtech, 8, win=tw)
                self.bay_run += 1; yield env.timeout(tri(rng, s["fct"])); self.bay_run -= 1
                self.bay_busy += 0
                self.bays.release(req)
            yield self.move("GPUT", "FQA", "man", 2)
            yield from self.work(self.fqa, tri(rng, s["fqa"]))
            yield self.move("FQA", "PACK", "man", 2)

    # ---------- run
    def run(self):
        c = self.cfg
        self.env.run(until=c["days"] * DAY)
        return self.summary()

    def summary(self):
        c = self.cfg; w0 = c["warm"] * DAY; T = (c["days"] - c["warm"]) * DAY; nd = c["days"] - c["warm"]
        done = [u for u in self.done if u["fin"] >= w0]
        out = {}
        out["thr_day"] = len(done) / nd
        for f in ["1U", "2U", "GPU"]:
            L = [(u["fin"] - u["rel"]) / 60 for u in done if u["f"] == f]
            out["thr_" + f] = len(L) / nd
            out["lt_" + f] = st.mean(L) if L else float("nan")
            out["lt90_" + f] = sorted(L)[int(0.9 * len(L))] if L else float("nan")
        L = [(u["fin"] - u["rel"]) / 60 for u in done]
        out["lt"] = st.mean(L) if L else float("nan")
        ws = [w for t, w in self.wip_samples if t >= w0]
        out["wip"] = st.mean(ws) if ws else 0
        out["wip_end"] = self.wip
        days_ok = defaultdict(int)
        for u in done:
            days_ok[u["day"]] += 1
        tgt = 50 + c["gpu_per_day"]
        out["days_met"] = sum(1 for d in range(c["warm"], c["days"]) if days_ok[d] >= 0.95 * tgt) / nd
        def util(b, cap=None):
            return b.busy / ((cap or b.cap) * c["days"] * WIN)
        out["u_amr"] = util(self.amr); out["u_hand"] = util(self.hand)
        out["w_amr"] = st.mean(self.amr.waits) if self.amr.waits else 0
        out["w_hand"] = st.mean(self.hand.waits) if self.hand.waits else 0
        out["w_amr_p95"] = sorted(self.amr.waits)[int(0.95 * len(self.amr.waits))] if self.amr.waits else 0
        out["u_kit"] = util(self.kit); out["u_gteam"] = util(self.gteam); out["u_fct"] = util(self.fct)
        out["u_fqa"] = util(self.fqa); out["u_rep"] = util(self.rep)
        out["u_asm"] = self.asm_busy / (len(self.asm_units) * c["days"] * WIN)
        out["asm_down"] = self.asm_down / c["days"]
        ss = [(r, w) for t, r, w in self.slot_samples if t >= w0]
        out["slot_run"] = st.mean(r for r, w in ss) / c["slots"]
        out["slot_idle_occ"] = st.mean(w for r, w in ss) / c["slots"]
        bs = [(r, w) for t, r, w in self.bay_samples if t >= w0]
        out["bay_run"] = st.mean(r for r, w in bs) / c["gpu_bays"]
        out["bay_idle_occ"] = st.mean(w for r, w in bs) / c["gpu_bays"]
        out["dist_day"] = self.dist_total / c["days"]
        out["amr_hourly"] = {h % 24: 0 for h in range(24)}
        cnt = defaultdict(int)
        for h, v in self.amr.hourly.items():
            if h * 60 >= w0:
                out["amr_hourly"][h % 24] += v; cnt[h % 24] += 1
        out["amr_hourly"] = {h: out["amr_hourly"][h] / (nd * 60 * self.amr.cap) for h in range(24)}
        out["hand_hourly"] = {h: 0 for h in range(24)}
        for h, v in self.hand.hourly.items():
            if h * 60 >= w0:
                out["hand_hourly"][h % 24] += v
        out["hand_hourly"] = {h: out["hand_hourly"][h] / (nd * 60 * self.hand.cap) for h in range(24)}
        # slot timeline of a representative day (last full day)
        dlast = c["days"] - 2
        out["slot_tl"] = [(t - dlast * DAY, r, w) for t, r, w in self.slot_samples if dlast * DAY <= t < (dlast + 1) * DAY]
        out["bay_tl"] = [(t - dlast * DAY, r, w) for t, r, w in self.bay_samples if dlast * DAY <= t < (dlast + 1) * DAY]
        out["daily"] = [days_ok[d] for d in range(c["warm"], c["days"])]
        return out

def replicate(cfg, reps=20, seed0=100):
    runs = [Plant(dict(cfg, seed=seed0 + k)).run() for k in range(reps)]
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    agg = {}
    for k in keys:
        xs = [r[k] for r in runs if not (isinstance(r[k], float) and math.isnan(r[k]))]
        m = st.mean(xs) if xs else float("nan"); sd = st.stdev(xs) if len(xs) > 1 else 0
        agg[k] = dict(mean=m, sd=sd, ci=2.093 * sd / math.sqrt(len(xs)) if xs else 0)
    agg["amr_hourly"] = {h: st.mean(r["amr_hourly"][h] for r in runs) for h in range(24)}
    agg["hand_hourly"] = {h: st.mean(r["hand_hourly"][h] for r in runs) for h in range(24)}
    agg["slot_tl"] = runs[0]["slot_tl"]; agg["bay_tl"] = runs[0]["bay_tl"]
    agg["daily_all"] = [x for r in runs for x in r["daily"]]
    agg["thr_runs"] = [r["thr_day"] for r in runs]
    return agg

if __name__ == "__main__":
    import time
    t = time.time()
    r = Plant(dict(layout="F")).run()
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (dict, list))})
    print("sec", time.time() - t)
