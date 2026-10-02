# -*- coding: utf-8 -*-
"""Run the HW2 simulation with full event tracing for the web visualisation."""
import json, math, simpy
import sim as S
import model as m
from sim import Plant, Busy, tri, STD, DAY, WIN

ST = {"q": 0, "w": 1, "t": 2, "tr": 3, "bi": 4, "bw": 5, "gb": 6, "gw": 7, "sh": 8}
DEPTS = m.DEPTS

class TPlant(Plant):
    def __init__(self, cfg):
        self.ev = []          # [t, uid, dept_idx, state, k]
        self.mv = []          # [id, t_start, t_arr, t_end, a, b, mode, carriers, uids, n_person]
        self.units = []       # [uid, fam, t_rel]
        self.uid = 0
        self.pool = {}
        self.slot_free = None; self.bay_free = None
        super().__init__(cfg)
        self.pool = {"amr": list(range(self.cfg["n_amr"])), "man": list(range(self.cfg["n_handler"]))}
        self.slot_free = list(range(self.cfg["slots"])); self.bay_free = list(range(self.cfg["gpu_bays"]))

    def E(self, uid, dept, st, k=-1):
        self.ev.append([round(self.env.now, 2), uid, DEPTS.index(dept), ST[st], k])

    # traced work: state q while waiting, w when started
    def work(self, busy, dur, n=1, win=WIN, uid=None, dept=None):
        if uid is not None:
            self.E(uid, dept, "q")
        yield from self.wait_window(dur, win)
        env = self.env
        while True:
            reqs = [busy.res.request() for _ in range(n)]
            yield simpy.AllOf(env, reqs)
            t = env.now % DAY
            if win is None or t + dur <= win or (dur > win and t < 1):
                break
            for r in reqs:
                busy.res.release(r)
            yield env.timeout(DAY - t)
        if uid is not None:
            self.E(uid, dept, "w")
        s = env.now
        yield env.timeout(dur)
        busy._acc(s, env.now, n)
        for r in reqs:
            busy.res.release(r)

    def move(self, a, b, mode, n_person=1, uids=()):
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
            car = [self.pool[mode].pop(0) for _ in range(n_person)]
            s = env.now
            mid = len(self.mv)
            self.mv.append([mid, round(s, 2), round(s + one, 2), round(s + 2 * one, 2), a, b, mode, car, list(uids), n_person])
            for u in uids:
                self.E(u, a, "tr", mid)
            yield env.timeout(one)
            for u in uids:
                self.E(u, b, "q")
            arrived.succeed()
            yield env.timeout(one)
            res._acc(s, env.now, n_person)
            for c in car:
                self.pool[mode].append(c)
            self.pool[mode].sort()
            for r in reqs:
                res.res.release(r)
        env.process(cyc())
        return arrived

    def batch_move(self, key, a, b, mode, size, uid, timeout=30):
        env = self.env
        self.E(uid, a, "t")
        B = self.batches.setdefault(key, {"items": [], "uids": []})
        ev = env.event(); B["items"].append(ev); B["uids"].append(uid)
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
        B = self.batches[key]; items = B["items"]; uids = B["uids"]; B["items"] = []; B["uids"] = []
        arr = self.move(a, b, mode, uids=uids)
        def fire():
            yield arr
            for e in items:
                e.succeed()
        self.env.process(fire())

    def assemble12(self, f, uid=None):
        env = self.env
        if self.cell_mode == "line":
            dur = 50.0 / 6; flow = 5 * dur
        elif self.cell_mode == "hall":
            dur = tri(self.rng, STD[f]["asm"]); flow = 0
        else:
            dur = tri(self.rng, STD[f]["asm"] / 3.0); flow = 0
        unit = min(self.asm_units, key=lambda u: len(u.queue) + u.count)
        k = self.asm_units.index(unit)
        remaining = dur
        self.E(uid, "ASM", "q")
        yield from self.wait_window(remaining)
        while remaining > 1e-6:
            yield from self.wait_window(remaining)
            with unit.request(priority=0) as req:
                yield req
                if env.now % DAY + remaining > WIN:
                    continue
                self.E(uid, "ASM", "w", k)
                s = env.now
                try:
                    yield env.timeout(remaining)
                    self.asm_busy += remaining; remaining = 0
                except simpy.Interrupt:
                    self.asm_busy += env.now - s; remaining -= env.now - s
                    self.E(uid, "ASM", "q")
        if flow:
            yield env.timeout(flow)

    def unit(self, f, day, idx):
        env = self.env; rng = self.rng; s = STD[f]; t_rel = env.now
        uid = self.uid; self.uid += 1
        self.units.append([uid, f, round(t_rel, 2)])
        self.wip += 1
        gpu = f == "GPU"
        yield from self.work(self.kit, tri(rng, s["kit"]), uid=uid, dept="KIT")
        if gpu:
            self.E(uid, "KIT", "t")
            e1 = self.move("KIT", "GPUA", "amr", uids=[uid]); e2 = self.move("KIT", "GPUA", "amr")
            yield simpy.AllOf(env, [e1, e2])
            yield from self.work(self.gteam, tri(rng, s["asm"] / 2.0), uid=uid, dept="GPUA")
            self.E(uid, "GPUA", "t")
            yield self.move("GPUA", "GPUT", "man", 2, uids=[uid])
            yield from self.gpu_test(f, uid)
        else:
            yield self.batch_move("KA", "KIT", "ASM", "amr", 2, uid, timeout=20)
            yield from self.assemble12(f, uid)
            yield self.batch_move("AT", "ASM", "TST", "amr", 4, uid, timeout=30)
            yield from self.test12(f, uid)
            yield self.batch_move("TF", "TST", "FQA", "amr", 4, uid, timeout=30)
            yield from self.work(self.fqa, tri(rng, s["fqa"]), uid=uid, dept="FQA")
            if rng.random() < 0.02:
                self.E(uid, "FQA", "t")
                yield self.move("FQA", "REP", "man", uids=[uid])
                yield from self.work(self.rep, rng.lognormvariate(math.log(45), 0.5), uid=uid, dept="REP")
                self.E(uid, "REP", "t")
                yield self.move("REP", "FQA", "man", uids=[uid])
                yield from self.work(self.fqa, tri(rng, s["fqa"]), uid=uid, dept="FQA")
            yield self.batch_move("FP", "FQA", "PACK", "amr", 4, uid, timeout=30)
        yield from self.work(self.packG if gpu else self.pack12, tri(rng, s["pack"]), uid=uid, dept="PACK")
        yield self.batch_move("PS" + ("G" if gpu else ""), "PACK", "SHIP", "man", 1 if gpu else 6, uid, timeout=60)
        self.E(uid, "SHIP", "sh")
        self.wip -= 1
        self.done.append(dict(f=f, rel=t_rel, fin=env.now, day=int(env.now // DAY), rday=day))

    def test12(self, f, uid):
        env = self.env; rng = self.rng; s = STD[f]
        yield from self.work(self.fct, tri(rng, s["fct"]), uid=uid, dept="TST")
        self.E(uid, "TST", "t")
        req = self.slots.request(); yield req
        k = self.slot_free.pop(0)
        yield from self.work(self.ttech, 3, win=self.cfg["test_win"])
        self.slot_run += 1; t0 = env.now
        self.E(uid, "TST", "bi", k)
        yield env.timeout(s["bi"])
        self.slot_run -= 1; self.slot_wait += 1
        self.E(uid, "TST", "bw", k)
        yield from self.work(self.ttech, 3, win=self.cfg["test_win"])
        self.slot_wait -= 1; self.slot_busy += env.now - t0
        self.slots.release(req); self.slot_free.append(k); self.slot_free.sort()
        self.E(uid, "TST", "t")
        if rng.random() < s["fpy"]:
            return
        yield self.move("TST", "REP", "man", uids=[uid])
        yield from self.work(self.rep, rng.lognormvariate(math.log(50), 0.5), uid=uid, dept="REP")
        self.E(uid, "REP", "t")
        yield self.move("REP", "TST", "man", uids=[uid])
        yield from self.work(self.fct, tri(rng, s["fct"]), uid=uid, dept="TST")
        self.E(uid, "TST", "t")

    def gpu_test(self, f, uid):
        env = self.env; rng = self.rng; s = STD[f]; tw = self.cfg["test_win"]
        self.E(uid, "GPUT", "q")
        req = self.bays.request(); yield req
        k = self.bay_free.pop(0)
        t0 = env.now
        yield from self.work(self.gtech, 8, win=tw)
        self.bay_run += 1
        self.E(uid, "GPUT", "gb", k)
        yield env.timeout(tri(rng, s["fct"]) + s["bi"])
        self.bay_run -= 1; self.bay_wait += 1
        self.E(uid, "GPUT", "gw", k)
        yield from self.work(self.gtech, 8, win=tw)
        self.bay_wait -= 1
        fail = rng.random() > s["fpy"]
        if self.inbay:
            self.bay_busy += env.now - t0
            self.bays.release(req); self.bay_free.append(k); self.bay_free.sort()
            self.E(uid, "GPUT", "q")
            if fail:
                yield from self.work(self.rep, rng.lognormvariate(math.log(60), 0.5), uid=uid, dept="GPUT")
                self.E(uid, "GPUT", "q")
                req = self.bays.request(); yield req
                k = self.bay_free.pop(0)
                yield from self.work(self.gtech, 8, win=tw)
                self.bay_run += 1; self.E(uid, "GPUT", "gb", k)
                yield env.timeout(tri(rng, s["fct"])); self.bay_run -= 1
                self.E(uid, "GPUT", "gw", k)
                yield from self.work(self.gtech, 5, win=tw)
                self.bays.release(req); self.bay_free.append(k); self.bay_free.sort()
            yield from self.work(self.fqa, tri(rng, s["fqa"]), uid=uid, dept="GPUT")
            self.E(uid, "GPUT", "t")
            yield self.move("GPUT", "PACK", "man", 2, uids=[uid])
        else:
            self.bay_busy += env.now - t0
            self.bays.release(req); self.bay_free.append(k); self.bay_free.sort()
            self.E(uid, "GPUT", "t")
            if fail:
                yield self.move("GPUT", "REP", "man", 2, uids=[uid])
                yield from self.work(self.rep, rng.lognormvariate(math.log(60), 0.5), uid=uid, dept="REP")
                self.E(uid, "REP", "t")
                yield self.move("REP", "GPUT", "man", 2, uids=[uid])
                self.E(uid, "GPUT", "q")
                req = self.bays.request(); yield req
                k = self.bay_free.pop(0)
                yield from self.work(self.gtech, 8, win=tw)
                self.bay_run += 1; self.E(uid, "GPUT", "gb", k)
                yield env.timeout(tri(rng, s["fct"])); self.bay_run -= 1
                self.bays.release(req); self.bay_free.append(k); self.bay_free.sort()
                self.E(uid, "GPUT", "t")
            yield self.move("GPUT", "FQA", "man", 2, uids=[uid])
            yield from self.work(self.fqa, tri(rng, s["fqa"]), uid=uid, dept="FQA")
            self.E(uid, "FQA", "t")
            yield self.move("FQA", "PACK", "man", 2, uids=[uid])

    def background(self):
        env = self.env; F = self.cfg["layout"] == "F"
        jobs = [("REC", "IQC", "man", 24), ("IQC", "WH", "man", 23), ("IQC", "REC", "man", 1),
                ("WH", "KIT", "amr", 30), ("WH", "KIT", "man", 8), ("KIT", "REC", "amr", 4),
                ("REP", "WH", "man", 2), ("WH", "REP", "man", 2)]
        jobs.append(("SHIP", "PACK", "man", 4) if F else ("REC", "PACK", "man", 4))
        if F:
            jobs += [("WH", "GPUT", "man", 1), ("GPUT", "WH", "man", 1)]
        while True:
            for a, b, mode, n in jobs:
                for _ in range(n):
                    t = self.rng.uniform(0, WIN - 30)
                    env.process(self._later(t, a, b, mode))
            yield env.timeout(DAY)

def paths_for(layout_key, pairs):
    lay = m.L[layout_key]; G = m.grid(lay); out = {}
    for a, b in pairs:
        d, path, _ = m.shortest(lay, a, b, G)
        pts = [(c[0] + 0.5, c[1] + 0.5) for c in path]
        # simplify collinear
        simp = [pts[0]]
        for i in range(1, len(pts) - 1):
            (x0, y0), (x1, y1), (x2, y2) = simp[-1], pts[i], pts[i + 1]
            if (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1) != 0:
                simp.append(pts[i])
        simp.append(pts[-1])
        out[a + ">" + b] = [[round(x, 1), round(y, 1)] for x, y in simp]
    return out

def layout_geom(key):
    lay = m.L[key]
    g = dict(depts={k: list(v) for k, v in lay["depts"].items()}, reserve={k: list(v) for k, v in lay["reserve"].items()},
             aisles=[list(r) for r in lay["aisles"]], docks=lay["docks"], name=lay["name"])
    if key == "F":
        g["walks"] = [[16, 0, 17.2, 40], [16, 20.8, 60, 22], [44, 0, 45.2, 18], [44, 22, 45.2, 40]]
    return g

def run(name, cfg, days=9):
    p = TPlant(dict(cfg, days=days, seed=cfg.get("seed", 101)))
    p.env.run(until=days * DAY)
    T0, T1 = 5 * DAY, 8 * DAY
    fin = {}
    for e in p.ev:
        if e[3] == ST["sh"]: fin[e[1]] = e[0]
    keep = {u[0] for u in p.units if u[2] < T1 and fin.get(u[0], 1e18) >= T0}
    units = [u for u in p.units if u[0] in keep]
    ev = [[round(e[0], 1)] + e[1:] for e in p.ev if e[1] in keep and e[0] < T1]
    mv = [[x[0], round(x[1], 1), round(x[2], 1), round(x[3], 1)] + x[4:] for x in p.mv if x[3] >= T0 - 1 and x[1] < T1]
    pairs = sorted({(x[4], x[5]) for x in mv})
    out = dict(units=units, ev=ev, mv=mv, T0=T0, T1=T1, paths=paths_for(cfg["layout"], pairs), cfg={k: p.cfg[k] for k in ("layout", "n_amr", "n_handler", "slots", "gpu_bays", "test_win")})
    done = [u for u in p.done if 5 * DAY <= u["fin"] < 8 * DAY]
    print(name, "units", len(p.units), "ev", len(p.ev), "mv", len(p.mv), "shipped d5-7", len(done))
    return out

if __name__ == "__main__":
    SC = {
        "final2": dict(layout="F", cells="cells", test_win=900),
        "final1": dict(layout="F", cells="cells", test_win=450),
        "A2": dict(layout="A", cells="hall", test_win=900),
    }
    R = {k: run(k, v) for k, v in SC.items()}
    geom = {k: layout_geom(k) for k in ("F", "A")}
    X = {}
    for p in (1, 2, 3):
        X.update(json.load(open(f"exp{p}.json")))
    keys = ["thr_day", "thr_1U", "thr_2U", "thr_GPU", "lt_1U", "lt_2U", "lt_GPU", "wip", "days_met", "u_amr", "u_hand", "w_amr", "w_hand",
            "dist_day", "slot_run", "slot_idle_occ", "bay_run", "bay_idle_occ"]
    summ = {"final2": {k: X["E3_F"][k]["mean"] for k in keys}, "final1": {k: X["E1_F_1shift"][k]["mean"] for k in keys},
            "A2": {k: X["E3_A"][k]["mean"] for k in keys}}
    data = dict(depts=DEPTS, names={k: m.DNAME[k] for k in DEPTS}, short={k: m.DSHORT[k] for k in DEPTS}, cat=m.CAT,
                geom=geom, scen=R, summary=summ)
    s = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    import os; os.makedirs("web", exist_ok=True)
    open("web/data.js", "w").write("window.SIMDATA=" + s + ";")
    print("bytes", len(s))
