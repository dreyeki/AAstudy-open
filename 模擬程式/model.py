# -*- coding: utf-8 -*-
"""SLP data model for AI Server assembly plant (HW2).
All numbers used in the report come from here."""
import math, heapq, itertools, json
import numpy as np

W, H = 60, 40  # building (m)
DAYS, HRS = 20, 8
T_GROSS = HRS * 60          # 480 min
T_AVAIL = 450               # 480 - 早會10 - 休息2x10
ALLOW = 0.90                # 人員寬放 PF&D -> 405 min effective
T_EFF = T_AVAIL * ALLOW     # 405
EQ_AVAIL = 0.85             # 設備稼動(OEE 概念)

# ---------------- P & Q ----------------
FAM = ["1U", "2U", "GPU"]
QDAY = {"1U": 30, "2U": 20, "GPU": 10}
SKU = [  # (sku, family, qty/day, description)
    ("1U-S1", "1U", 12, "通用運算 2S/16 DIMM/4 NVMe"),
    ("1U-S2", "1U", 9, "雲端前端 1S/12 DIMM/10 NVMe"),
    ("1U-S3", "1U", 6, "網通/邊緣 1S/8 DIMM/2 SSD"),
    ("1U-S4", "1U", 3, "客製 BMC/FW 版本"),
    ("2U-S1", "2U", 8, "儲存型 2S/32 DIMM/24 SSD"),
    ("2U-S2", "2U", 6, "推論型 2S+2 PCIe GPU"),
    ("2U-S3", "2U", 4, "虛擬化 2S/32 DIMM/12 SSD"),
    ("2U-S4", "2U", 2, "液冷驗證機"),
    ("GPU-G1", "GPU", 5, "8-GPU HGX 氣冷 8U"),
    ("GPU-G2", "GPU", 3, "8-GPU HGX 液冷(冷板)"),
    ("GPU-G3", "GPU", 2, "4-GPU PCIe 4U"),
]
# per-unit standard data (min) & physical attributes
STD = {  # kit, asm, fct, burn-in(h), fqa, pack
    "1U": dict(kit=6, asm=35, fct=20, bi=4, fqa=6, pack=8, kg=18, kw=0.8, fpy=0.95),
    "2U": dict(kit=8, asm=50, fct=25, bi=6, fqa=8, pack=10, kg=30, kw=1.5, fpy=0.95),
    "GPU": dict(kit=18, asm=150, fct=45, bi=8, fqa=20, pack=25, kg=130, kw=10.2, fpy=0.90),
}

def work_content(f):
    s = STD[f]
    return s["kit"] + s["asm"] + s["fct"] + s["fqa"] + s["pack"]

# ---------------- Capacity / resources ----------------
def capacity():
    r = {}
    q = QDAY
    takt_all = T_AVAIL / sum(q.values())
    r["takt_all"] = takt_all
    # kitting
    kit_min = sum(q[f] * STD[f]["kit"] for f in FAM)
    r["kit"] = dict(load=kit_min, need=kit_min / T_EFF, n=2)
    # 1U/2U assembly: 2 family cells, each 15x1U + 10x2U
    asm12 = q["1U"] * STD["1U"]["asm"] + q["2U"] * STD["2U"]["asm"]
    cell_load = asm12 / 2
    r["asm12"] = dict(load=asm12, need=asm12 / T_EFF, cells=2, op_per_cell=3, n=6,
                      cell_load=cell_load, cell_util=cell_load / (3 * T_EFF),
                      takt_cell=T_AVAIL / 25)
    gpu_asm = q["GPU"] * STD["GPU"]["asm"]
    r["asmG"] = dict(load=gpu_asm, need=gpu_asm / T_EFF, n=5, bays=3,
                     bay_min=gpu_asm / 2, bay_need=(gpu_asm / 2) / T_AVAIL)
    fct12 = q["1U"] * STD["1U"]["fct"] + q["2U"] * STD["2U"]["fct"]
    r["fct12"] = dict(load=fct12, need=fct12 / (T_AVAIL * EQ_AVAIL), n=4)
    # burn-in slots (cycles/day by slot)
    cyc = {"1U": 3, "2U": 2, "GPU": 2}
    slots12 = (q["1U"] / cyc["1U"] + q["2U"] / cyc["2U"]) / EQ_AVAIL
    r["bi12"] = dict(raw=q["1U"] / 3 + q["2U"] / 2, need=slots12, slots=32, racks=4)
    bayG = (q["GPU"] / cyc["GPU"]) / EQ_AVAIL
    r["tstG"] = dict(raw=q["GPU"] / 2, need=bayG, bays=7)
    fqa = sum(q[f] * STD[f]["fqa"] for f in FAM)
    r["fqa"] = dict(load=fqa, need=fqa / T_EFF, n=2)
    pack12 = q["1U"] * STD["1U"]["pack"] + q["2U"] * STD["2U"]["pack"]
    packG = q["GPU"] * STD["GPU"]["pack"]
    r["pack"] = dict(load12=pack12, loadG=packG, need=(pack12 + packG) / T_EFF, n=3)
    fails = q["1U"] * (1 - STD["1U"]["fpy"]) + q["2U"] * (1 - STD["2U"]["fpy"]) + q["GPU"] * (1 - STD["GPU"]["fpy"])
    r["rep"] = dict(fails=fails, load=fails * 60, need=fails * 60 / T_EFF, n=2)
    return r

# station capacity table (units/day) vs demand for bottleneck view
def station_table():
    c = capacity()
    rows = []
    def add(name, par, ct, demand, note, avail=T_AVAIL * EQ_AVAIL):
        cap = par * avail / ct
        rows.append(dict(name=name, par=par, ct=ct, cap=cap, demand=demand, util=demand / cap, note=note))
    # use mix-weighted CT for 1U/2U
    mix_ct = lambda k: (30 * STD["1U"][k] + 20 * STD["2U"][k]) / 50
    add("Kitting 備料", 2, (sum(QDAY[f]*STD[f]["kit"] for f in FAM))/60, 60, "2 站 Pick-to-Light", T_EFF)
    add("1U/2U 組裝 (2 Cell×3 人)", 6, mix_ct("asm"), 50, "人時限制", T_EFF)
    add("GPU 組裝 (5 人)", 5, STD["GPU"]["asm"], 10, "人時限制", T_EFF)
    add("1U/2U 功能測試 FCT", 4, mix_ct("fct"), 50, "測試機台")
    # burn-in: capacity = slots*cycles
    cap_bi = 32 * EQ_AVAIL * (30 / 50 * 3 + 20 / 50 * 2)  # weighted cycles
    rows.append(dict(name="1U/2U 燒機 Burn-in", par=32, ct=(30*4+20*6)/50*60, cap=cap_bi, demand=50,
                     util=50 / cap_bi, note="32 槽×加權 2.6 循環/日"))
    cap_g = 7 * EQ_AVAIL * 2
    rows.append(dict(name="GPU 測試灣 (FCT+燒機)", par=7, ct=(8*60+45), cap=cap_g, demand=10,
                     util=10 / cap_g, note="7 灣×2 循環/日"))
    add("FQA 最終品檢", 2, sum(QDAY[f]*STD[f]["fqa"] for f in FAM)/60, 60, "2 人", T_EFF)
    add("包裝 (2+1 站)", 3, (c["pack"]["load12"] + c["pack"]["loadG"]) / 60, 60, "GPU 木箱站 1", T_EFF)
    return rows

# ---------------- Departments & space requirement ----------------
DEPTS = ["REC", "IQC", "WH", "KIT", "ASM", "GPUA", "TST", "GPUT", "REP", "FQA", "PACK", "SHIP", "OFF", "UTL"]
DNAME = {
    "REC": "收貨月台", "IQC": "進料檢驗 IQC", "WH": "原物料倉", "KIT": "備料 Kitting",
    "ASM": "1U/2U 組裝", "GPUA": "GPU 組裝", "TST": "1U/2U 測試燒機", "GPUT": "GPU 測試燒機",
    "REP": "維修 Repair", "FQA": "最終品檢 FQA", "PACK": "包裝", "SHIP": "成品/出貨月台",
    "OFF": "辦公/更衣/休息", "UTL": "機電公用/AMR 充電",
}
DSHORT = {
    "REC": "收貨", "IQC": "IQC", "WH": "倉儲", "KIT": "備料", "ASM": "1U/2U組裝", "GPUA": "GPU組裝",
    "TST": "1U/2U測試", "GPUT": "GPU測試", "REP": "維修", "FQA": "FQA", "PACK": "包裝", "SHIP": "出貨",
    "OFF": "辦公", "UTL": "機電",
}
CAT = {  # for hatch
    "REC": "log", "SHIP": "log", "WH": "sto", "KIT": "sto", "IQC": "qa", "FQA": "qa", "REP": "qa",
    "ASM": "pro", "GPUA": "pro", "TST": "tst", "GPUT": "tst", "PACK": "log", "OFF": "off", "UTL": "sup",
}
# (item, qty, unit m2, note) ; area = sum * (1+allow)
SPACE = {
    "REC": ([("卸貨月台區(含 dock leveler)", 2, 14, "2 門"), ("進貨暫存棧板位", 17, 1.2 * 2.0, "尖峰 60% 到貨，含叉取空間"),
             ("RTV/紙箱回收區", 1, 8, ""), ("收貨作業桌/掃描", 1, 4, "")], 0.15),
    "IQC": ([("檢驗工作台(ESD)", 3, 3.6, "1.8×0.9 m＋作業空間"), ("待驗暫存棧板位", 8, 2.4, ""),
             ("MRB/隔離籠", 1, 6, ""), ("量測/ESD 測試設備", 1, 4, "")], 0.50),
    "WH": ([("選擇式棧板料架(4 層)", 18, 3.0 * 2.5, "120 棧板位＝24 板/日×5 日；含 2.8 m 堆高通道"),
            ("小零件料架(DIMM/SSD/線材)", 20, 0.6 * 2.5, ""), ("高價料管制籠(CPU/GPU)", 1, 36, "門禁＋CCTV"),
            ("上架/揀貨暫存", 1, 20, ""), ("倉管站", 1, 8, "")], 0.10),
    "KIT": ([("流利架＋Pick-to-Light", 8, 2.4, "超市式備料"), ("配料工作台", 2, 3.6, ""),
             ("Kit 台車暫存(0.5 日)", 23, 1.2 * 1.5, "1U/2U 13 車＋GPU 10 車"), ("空台車緩衝", 1, 10, ""),
             ("拆箱(De-trash)區", 1, 8, "")], 0.50),
    "ASM": ([("U 型 Cell(3 站＋線邊架＋在製品)", 2, 56, "8×7 m/Cell"), ("AMR 取放點", 4, 2, ""),
             ("工具間/扭力校正", 1, 8, ""), ("組前預檢治具", 1, 6, ""), ("組長站/ESD 入口", 1, 10, "")], 0.25),
    "GPUA": ([("固定位組裝灣(升降台＋懸臂吊)", 3, 16, "4×4 m/灣"), ("GPU Tray 次組立台", 2, 4, ""),
              ("SN 管制零件籠", 1, 8, ""), ("液冷洩漏測試站", 1, 6, "G2 冷板機種"), ("機箱待組暫存", 4, 2, "")], 0.50),
    "TST": ([("FCT 測試台", 4, 5.8, "2.4×1.2 m＋作業區"), ("燒機機櫃(8 槽)", 4, 4.0, "含前後維修空間"),
             ("燒機擴充機櫃位", 1, 4.0, "預留"), ("測前/測後暫存台車", 12, 1.2, ""), ("快速除錯台", 1, 4, ""),
             ("測試伺服器/網路櫃", 2, 3, ""), ("熱通道/排風", 1, 8, "")], 0.50),
    "GPUT": ([("GPU 測試灣(FCT＋燒機，12 kW)", 7, 9, "3×3 m/灣"), ("CDU 液冷分配單元", 2, 3, ""),
              ("配電 PDU", 1, 6, ""), ("待測/測畢暫存", 4, 2, ""), ("測試控制站", 1, 4, "")], 0.40),
    "REP": ([("維修工作台(ESD)", 3, 3.6, ""), ("備品籠", 1, 6, ""), ("不良品隔離", 1, 4, ""),
             ("診斷設備", 1, 6, ""), ("待修台車", 6, 1.2, "")], 0.50),
    "FQA": ([("FQA/OBA 檢驗台", 2, 3.6, ""), ("OBA 拆箱覆檢台", 1, 4, ""), ("Hold/MRB 區", 1, 6, ""),
             ("待檢/檢畢暫存", 8, 1.2, "")], 0.50),
    "PACK": ([("1U/2U 包裝站", 2, 8, "封箱＋貼標"), ("GPU 木箱包裝站(吊具)", 1, 16, ""),
              ("包材暫存(1 日)", 1, 30, ""), ("打帶/纏膜", 1, 8, ""), ("待包暫存", 8, 1.5, "")], 0.30),
    "SHIP": ([("出貨月台(含 dock leveler)", 2, 14, "2 門"), ("成品棧板暫存(1.5 日)", 27, 1.2 * 1.6, "18 板/日×1.5"),
              ("出貨文件/稽核站", 1, 6, "")], 0.15),
    "OFF": ([("工程/生管辦公(12 人)", 12, 6, "ME/PE/IE/PC/QA"), ("會議室", 1, 16, ""), ("更衣室/置物櫃", 1, 30, "45 人"),
             ("ESD 檢測入口", 1, 10, ""), ("休息/茶水", 1, 30, "")], 0.15),
    "UTL": ([("配電盤/變電設備", 1, 20, "測試負載約 120 kW"), ("UPS 室", 1, 12, ""), ("空壓機", 1, 8, ""),
             ("冷卻水/CDU 介面", 1, 10, ""), ("MES/網路機房", 1, 8, ""), ("AMR 充電站", 4, 3, ""),
             ("空容器/空棧板緩衝", 1, 20, "")], 0.30),
}

def space_req():
    out = {}
    for d in DEPTS:
        items, allow = SPACE[d]
        base = sum(q * a for _, q, a, _ in items)
        out[d] = dict(base=base, allow=allow, req=base * (1 + allow))
    return out

# ---------------- Flows (From-To) ----------------
HU = {  # handling unit: factor, name, mode
    "PAL": (2.0, "棧板", "電動拖板車"),
    "TOTE": (1.0, "料箱/Kit 車", "AMR"),
    "RACK": (1.5, "機架台車(4 台)", "AMR"),
    "GPU": (3.0, "GPU 單機升降台車", "手動升降台車"),
    "SMALL": (0.5, "小零件箱", "手推車"),
    "CART": (1.0, "單機手推車", "手推車"),
}
# (from, to, trips/day, HU, description)
ROUTES = [
    ("REC", "IQC", 24, "PAL", "零組件進料"),
    ("IQC", "WH", 23, "PAL", "允收入庫"),
    ("IQC", "REC", 1, "PAL", "拒收退貨 RTV"),
    ("REC", "PACK", 4, "PAL", "包材進料"),
    ("WH", "KIT", 30, "TOTE", "零件補貨(料箱)"),
    ("WH", "KIT", 8, "PAL", "機箱補貨(棧板)"),
    ("KIT", "ASM", 25, "TOTE", "1U/2U Kit(2 套/車)"),
    ("KIT", "GPUA", 10, "TOTE", "GPU 高價料 Kit(SN)"),
    ("KIT", "GPUA", 10, "RACK", "GPU 機箱台車"),
    ("KIT", "REC", 4, "TOTE", "拆箱紙箱回收"),
    ("ASM", "TST", 13, "RACK", "組畢待測"),
    ("TST", "REP", 3, "CART", "測試不良"),
    ("REP", "TST", 3, "CART", "修復重測"),
    ("TST", "FQA", 13, "RACK", "測畢待檢"),
    ("GPUA", "GPUT", 10, "GPU", "GPU 組畢待測"),
    ("GPUT", "REP", 1, "GPU", "GPU 測試不良"),
    ("REP", "GPUT", 1, "GPU", "GPU 修復重測"),
    ("GPUT", "FQA", 10, "GPU", "GPU 測畢待檢"),
    ("FQA", "REP", 1, "CART", "FQA 不良"),
    ("FQA", "PACK", 13, "RACK", "1U/2U 允收"),
    ("FQA", "PACK", 10, "GPU", "GPU 允收"),
    ("PACK", "SHIP", 18, "PAL", "成品棧板"),
    ("REP", "WH", 2, "SMALL", "不良料退庫"),
    ("WH", "REP", 2, "SMALL", "維修備品"),
]

def fromto(routes=ROUTES):
    n = len(DEPTS)
    M = np.zeros((n, n)); T = np.zeros((n, n))
    for a, b, t, hu, _ in routes:
        i, j = DEPTS.index(a), DEPTS.index(b)
        M[i, j] += t * HU[hu][0]; T[i, j] += t
    return M, T

# ---------------- REL chart ----------------
REASON = {
    1: "物流量大/流程順序", 2: "共用人員或督導", 3: "共用公用設施(電力/冷卻/網路)",
    4: "品質回饋(不良即時處理)", 5: "文件資訊往來/管理", 6: "高價料保全與 SN 管制",
    7: "共用月台/卡車場", 8: "噪音/熱(需隔離)", 9: "安全(重物/高壓電/消防)",
}
# qualitative (non-flow) ratings
QUAL = {
    ("TST", "REP"): ("A", [4, 2]),
    ("GPUT", "REP"): ("E", [4]),
    ("GPUA", "GPUT"): ("A", [1, 9]),
    ("PACK", "SHIP"): ("A", [1, 7]),
    ("ASM", "TST"): ("E", [1, 4]),
    ("GPUT", "UTL"): ("E", [3]),
    ("TST", "UTL"): ("I", [3]),
    ("GPUA", "UTL"): ("O", [3]),
    ("ASM", "UTL"): ("O", [3]),
    ("REC", "SHIP"): ("I", [7, 2]),
    ("ASM", "GPUA"): ("I", [2]),
    ("TST", "GPUT"): ("I", [2, 3]),
    ("WH", "GPUA"): ("O", [6]),
    ("OFF", "ASM"): ("I", [5, 2]),
    ("OFF", "GPUA"): ("O", [5]),
    ("OFF", "FQA"): ("O", [5]),
    ("OFF", "IQC"): ("O", [5]),
    ("OFF", "REC"): ("O", [5]),
    ("OFF", "WH"): ("O", [5]),
    ("IQC", "FQA"): ("O", [2]),
    ("REP", "ASM"): ("O", [4]),
    ("IQC", "REP"): ("O", [4]),
    ("OFF", "TST"): ("X", [8]),
    ("OFF", "GPUT"): ("X", [8, 9]),
    ("OFF", "UTL"): ("X", [8, 9]),
}
GRADES = "AEIOUX"
GVAL = {"A": 4, "E": 3, "I": 2, "O": 1, "U": 0, "X": -4}
FLOW_TH = [(40, "A"), (25, "E"), (10, "I"), (0.01, "O")]

def flow_grade(v):
    for th, g in FLOW_TH:
        if v >= th:
            return g
    return "U"

def rel_chart():
    M, _ = fromto()
    rel = {}
    for i, j in itertools.combinations(range(len(DEPTS)), 2):
        a, b = DEPTS[i], DEPTS[j]
        v = M[i, j] + M[j, i]
        fg = flow_grade(v)
        q = QUAL.get((a, b)) or QUAL.get((b, a))
        reasons = [1] if fg != "U" else []
        g = fg
        if q:
            qg, qr = q
            if qg == "X":
                g = "X"; reasons = qr
            elif GRADES.index(qg) < GRADES.index(fg):
                g = qg; reasons = sorted(set(reasons + qr))
            else:
                reasons = sorted(set(reasons + qr))
        rel[(a, b)] = dict(flow=v, fg=fg, g=g, reasons=reasons)
    return rel

def rel_get(rel, a, b):
    return rel.get((a, b)) or rel.get((b, a))

# ---------------- Layouts ----------------
# rect = (x0, y0, x1, y1)
L = {}
L["A"] = dict(
    name="Layout A 部門本位功能式",
    depts={
        "REC": (0, 30, 12, 40), "UTL": (0, 22, 12, 30), "WH": (12, 28, 29, 40), "KIT": (12, 22, 29, 28),
        "OFF": (0, 0, 16, 11), "FQA": (0, 11, 8, 19), "REP": (8, 11, 16, 19), "IQC": (16, 11, 29, 19),
        "TST": (16, 0, 29, 11),
        "PACK": (32, 0, 46, 8), "SHIP": (46, 0, 60, 8), "ASM": (32, 8, 46, 19), "GPUA": (46, 8, 60, 19),
        "GPUT": (46, 22, 60, 32),
    },
    reserve={"R1": (32, 22, 46, 40), "R2": (46, 32, 60, 40)},
    aisles=[(0, 19, 60, 22), (29, 0, 32, 19), (29, 22, 32, 40)],
    docks={"REC": "W", "SHIP": "S"},
)
L["B"] = dict(
    name="Layout B 直線 I 型流",
    depts={
        "REC": (0, 0, 6, 18), "IQC": (6, 0, 10, 18), "WH": (10, 0, 26, 18), "KIT": (26, 0, 34, 18),
        "ASM": (34, 0, 45, 18), "TST": (45, 0, 52, 18), "PACK": (52, 0, 60, 18),
        "OFF": (0, 21, 10, 40), "UTL": (10, 21, 16, 40), "GPUA": (30, 21, 38, 40), "GPUT": (38, 21, 45, 40),
        "FQA": (45, 21, 52, 30), "REP": (45, 30, 52, 40), "SHIP": (52, 21, 60, 33),
    },
    reserve={"R1": (16, 21, 30, 40), "R2": (52, 33, 60, 40)},
    aisles=[(0, 18, 60, 21)],
    docks={"REC": "W", "SHIP": "E"},
)
L["C"] = dict(
    name="Layout C U 型流＋產品族群單元",
    depts={
        "REC": (0, 0, 16, 6), "IQC": (0, 6, 16, 10), "WH": (0, 10, 16, 28), "OFF": (0, 28, 16, 40),
        "KIT": (19, 21, 27, 40), "GPUA": (27, 21, 36, 40), "GPUT": (36, 21, 44, 40), "UTL": (47, 30, 60, 40),
        "ASM": (19, 0, 30, 18), "REP": (30, 0, 40, 6), "TST": (30, 6, 40, 18), "FQA": (40, 6, 44, 18),
        "SHIP": (47, 0, 60, 8), "PACK": (47, 8, 60, 18),
    },
    reserve={"R1": (47, 21, 60, 30), "R2": (40, 0, 44, 6)},
    aisles=[(16, 0, 19, 40), (44, 0, 47, 40), (19, 18, 44, 21), (47, 18, 60, 21)],
    docks={"REC": "S", "SHIP": "S"},
)
L["F"] = dict(
    name="最終佈置 Final（C 修正版）",
    depts={
        "REC": (0, 0, 16, 6), "IQC": (0, 6, 16, 10), "WH": (0, 10, 16, 28), "OFF": (0, 28, 16, 40),
        "KIT": (19, 22, 27, 40), "GPUA": (27, 22, 36, 40), "GPUT": (36, 22, 44, 40), "UTL": (48, 30, 60, 40),
        "ASM": (19, 0, 30, 18), "REP": (30, 0, 40, 6), "TST": (30, 6, 40, 18), "FQA": (40, 6, 44, 18),
        "SHIP": (48, 0, 60, 8), "PACK": (48, 8, 60, 18),
    },
    reserve={"R1": (48, 22, 60, 30), "R2": (40, 0, 44, 6)},
    aisles=[(16, 0, 19, 40), (19, 18, 60, 22), (44, 0, 48, 18), (44, 22, 48, 40)],
    docks={"REC": "S", "SHIP": "S"},
)
_walk = set()
for y in range(0, 40):
    _walk.add((16, y))            # 西側通道西緣人行道
for x in range(16, 60):
    _walk.add((x, 21))            # 中央主通道北緣人行道
for y in range(0, 40):
    _walk.add((44, y))            # 東側主通道西緣人行道
L["F"]["walk"] = _walk
SITE = {  # 廠區條件：南側卡車道路、西側員工主入口與停車場、北側為擴建預留地
    "truck": "S", "people": "W", "expand": "N",
}
# 擴產情境：GPU 需求 +100%、1U/2U +20%（需增加之面積 m2）
EXPANSION_NEED = {"GPUT": 110, "GPUA": 70, "TST": 30, "ASM": 56}


GROWTH = ["GPUA", "GPUT", "TST", "ASM"]

def final_routes():
    r = []
    for a, b, t, hu, d in ROUTES:
        if (a, b) == ("GPUT", "FQA"):
            continue  # GPU FQA 於測試灣原位執行
        if (a, b, hu) == ("FQA", "PACK", "GPU"):
            r.append(("GPUT", "PACK", t, hu, "GPU 原位 FQA 後直送包裝")); continue
        if (a, b) in [("GPUT", "REP"), ("REP", "GPUT")]:
            continue  # GPU 原位除錯
        if (a, b) == ("REC", "PACK"):
            r.append(("SHIP", "PACK", t, hu, "包材改由出貨月台離峰進貨")); continue
        r.append((a, b, t, hu, d))
    r.append(("WH", "GPUT", 1, "SMALL", "GPU 原位除錯備品"))
    r.append(("GPUT", "WH", 1, "SMALL", "GPU 不良料退庫"))
    return r

# ---------------- geometry helpers ----------------
def area(r):
    return (r[2] - r[0]) * (r[3] - r[1])

def centroid(r):
    return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)

def grid(lay):
    G = np.full((W, H), "", dtype=object)
    for k, r in lay["depts"].items():
        G[r[0]:r[2], r[1]:r[3]] = k
    for k, r in lay["reserve"].items():
        G[r[0]:r[2], r[1]:r[3]] = k
    for r in lay["aisles"]:
        G[r[0]:r[2], r[1]:r[3]] = "AISLE"
    return G

def check(lay):
    tot = sum(area(r) for r in lay["depts"].values()) + sum(area(r) for r in lay["reserve"].values())
    G = np.zeros((W, H), int)
    for r in list(lay["depts"].values()) + list(lay["reserve"].values()):
        G[r[0]:r[2], r[1]:r[3]] += 1
    for r in lay["aisles"]:
        G[r[0]:r[2], r[1]:r[3]] += 1
    # aisles may overlap each other -> count unique
    A = np.zeros((W, H), bool)
    for r in lay["aisles"]:
        A[r[0]:r[2], r[1]:r[3]] = True
    D = np.zeros((W, H), int)
    for r in list(lay["depts"].values()) + list(lay["reserve"].values()):
        D[r[0]:r[2], r[1]:r[3]] += 1
    assert D.max() <= 1, "dept overlap"
    assert not (A & (D > 0)).any(), "aisle overlaps dept"
    assert ((A) | (D > 0)).all(), "uncovered cells %d" % (~(A | (D > 0))).sum()
    return tot, A.sum()

def pt_cell(p):
    return (min(int(p[0]), W - 1), min(int(p[1]), H - 1))

def shortest(lay, a, b, G=None, mode="veh"):
    """Dijkstra on 1 m grid from centroid of a to centroid of b.
    aisle/src/dst cost 1; reserve 1.5; other depts 6 (pass-through)."""
    if G is None:
        G = grid(lay)
    s = pt_cell(centroid(lay["depts"][a])); t = pt_cell(centroid(lay["depts"][b]))
    walk = lay.get("walk")  # set of walkway cells (final)
    def cost(c):
        v = G[c]
        if mode == "ped" and walk is not None:
            if c in walk or v in (a, b) or v == "OFF":
                return 1.0
            if v == "AISLE":
                return 4.0  # crossing vehicle lane
        if mode == "veh" and walk is not None and c in walk:
            return 4.0
        if v in (a, b) or v == "AISLE":
            return 1.0
        if v.startswith("R"):
            return 1.5
        return 6.0
    dist = {s: 0}; prev = {}
    pq = [(0, s)]
    while pq:
        d, c = heapq.heappop(pq)
        if c == t:
            break
        if d > dist[c]:
            continue
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < W and 0 <= n[1] < H:
                nd = d + cost(n)
                if nd < dist.get(n, 1e18):
                    dist[n] = nd; prev[n] = c; heapq.heappush(pq, (nd, n))
    path = [t]
    while path[-1] != s:
        path.append(prev[path[-1]])
    path = path[::-1]
    through = sorted({G[c] for c in path if G[c] not in (a, b, "AISLE", "") and not G[c].startswith("R")})
    return len(path) - 1, path, through

def adjacent(lay, a, b, gap=4.0, minov=2.0):
    r1, r2 = lay["depts"][a], lay["depts"][b]
    # overlap in x and edge distance in y, or vice versa
    ox = min(r1[2], r2[2]) - max(r1[0], r2[0])
    oy = min(r1[3], r2[3]) - max(r1[1], r2[1])
    dy = max(r2[1] - r1[3], r1[1] - r2[3])
    dx = max(r2[0] - r1[2], r1[0] - r2[2])
    if ox >= minov and 0 <= dy <= gap:
        return True
    if oy >= minov and 0 <= dx <= gap:
        return True
    return False

def edge_dist(lay, a, b):
    r1, r2 = lay["depts"][a], lay["depts"][b]
    dx = max(0, r2[0] - r1[2], r1[0] - r2[2]); dy = max(0, r2[1] - r1[3], r1[1] - r2[3])
    return math.hypot(dx, dy)

# ---------------- evaluation ----------------
AMR_V = 48.0      # m/min (0.8 m/s 含路口減速)
AMR_HANDLE = 2.0  # min/趟 對接+掃描
WALK_V = 40.0     # m/min 推車行走
MAN_HANDLE = 3.0  # min/趟
AMR_AVAIL = T_AVAIL * 0.85

HEADCOUNT = {"REC": 1, "IQC": 2, "WH": 2, "KIT": 2, "ASM": 6, "GPUA": 5, "TST": 3, "GPUT": 2,
             "REP": 2, "FQA": 2, "PACK": 3, "SHIP": 1, "UTL": 1}
ENG_VISIT = {"ASM": 10, "GPUA": 10, "TST": 8, "GPUT": 8, "REP": 4, "IQC": 3, "FQA": 3}
PED_TRIPS_PER_HEAD = 6

def evaluate(key, routes=None):
    lay = L[key]
    if routes is None:
        routes = final_routes() if key == "F" else ROUTES
    G = grid(lay)
    res = dict(key=key)
    tot, aisle = check(lay)
    res["aisle_area"] = int(aisle)
    res["reserve_area"] = sum(area(r) for r in lay["reserve"].values())
    res["dept_area"] = sum(area(r) for r in lay["depts"].values())
    # route distances
    rt = []; LD = 0; trips = 0; vtraffic = np.zeros((W, H)); passthru = []
    amr_min = 0; man_min = 0; gpu_heavy = 0
    for a, b, t, hu, d in routes:
        dist, path, thr = shortest(lay, a, b, G)
        f = HU[hu][0]
        ld = dist * t * f
        rt.append(dict(a=a, b=b, t=t, hu=hu, d=d, dist=dist, ehu=t * f, ld=ld, thr=thr))
        LD += ld; trips += t
        for c in path:
            vtraffic[c] += t
        if thr:
            passthru.append((a, b, t, thr))
        mode = HU[hu][2]
        if mode == "AMR":
            amr_min += t * (2 * dist / AMR_V + AMR_HANDLE)
        else:
            man_min += t * (2 * dist / WALK_V + MAN_HANDLE)
        if hu == "GPU":
            gpu_heavy += t * dist
    res["routes"] = rt; res["LD"] = LD; res["trips"] = trips
    res["avg_dist"] = sum(r["dist"] * r["t"] for r in rt) / trips
    res["passthru"] = passthru
    res["passthru_trips"] = sum(p[2] for p in passthru)
    res["amr_min"] = amr_min; res["amr_n"] = amr_min / AMR_AVAIL
    res["man_min"] = man_min
    res["gpu_heavy_m"] = gpu_heavy
    # people flows
    ptraffic = np.zeros((W, H))
    for d, h in HEADCOUNT.items():
        dist, path, _ = shortest(lay, "OFF", d, G, mode="ped")
        n = h * PED_TRIPS_PER_HEAD + ENG_VISIT.get(d, 0) * 2
        for c in path:
            ptraffic[c] += n
    aisle_mask = (G == "AISLE")
    conflict = (ptraffic * vtraffic)[aisle_mask].sum() / 1000.0
    res["conflict"] = conflict
    res["conflict_cells"] = int(((ptraffic > 0) & (vtraffic > 0) & aisle_mask).sum())
    res["vtraffic"] = vtraffic; res["ptraffic"] = ptraffic
    # REL satisfaction
    rel = rel_chart()
    sat = {}; score = 0; xv = []
    for (a, b), v in rel.items():
        g = v["g"]
        adj = adjacent(lay, a, b)
        if g in "AE":
            sat.setdefault(g, [0, 0]); sat[g][1] += 1
            if adj:
                sat[g][0] += 1
        if g == "X":
            ed = edge_dist(lay, a, b)
            if ed < 10:
                xv.append((a, b, round(ed, 1)))
        if adj:
            score += GVAL[g]
    res["rel_sat"] = sat; res["rel_score"] = score; res["x_viol"] = xv
    # closeness-weighted distance (A,E,I only), rectilinear centroid
    cwd = 0
    for (a, b), v in rel.items():
        if v["g"] in "AEI":
            c1, c2 = centroid(lay["depts"][a]), centroid(lay["depts"][b])
            cwd += GVAL[v["g"]] * (abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]))
    res["cwd"] = cwd
    # shape factor: perimeter / (4 sqrt(A)) ; 1 = square
    sf = []
    for k, r in lay["depts"].items():
        w_, h_ = r[2] - r[0], r[3] - r[1]
        sf.append(max(w_, h_) / min(w_, h_))
    res["aspect_avg"] = float(np.mean(sf)); res["aspect_max"] = float(max(sf))
    res["aspect_bad"] = sum(1 for s in sf if s > 3)
    # space fit vs requirement
    req = space_req()
    fit = {k: area(lay["depts"][k]) / req[k]["req"] for k in DEPTS}
    res["fit"] = fit; res["short"] = [k for k, v in fit.items() if v < 0.95]
    # reserve adjacency to growth depts
    planned = 0; dead = 0
    for rk, rr in lay["reserve"].items():
        tmp = dict(lay); tmp = dict(depts=dict(lay["depts"], **{rk: rr}))
        if any(adjacent(tmp, rk, g) for g in GROWTH):
            planned += area(rr)
        else:
            dead += area(rr)
    res["reserve_planned"] = planned; res["reserve_dead"] = dead
    res["util_eff"] = (res["dept_area"] + aisle + planned) / (W * H)
    # expandable wall adjacent to growth depts (east / north walls without docks)
    ext = 0
    for g in ["GPUA", "GPUT"]:
        r = lay["depts"][g]
        if r[2] == W and lay["docks"].get("SHIP") != "E":
            ext += r[3] - r[1]
        if r[3] == H:
            ext += r[2] - r[0]
    res["gpu_ext_wall"] = ext
    # site-level: truck docks on the people side?
    res["dock_conflict"] = sum(1 for v in lay["docks"].values() if v == SITE["people"])
    res["dock_sides"] = len(set(lay["docks"].values()))
    # expansion scenario: satisfy growth need from own surplus + adjacent reserve
    remaining = {rk: area(rr) for rk, rr in lay["reserve"].items()}
    got = {}
    for g in ["GPUT", "GPUA", "TST", "ASM"]:
        need = EXPANSION_NEED[g]
        own = max(0.0, area(lay["depts"][g]) - req[g]["req"])
        take = min(need, own); need -= take
        for rk, rr in lay["reserve"].items():
            if need <= 0:
                break
            tmp = dict(depts=dict(lay["depts"], **{rk: rr}))
            if adjacent(tmp, rk, g) and remaining[rk] > 0:
                t2 = min(need, remaining[rk]); remaining[rk] -= t2; need -= t2; take += t2
        got[g] = take
    res["exp_got"] = got
    res["exp_ratio"] = sum(got.values()) / sum(EXPANSION_NEED.values())
    # GPU zone on expansion wall (north) length, only if GPUA-GPUT contiguous
    ln = 0
    if adjacent(lay, "GPUA", "GPUT", gap=0.0):
        for g in ["GPUA", "GPUT"]:
            r = lay["depts"][g]
            if r[3] == H:
                ln += r[2] - r[0]
    res["gpu_north_wall"] = ln
    return res

if __name__ == "__main__":
    c = capacity()
    for k, v in c.items():
        print(k, v)
    print()
    for r in station_table():
        print(r)
    sr = space_req()
    print("space req", {k: round(v["req"]) for k, v in sr.items()}, round(sum(v["req"] for v in sr.values())))
    M, T = fromto()
    print("EHU total", M.sum(), "trips", T.sum())
    rel = rel_chart()
    from collections import Counter
    print(Counter(v["g"] for v in rel.values()))
    for k in "ABCF":
        r = evaluate(k)
        print("=====", k, L[k]["name"])
        print(" LD %.0f  trips %d avg %.1f  AMR %.2f  man_min %.0f gpu_heavy %.0f" % (r["LD"], r["trips"], r["avg_dist"], r["amr_n"], r["man_min"], r["gpu_heavy_m"]))
        print(" pass", r["passthru"], r["passthru_trips"])
        print(" conflict %.1f cells %d" % (r["conflict"], r["conflict_cells"]))
        print(" rel", r["rel_sat"], r["rel_score"], "X", r["x_viol"], "cwd", r["cwd"])
        print(" aspect %.2f max %.1f bad %d" % (r["aspect_avg"], r["aspect_max"], r["aspect_bad"]))
        print(" short", r["short"], {k: round(v, 2) for k, v in r["fit"].items()})
        print(" aisle", r["aisle_area"], "reserve planned", r["reserve_planned"], "dead", r["reserve_dead"], "util_eff %.3f" % r["util_eff"], "ext", r["gpu_ext_wall"])

# ---------------- family path & scoring ----------------
FAMILY_PATH = {
    "1U/2U": ["REC", "IQC", "WH", "KIT", "ASM", "TST", "FQA", "PACK", "SHIP"],
    "GPU": ["REC", "IQC", "WH", "KIT", "GPUA", "GPUT", "FQA", "PACK", "SHIP"],
}
FAMILY_PATH_F = {
    "1U/2U": FAMILY_PATH["1U/2U"],
    "GPU": ["REC", "IQC", "WH", "KIT", "GPUA", "GPUT", "PACK", "SHIP"],
}

def family_len(key):
    lay = L[key]; G = grid(lay)
    fp = FAMILY_PATH_F if key == "F" else FAMILY_PATH
    return {f: sum(shortest(lay, a, b, G)[0] for a, b in zip(p, p[1:])) for f, p in fp.items()}

WEIGHTS = {"dist": 0.30, "safe": 0.20, "flex": 0.20, "exp": 0.15, "space": 0.15}
CRIT_NAME = {"dist": "物流距離", "space": "空間利用率", "flex": "生產彈性", "safe": "人流物流安全", "exp": "未來擴充性"}
# 生產彈性評分(1-10)：依表列證據評定
FLEX = {
    "A": {"mix": 7, "step": 7, "iso": 5},
    "B": {"mix": 4, "step": 4, "iso": 3},
    "C": {"mix": 9, "step": 9, "iso": 9},
}

def score_all(keys=("A", "B", "C")):
    req = space_req(); sreq = sum(v["req"] for v in req.values())
    ev = {k: evaluate(k) for k in keys}
    ind = {}
    for k, r in ev.items():
        lay = L[k]
        short_m2 = sum(max(0, req[d]["req"] - area(lay["depts"][d])) for d in DEPTS)
        met = sum(1 for d in DEPTS if area(lay["depts"][d]) >= 0.95 * req[d]["req"]) / len(DEPTS)
        over = sum(max(0, area(lay["depts"][d]) - 1.3 * req[d]["req"]) for d in DEPTS if d not in GROWTH)
        waste = r["reserve_dead"] + over
        good_shape = sum(1 for d in DEPTS if max((lay["depts"][d][2] - lay["depts"][d][0]) / (lay["depts"][d][3] - lay["depts"][d][1]),
                                                (lay["depts"][d][3] - lay["depts"][d][1]) / (lay["depts"][d][2] - lay["depts"][d][0])) <= 3) / len(DEPTS)
        exp_net = max(0.0, sum(r["exp_got"].values()) - short_m2) / sum(EXPANSION_NEED.values())
        ind[k] = dict(LD=r["LD"], avg=r["avg_dist"], short_m2=short_m2, met=met, waste=waste,
                      useful=1 - waste / (W * H), shape=good_shape, conflict=r["conflict"],
                      passthru=r["passthru_trips"], gpu=r["gpu_heavy_m"],
                      xdock=len(r["x_viol"]) + r["dock_conflict"], exp_net=exp_net,
                      wall=r["gpu_north_wall"], flex=FLEX[k])
    mn = lambda f: min(ind[k][f] for k in keys)
    mx = lambda f: max(ind[k][f] for k in keys)
    sc = {}
    for k in keys:
        i = ind[k]
        s = {}
        s["dist"] = 10 * mn("LD") / i["LD"]
        s["space_sub"] = dict(met=10 * i["met"], useful=10 * i["useful"], shape=10 * i["shape"])
        s["space"] = sum(s["space_sub"].values()) / 3
        s["safe_sub"] = dict(conflict=10 * mn("conflict") / i["conflict"],
                             passthru=10 * (1 - i["passthru"] / 60),
                             gpu=10 * mn("gpu") / i["gpu"],
                             xdock=max(0, 10 - 4 * i["xdock"]))
        s["safe"] = sum(s["safe_sub"].values()) / 4
        s["flex_sub"] = i["flex"]; s["flex"] = sum(i["flex"].values()) / 3
        s["exp_sub"] = dict(net=10 * i["exp_net"] / mx("exp_net"), wall=10 * i["wall"] / mx("wall") if mx("wall") else 0)
        s["exp"] = sum(s["exp_sub"].values()) / 2
        s["total"] = sum(WEIGHTS[c] * s[c] for c in WEIGHTS)
        sc[k] = s
    return ind, sc

def sensitivity(sc):
    sets = {
        "基準權重": WEIGHTS,
        "五項等權": {c: 0.2 for c in WEIGHTS},
        "物流距離 50%": {"dist": 0.5, "safe": 0.125, "flex": 0.125, "exp": 0.125, "space": 0.125},
        "安全 40%": {"dist": 0.15, "safe": 0.40, "flex": 0.15, "exp": 0.15, "space": 0.15},
        "擴充 40%": {"dist": 0.15, "safe": 0.15, "flex": 0.15, "exp": 0.40, "space": 0.15},
    }
    out = {}
    for n, w in sets.items():
        out[n] = {k: sum(w[c] * sc[k][c] for c in w) for k in sc}
    return out
