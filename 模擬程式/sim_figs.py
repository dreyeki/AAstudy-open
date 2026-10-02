# -*- coding: utf-8 -*-
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["hatch.linewidth"] = 0.5
OUT = "sfig"; os.makedirs(OUT, exist_ok=True)
R = {}
for p in (1, 2, 3):
    R.update(json.load(open(f"exp{p}.json")))

def save(fig, n):
    fig.savefig(os.path.join(OUT, n), dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)

def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

M = lambda k, m: R[k][m]["mean"]
CI = lambda k, m: R[k][m]["ci"]

# ---------- S1 model logic
def s1():
    fig, ax = plt.subplots(figsize=(11, 5.4)); ax.axis("off"); ax.set_xlim(0, 112); ax.set_ylim(-2, 54)
    def box(x, y, w, h, t, fc="white", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25", fc=fc, ec="black", lw=1, ls=ls))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.6)
    def arr(x1, y1, x2, y2, ls="-", lab=None, dy=1.4, dx=0):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, ls=ls, color="black"))
        if lab:
            ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, lab, ha="center", fontsize=6.4, color="0.15",
                    bbox=dict(fc="white", ec="none", pad=0.2))
    box(0.5, 22, 11, 8, "每日投料\n60 台\n（循環排序）", "0.9")
    box(15, 22, 9, 8, "Kitting\n人員×2")
    box(28, 8, 12, 8, "1U/2U 組裝\n（依方案型態）")
    box(46, 8, 9, 8, "FCT\n測試機×4")
    box(59, 8, 13, 8, "燒機槽位×32\n（機台時間可跨夜）", "0.85")
    box(77, 8, 8, 8, "FQA\n×2")
    box(28, 38, 12, 8, "GPU 組裝\n2 組×2 人")
    box(47, 38, 17, 8, "GPU 測試灣×7\nFCT 45 分＋燒機 8 h", "0.85")
    box(69, 38, 14, 8, "原位除錯／FQA\n（Final）", ls="--")
    box(90, 22, 9, 8, "包裝\n2＋1 站")
    box(102.5, 22, 9, 8, "出貨\n（完工）", "0.9")
    box(59.5, -1.5, 12, 6, "維修×2\n不良 5%／10%", ls="--")
    arr(11.5, 26, 15, 26)
    arr(24, 24, 28, 13, lab="AMR Kit 車", dx=-4.5, dy=0)
    arr(24, 28, 28, 41, lab="AMR 料箱＋機箱", dx=-5.5, dy=0)
    arr(40, 12, 46, 12, lab="AMR×4 台", dy=5.2)
    arr(55, 12, 59, 12)
    arr(72, 12, 77, 12, lab="AMR×4 台", dy=5.2)
    arr(85, 13, 90, 24, lab="AMR×4 台", dx=4, dy=-1)
    arr(40, 42, 47, 42, lab="升降台車(2 人)", dy=5.2)
    arr(64, 42, 69, 42)
    arr(83, 40, 91, 30, lab="升降台車(2 人)", dx=5, dy=1)
    arr(99, 26, 102.5, 26)
    arr(65.5, 8, 65.5, 4.5, ls=":"); arr(59.5, 2, 50.5, 8, ls=":", lab="修後重測", dx=-3.5, dy=-1.2)
    ax.text(0.5, 53, "資源約束：人工作業須在工作時窗內開始並完成；燒機與 GPU 測試為機台時間，可跨夜；裝卸與判讀需測試技術員"
                     "（單班時窗 450 分；加開中班 900 分）。\n背景物流：進料、入庫、補料、紙箱回收等每日約 100 趟，隨機發生並與產品搬運共用 AMR 與搬運人員。",
            fontsize=7.6, va="top")
    save(fig, "s01_logic.png")

# ---------- S2 timelines
def s2():
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.6), sharex=True)
    for j, key in enumerate(["E1_F_1shift", "E1_F_2shift"]):
        for i, (tl, cap, nm) in enumerate([("slot_tl", 32, "1U/2U 燒機槽位（32）"), ("bay_tl", 7, "GPU 測試灣（7）")]):
            ax = axes[i, j]
            T = np.array(R[key][tl], float)
            t = T[:, 0] / 60; run = T[:, 1]; wait = T[:, 2]
            ax.fill_between(t, 0, run, step="post", color="0.35", lw=0)
            ax.fill_between(t, run, run + wait, step="post", facecolor="white", edgecolor="black", hatch="////", lw=0.3)
            ax.axhline(cap, color="black", ls="--", lw=0.8)
            win = 450 if j == 0 else 900
            ax.axvspan(win / 60, 24, color="0.93", zorder=0)
            ax.set_ylim(0, cap * 1.15); ax.set_xlim(0, 24); clean(ax)
            ax.set_title(("單班（僅日班裝卸）" if j == 0 else "加開中班（裝卸時窗 15 h）") + "｜" + nm, fontsize=9)
            if j == 0:
                ax.set_ylabel("佔用數", fontsize=8.5)
            if i == 1:
                ax.set_xlabel("時刻（小時，0＝上班）", fontsize=8.5)
            ax.set_xticks(range(0, 25, 4))
    hs = [Rectangle((0, 0), 1, 1, fc="0.35"), Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch="////"),
          Rectangle((0, 0), 1, 1, fc="0.93"), Line2D([0], [0], color="black", ls="--")]
    fig.legend(hs, ["運轉中", "已完成、等待人員卸載（佔位閒置）", "無人裝卸時段", "容量"], loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "s02_timeline.png")

# ---------- S3 sweeps
def s3():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ax = axes[0]
    bs = list(range(5, 13))
    for w, st_, mk in [(450, "--", "o"), (900, "-", "s")]:
        y = [M(f"E2_bay_{w}_{b}", "thr_GPU") for b in bs]; e = [CI(f"E2_bay_{w}_{b}", "thr_GPU") for b in bs]
        ax.errorbar(bs, y, yerr=e, color="black", ls=st_, marker=mk, ms=5, mfc="white" if w == 450 else "black", capsize=2, lw=1.2,
                    label="單班" if w == 450 else "加開中班")
    ax.axhline(10, color="0.4", lw=0.8, ls=":"); ax.text(5.1, 10.15, "需求 10 台/日", fontsize=7.5, color="0.3")
    ax.axvline(7, color="0.6", lw=0.8); ax.text(7.1, 4.3, "主報告設計 7 灣", fontsize=7.5, color="0.3")
    ax.set_xlabel("GPU 測試灣數", fontsize=8.5); ax.set_ylabel("GPU 日產出（台/日）", fontsize=8.5)
    ax.set_ylim(4, 11); ax.legend(fontsize=8, frameon=False, loc="lower right"); clean(ax)
    ax.set_title("(a) GPU 測試灣數 vs. 產出", fontsize=9.5)
    ax = axes[1]
    ss = [16, 20, 24, 28, 32]
    for w, st_, mk in [(450, "--", "o"), (900, "-", "s")]:
        y = [M(f"E2_slot_{w}_{s}", "thr_1U") + M(f"E2_slot_{w}_{s}", "thr_2U") for s in ss]
        ax.plot(ss, y, color="black", ls=st_, marker=mk, ms=5, mfc="white" if w == 450 else "black", lw=1.2, label="單班" if w == 450 else "加開中班")
    ax.axhline(50, color="0.4", lw=0.8, ls=":"); ax.text(16.2, 50.4, "需求 50 台/日", fontsize=7.5, color="0.3")
    ax.axvline(32, color="0.6", lw=0.8); ax.text(29.2, 33.5, "主報告\n設計 32 槽", fontsize=7.5, color="0.3")
    ax.set_xlabel("1U/2U 燒機槽位數", fontsize=8.5); ax.set_ylabel("1U/2U 日產出（台/日）", fontsize=8.5)
    ax.set_xticks(ss); ax.set_ylim(30, 53); ax.legend(fontsize=8, frameon=False, loc="center right"); clean(ax)
    ax.set_title("(b) 燒機槽位數 vs. 產出", fontsize=9.5)
    fig.tight_layout(); save(fig, "s03_sweep.png")

# ---------- S4 layouts
def s4():
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6))
    ks = ["A", "B", "C", "F"]; lab = ["A", "B", "C", "Final"]
    sty = [("white", "..."), ("0.75", "///"), ("0.5", ""), ("0.15", "")]
    for ax, (m, t, u) in zip(axes, [("dist_day", "(a) 每日搬運行走距離", "m/日"), ("w_hand", "(b) 搬運人員平均等待", "分/次"), ("lt_2U", "(c) 2U 平均前置時間（投料→出貨）", "小時")]):
        y = [M("E3_" + k, m) for k in ks]; e = [CI("E3_" + k, m) for k in ks]
        for i in range(4):
            ax.bar(i, y[i], yerr=e[i], color=sty[i][0], hatch=sty[i][1], edgecolor="black", lw=0.6, capsize=2, width=0.65)
            ax.text(i, y[i] + max(y) * 0.03 + e[i], ("{:,.0f}" if y[i] > 100 else "{:.1f}").format(y[i]), ha="center", fontsize=7.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(lab, fontsize=8.5); ax.set_title(t, fontsize=9.5); ax.set_ylabel(u, fontsize=8); clean(ax)
        ax.set_ylim(0, max(y) * 1.2)
    fig.tight_layout(); save(fig, "s04_layouts.png")

def s5():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), sharey=True)
    hrs = np.arange(8)
    for ax, key, t in [(axes[0], "hand_hourly", "(a) 搬運人員（3 人）"), (axes[1], "amr_hourly", "(b) AMR（2 台）")]:
        for k, st_, mk, nm in [("A", "--", "o", "Layout A"), ("F", "-", "s", "Final")]:
            h = R["E3_" + k][key]; y = [h[str(i)] if str(i) in h else h[i] for i in hrs]
            ax.plot(hrs + 0.5, np.array(y) * 100, color="black", ls=st_, marker=mk, ms=5, mfc="white" if k == "A" else "black", label=nm, lw=1.2)
        ax.set_xticks(range(0, 9)); ax.set_xticklabels(["08", "09", "10", "11", "12", "13", "14", "15", "16"], fontsize=8)
        ax.set_xlabel("時刻", fontsize=8.5); ax.set_title(t, fontsize=9.5); clean(ax); ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("小時利用率（%）", fontsize=8.5)
    fig.tight_layout(); save(fig, "s05_hourly.png")

def s6():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    for ax, (kind, ns, m, t) in zip(axes, [("amr", [1, 2, 3], "w_amr", "(a) AMR 台數 vs. 平均等待"), ("hand", [2, 3, 4], "w_hand", "(b) 搬運人員數 vs. 平均等待")]):
        for k, st_, mk, nm in [("A", "--", "o", "Layout A"), ("F", "-", "s", "Final")]:
            y = [M(f"E4_{kind}_{k}_{n}", m) for n in ns]
            ax.plot(ns, y, color="black", ls=st_, marker=mk, mfc="white" if k == "A" else "black", label=nm, lw=1.2)
            for n, v in zip(ns, y):
                ax.text(n + 0.06, v + (0.9 if k == "A" else -1.6), "%.1f" % v, fontsize=7)
        ax.set_xticks(ns); ax.set_xlabel("台數" if kind == "amr" else "人數", fontsize=8.5); ax.set_ylabel("平均等待（分/次）", fontsize=8.5)
        ax.set_title(t, fontsize=9.5); ax.legend(fontsize=8, frameon=False); clean(ax)
    fig.tight_layout(); save(fig, "s06_fleet.png")

def s7():
    fig, ax = plt.subplots(figsize=(8.6, 3.7))
    pos = []; data = []; labs = []
    for i, lvl in enumerate(["none", "mild", "severe"]):
        for j, k in enumerate("BF"):
            key = f"E5_{k}_{lvl}"
            d = np.array(R[key]["daily_all"], float) - 10  # remove GPU approx? use 1U/2U-> approx via thr
            data.append(np.array(R[key]["thr_runs"]) - M(key, "thr_GPU")); pos.append(i * 3 + j)
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, medianprops=dict(color="black"))
    for k, b in enumerate(bp["boxes"]):
        b.set_facecolor("white" if k % 2 == 0 else "0.55"); b.set_hatch("///" if k % 2 == 0 else "")
    ax.set_xticks([0.5, 3.5, 6.5]); ax.set_xticklabels(["無停機", "輕度停機\n（站 MTBF 1,200 分、MTTR 30 分）", "重度停機\n（站 MTBF 480 分、MTTR 45 分）"], fontsize=8.5)
    ax.axhline(50, color="0.4", ls=":", lw=0.8); ax.set_ylabel("1U/2U 日產出（20 次重複的月平均）", fontsize=8.5); clean(ax)
    hs = [Rectangle((0, 0), 1, 1, fc="white", hatch="///", ec="black"), Rectangle((0, 0), 1, 1, fc="0.55", ec="black")]
    ax.legend(hs, ["Layout B 單一直線", "Final 雙 Cell"], fontsize=8.5, frameon=False, loc="lower left")
    save(fig, "s07_breakdown.png")

def s8():
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ks = [("E3_F", "現況需求\n（GPU 10）"), ("E6_now", "GPU 20\n資源不變"), ("E6_teams", "GPU 20\n組裝 4 組＋包裝 2 站"), ("E6_full", "GPU 20\n再＋測試灣 12")]
    y = [M(k, "thr_GPU") for k, _ in ks]; e = [CI(k, "thr_GPU") for k, _ in ks]
    tg = [10, 20, 20, 20]
    for i, (k, lb) in enumerate(ks):
        ax.bar(i, y[i], yerr=e[i], color="0.3" if i in (0, 3) else "white", hatch="" if i in (0, 3) else "///", edgecolor="black", width=0.6, capsize=2)
        ax.plot([i - 0.35, i + 0.35], [tg[i], tg[i]], color="black", ls="--", lw=1)
        ax.text(i, y[i] + 0.6, "%.1f" % y[i], ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels([lb for _, lb in ks], fontsize=8.2); ax.set_ylabel("GPU 日產出（台/日）", fontsize=8.5)
    ax.text(3.45, 20.3, "需求", fontsize=7.5); clean(ax); ax.set_ylim(0, 23)
    save(fig, "s08_growth.png")

for f in (s1, s2, s3, s4, s5, s6, s7, s8):
    f()
print("figs ok")
