"""Model Evaluation 课堂练习 —— 计算脚本（label 编码版）。"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "train.csv")
print("原始数据:", df.shape, " 取消率 %.4f" % df["Canceled"].mean())

# ---------------------------------------------------------------- 列定义
NUM_COLS = ["LeadTimeDays", "RoomRateUSD", "PartySize", "Nights", "ArrivalMonth",
            "SpecialRequests", "PriorStays", "PriorCancellations"]
BOOL_COLS = ["IsWeekendStay", "ParkingRequested"]
CAT_COLS = ["Channel", "DepositType", "LoyaltyTier", "GuestRegion"]


def engineer(d):
    """Part 2 派生特征（复用 Lakeside 规则里的发现）。"""
    d = d.copy()
    d["reservation"] = d["BookingId"].str.extract(r"B-(\d+)-")
    d["res_size"] = d.groupby("reservation")["BookingId"].transform("count").astype(int)
    d["total_rate"] = d["RoomRateUSD"] * d["Nights"]
    d["leadtime_roomrate"] = d["LeadTimeDays"] * d["RoomRateUSD"]
    d["lead_bin"] = pd.cut(
        d["LeadTimeDays"], bins=[-1, 7, 30, 60, 90, 120, 180, 1000],
        labels=["<=7", "8-30", "31-60", "61-90", "91-120", "121-180", ">180"],
    ).astype(str)
    return d


def fit_prep(d, num_cols, cat_cols):
    """预处理统计量（中位数、分类类别）只从 train 拟合，避免泄漏。"""
    medians = {c: d[c].median() for c in num_cols}
    cat_cats = {c: list(pd.unique(d[c].fillna("missing").astype(str))) for c in cat_cols}
    return medians, cat_cats


def prep(df, num_cols, bool_cols, cat_cols, medians, cat_cats):
    d = df.copy()
    for c in num_cols:
        d[c] = d[c].fillna(medians[c])
    for c in bool_cols:
        d[c] = d[c].astype(int)
    for c in cat_cols:
        d[c] = pd.Categorical(d[c].fillna("missing").astype(str),
                              categories=cat_cats[c]).codes
    return d[num_cols + bool_cols + cat_cols]


# ---------------------------------------------------------------- 划分（先锁测试集）
train_val, test = train_test_split(df, test_size=0.20, random_state=409)
train, val = train_test_split(train_val, test_size=0.25, random_state=409)
print(f"train {len(train)} ({len(train)/len(df):.1%}) | val {len(val)} ({len(val)/len(df):.1%}) | test {len(test)} ({len(test)/len(df):.1%})")
print("test 已锁，Part 3 前不再使用\n")


def report(name, Xtr, ytr, Xva, yva, **kw):
    m = DecisionTreeClassifier(random_state=0, **kw)
    m.fit(Xtr, ytr)
    tr, va = m.score(Xtr, ytr), m.score(Xva, yva)
    print(f"{name}: train {tr:.4f} | val {va:.4f} | 深度 {m.get_depth()} | 叶子 {m.get_n_leaves()}")
    return m, tr, va


# ================================================================ Part 1
print("=" * 70)
print("Part 1a —— 4 个最相关列，默认树")
print("=" * 70)
n1a, c1a = ["LeadTimeDays", "PriorCancellations"], ["DepositType", "Channel"]
med, cat = fit_prep(train, n1a, c1a)
Xtr = prep(train, n1a, [], c1a, med, cat)
Xva = prep(val, n1a, [], c1a, med, cat)
ytr, yva = train["Canceled"].values, val["Canceled"].values
m1a, tr1a, va1a = report("Part 1a", Xtr, ytr, Xva, yva)
print(f"  特征数 {Xtr.shape[1]}\n")

print("=" * 70)
print("Part 1b —— 全部原始列，默认树")
print("=" * 70)
med, cat = fit_prep(train, NUM_COLS, CAT_COLS)
Xtr = prep(train, NUM_COLS, BOOL_COLS, CAT_COLS, med, cat)
Xva = prep(val, NUM_COLS, BOOL_COLS, CAT_COLS, med, cat)
m1b, tr1b, va1b = report("Part 1b", Xtr, ytr, Xva, yva)
print(f"  特征数 {Xtr.shape[1]}\n")

# ================================================================ Part 2
print("=" * 70)
print("Part 2 —— 特征工程 + 参数调优（5 折 CV，只用 train+val）")
print("=" * 70)

train_val_e = engineer(train_val)
P2_NUM = NUM_COLS + ["res_size", "total_rate", "leadtime_roomrate"]
P2_CAT = CAT_COLS + ["lead_bin"]

# 原始列 baseline 的预处理（用于 log 对比）
med_raw, cat_raw = fit_prep(train_val, NUM_COLS, CAT_COLS)
X_raw = prep(train_val, NUM_COLS, BOOL_COLS, CAT_COLS, med_raw, cat_raw)

med2, cat2 = fit_prep(train_val_e, P2_NUM, P2_CAT)
X2 = prep(train_val_e, P2_NUM, BOOL_COLS, P2_CAT, med2, cat2)
y2 = train_val_e["Canceled"].values
print(f"原始列特征数 {X_raw.shape[1]}，工程列特征数 {X2.shape[1]}，CV 样本 {X2.shape[0]}\n")


def cv_report(X, y, **kw):
    res = cross_validate(DecisionTreeClassifier(random_state=0, **kw), X, y,
                         cv=5, scoring="accuracy", return_train_score=True)
    return res["train_score"].mean(), res["test_score"].mean()


print("--- 尝试日志 ---")
log = []
for name, X, kw in [
    ("原始列, 默认", X_raw, {}),
    ("原始列, max_depth=6", X_raw, {"max_depth": 6}),
    ("工程列, 默认", X2, {}),
    ("工程列, max_depth=6", X2, {"max_depth": 6}),
]:
    tr, va = cv_report(X, y2, **kw)
    log.append({"attempt": name, "train_acc": round(tr, 4), "cv_acc": round(va, 4)})
    print(f"  {name:20s}  train {tr:.4f}  CV {va:.4f}")

print("\n--- max_depth 扫描（固定工程列）---")
depths = [2, 4, 6, 8, 12, 20]
sweep_tr, sweep_cv = [], []
for d in depths:
    tr, va = cv_report(X2, y2, max_depth=d)
    sweep_tr.append(tr)
    sweep_cv.append(va)
    print(f"  max_depth={d:2d}  train {tr:.4f}  CV {va:.4f}")

best_i = int(np.argmax(sweep_cv))
best_d = depths[best_i]
print(f"\n最佳 max_depth = {best_d} -> CV {sweep_cv[best_i]:.4f}")

plt.figure(figsize=(7, 4.5))
plt.plot(depths, sweep_tr, "o-", label="training accuracy")
plt.plot(depths, sweep_cv, "s-", label="CV accuracy")
plt.axvline(best_d, color="gray", ls="--", alpha=0.6)
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.title("Decision tree: training vs CV accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(BASE / "max_depth_curve.png", dpi=120)
print("图已保存 max_depth_curve.png")

# ================================================================ Part 3
print("\n" + "=" * 70)
print("Part 3 —— 唯一一次看测试集")
print("=" * 70)

train_val_e = engineer(train_val)
med3, cat3 = fit_prep(train_val_e, P2_NUM, P2_CAT)
Xtv = prep(train_val_e, P2_NUM, BOOL_COLS, P2_CAT, med3, cat3)
ytv = train_val_e["Canceled"].values
test_e = engineer(test)
Xte = prep(test_e, P2_NUM, BOOL_COLS, P2_CAT, med3, cat3)

m_final = DecisionTreeClassifier(max_depth=best_d, random_state=0).fit(Xtv, ytv)
acc_final = m_final.score(Xte, test["Canceled"].values)

med_b, cat_b = fit_prep(train_val, NUM_COLS, CAT_COLS)
Xtv_b = prep(train_val, NUM_COLS, BOOL_COLS, CAT_COLS, med_b, cat_b)
Xte_b = prep(test, NUM_COLS, BOOL_COLS, CAT_COLS, med_b, cat_b)
m_base = DecisionTreeClassifier(random_state=0).fit(Xtv_b, ytv)
acc_base = m_base.score(Xte_b, test["Canceled"].values)

print(f"最终模型(max_depth={best_d})  测试准确率 {acc_final:.4f}  (CV 选中时 {sweep_cv[best_i]:.4f})")
print(f"Part1 基线(全列默认树)       测试准确率 {acc_base:.4f}")
print("最佳 Lakeside 规则(训练集) = 0.7544")
