"""Lakeside 预订取消预测 —— 模型无关规则（Assignment 2, Part 1）。

只用手写规则，不训练任何模型。评分标准是准确率。
基线：全预测 0（没人取消）约 0.598。
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent

train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")
sample = pd.read_csv(BASE / "sample_submission.csv")

y = train["Canceled"].values
N = len(y)


def acc(pred):
    return (pred == y).mean()


def evaluate(name, rule):
    """rule: 返回 0/1 预测数组的函数（作用于 train）。"""
    pred = rule(train).astype(int)
    a = acc(pred)
    print(f"{name:55s} 准确率 {a:.4f}")
    return a


print(f"训练样本 {N}，取消率 {y.mean():.4f}")
print(f"基线（全预测 0）准确率 {1 - y.mean():.4f}\n")

# ---------------------------------------------------------------- 单列规则
print("=" * 70)
print("规则 1（单列规则）：提前天数 LeadTimeDays")
print("=" * 70)

# 阈值 >=90 时，训练准确率最高（见前面的单列搜索）
a1 = evaluate("Rule1: cancel if LeadTimeDays >= 90", lambda df: df["LeadTimeDays"] >= 90)

# 证据：阈值两侧的取消率
lt_hi = train["LeadTimeDays"] >= 90
print(f"  LeadTimeDays >= 90：{lt_hi.sum()} 条，取消率 {y[lt_hi].mean():.4f}")
print(f"  LeadTimeDays <  90：{(~lt_hi).sum()} 条，取消率 {y[~lt_hi].mean():.4f}\n")

# ---------------------------------------------------------------- 加订金类型
print("=" * 70)
print("规则 2（加 DepositType）：NonRefundable 几乎不取消")
print("=" * 70)
a2 = evaluate(
    "Rule2: cancel if (LeadTimeDays>=90) AND (DepositType != NonRefundable)",
    lambda df: (df["LeadTimeDays"] >= 90) & (df["DepositType"] != "NonRefundable"),
)
print(f"  相比规则 1 提升：+{a2 - a1:.4f}\n")

# 证据
nr = train["DepositType"] == "NonRefundable"
print(f"  NonRefundable：{nr.sum()} 条，取消率 {y[nr].mean():.4f}")
# 其中提前 >=90 天的 NonRefundable
nr_hi = nr & (train["LeadTimeDays"] >= 90)
print(f"  NonRefundable 且提前>=90天：{nr_hi.sum()} 条，取消率 {y[nr_hi].mean():.4f}（规则1会把它们错判成取消）\n")

# ---------------------------------------------------------------- 加历史取消
print("=" * 70)
print("规则 3（加 PriorCancellations）：有前科的客人更易取消")
print("=" * 70)
a3 = evaluate(
    "Rule3: Rule2 OR (PriorCancellations >= 2)",
    lambda df: ((df["LeadTimeDays"] >= 90) & (df["DepositType"] != "NonRefundable"))
    | (df["PriorCancellations"] >= 2),
)
print(f"  相比规则 2 提升：+{a3 - a2:.4f}\n")

pc = train["PriorCancellations"] >= 2
print(f"  PriorCancellations>=2：{pc.sum()} 条，取消率 {y[pc].mean():.4f}")
print(f"  PriorCancellations<2 ：{(~pc).sum()} 条，取消率 {y[~pc].mean():.4f}\n")

# ---------------------------------------------------------------- 生成提交文件
def submit(name, rule):
    pred = rule(test).astype(int)
    sub = sample.copy()
    sub["Canceled"] = pred
    out = BASE / f"submission_{name}.csv"
    sub.to_csv(out, index=False)
    print(f"  已生成 {out.name}：{len(sub)} 行，预测取消率 {pred.mean():.4f}")

print("=" * 70)
print("生成提交文件")
print("=" * 70)
submit("rule1_leadtime90", lambda df: df["LeadTimeDays"] >= 90)
submit(
    "rule2_deposit",
    lambda df: (df["LeadTimeDays"] >= 90) & (df["DepositType"] != "NonRefundable"),
)
submit(
    "rule3_best",
    lambda df: ((df["LeadTimeDays"] >= 90) & (df["DepositType"] != "NonRefundable"))
    | (df["PriorCancellations"] >= 2),
)
