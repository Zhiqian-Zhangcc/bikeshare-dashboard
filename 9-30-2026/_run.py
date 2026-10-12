"""Lakeside Full Pipeline + Grid Search —— 计算脚本，产出 notebook 里要用的所有真实数字。"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "train.csv")
print("原始数据:", df.shape, " 取消率 %.4f" % df["Canceled"].mean())

# 与课堂练习完全一致的锁定测试集
train_val, test = train_test_split(df, test_size=0.20, random_state=409)
print(f"train_val {len(train_val)} ({len(train_val)/len(df):.1%}) | "
      f"test {len(test)} ({len(test)/len(df):.1%})")
print("test 取消率 %.4f，实际取消数 %d" % (test["Canceled"].mean(), int(test["Canceled"].sum())))
print("train_val 取消率 %.4f" % train_val["Canceled"].mean())

# ---------------------------------------------------------------- 特征工程（与课堂练习一致）
def engineer(d):
    d = d.copy()
    d["reservation"] = d["BookingId"].str.extract(r"B-(\d+)-")
    d["res_size"] = d.groupby("reservation")["BookingId"].transform("count").astype(int)
    d["total_rate"] = d["RoomRateUSD"] * d["Nights"]
    d["leadtime_roomrate"] = d["LeadTimeDays"] * d["RoomRateUSD"]
    d["lead_bin"] = pd.cut(
        d["LeadTimeDays"], bins=[-1, 7, 30, 60, 90, 120, 180, 1000],
        labels=["<=7", "8-30", "31-60", "61-90", "91-120", "121-180", ">180"],
    ).astype(str)
    d["IsWeekendStay"] = d["IsWeekendStay"].astype(int)
    d["ParkingRequested"] = d["ParkingRequested"].astype(int)
    return d


NUM = ["LeadTimeDays", "RoomRateUSD", "PartySize", "Nights", "ArrivalMonth",
       "SpecialRequests", "PriorStays", "PriorCancellations",
       "res_size", "total_rate", "leadtime_roomrate", "IsWeekendStay", "ParkingRequested"]
CAT = ["Channel", "DepositType", "LoyaltyTier", "GuestRegion", "lead_bin"]

train_val = engineer(train_val)
test = engineer(test)
X_train_val = train_val[NUM + CAT]
y_train_val = train_val["Canceled"].values
X_test = test[NUM + CAT]
y_test = test["Canceled"].values
print("特征数", X_train_val.shape[1], " CV 样本", X_train_val.shape[0])

# ---------------------------------------------------------------- 全流程 pipeline
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, NUM),
    ("cat", cat_pipe, CAT),
])
pipe = Pipeline([
    ("prep", preprocessor),
    ("tree", DecisionTreeClassifier(random_state=0)),
])

param_grid = {
    "tree__max_depth": [4, 6, 8, 12, None],
    "tree__min_samples_leaf": [1, 5, 20, 50],
    "tree__min_samples_split": [2, 10, 50],
    "tree__criterion": ["gini", "entropy"],
}
n_combos = 5 * 4 * 3 * 2
print("参数组合数 =", n_combos)

gs = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=1,
                  return_train_score=True)
gs.fit(X_train_val, y_train_val)

print("\nbest params:", gs.best_params_)
print("best CV accuracy: %.4f" % gs.best_score_)
print("refit train accuracy: %.4f" % gs.score(X_train_val, y_train_val))

res = pd.DataFrame(gs.cv_results_)
print("cv_results_ 行数(每个 combo 一行) =", len(res))
print("mean_fit_time 总数 =", res["mean_fit_time"].sum().round(2), "秒")

# 前几名 combo 的 CV 分数（用于说明自动搜索是否赢过手调）
top = res.sort_values("rank_test_score").head(6)
print("\n前几名组合:")
print(top[["param_tree__max_depth", "param_tree__min_samples_leaf",
           "param_tree__min_samples_split", "param_tree__criterion",
           "mean_test_score", "mean_train_score"]].round(4).to_string(index=False))

# ---------------------------------------------------------------- 唯一一次看测试集
y_pred = gs.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)  # [[TN,FP],[FN,TP]]，positive=1
TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
print("\n=== 测试集 ===")
print("test accuracy: %.4f" % acc)
print("confusion matrix [[TN,FP],[FN,TP]]:")
print(cm)
print("TN", TN, "FP", FP, "FN", FN, "TP", TP)
print("precision(正类=canceled): %.4f" % precision_score(y_test, y_pred, pos_label=1))
print("recall(正类=canceled):    %.4f" % recall_score(y_test, y_pred, pos_label=1))
cost_model = 400 * FP + 150 * FN
cost_nobody = 150 * (TP + FN)
print("模型错误总成本 400*FP + 150*FN =", cost_model)
print("全预测0成本    150*(TP+FN)     =", cost_nobody)

# ---------------------------------------------------------------- 特征重要性
tree = gs.best_estimator_.named_steps["tree"]
prep = gs.best_estimator_.named_steps["prep"]
feat_names = prep.get_feature_names_out()
importances = tree.feature_importances_

print("\n=== one-hot 级别重要性(前 15) ===")
oh = pd.Series(importances, index=feat_names).sort_values(ascending=False)
print(oh.head(15).round(4).to_string())

# 聚合回原始特征
imp_map = {}
for name, val in zip(feat_names, importances):
    if name.startswith("num__"):
        orig = name[len("num__"):]
    elif name.startswith("cat__"):
        orig = name[len("cat__"):].rsplit("_", 1)[0]
    else:
        orig = name
    imp_map[orig] = imp_map.get(orig, 0.0) + val
imp = pd.Series(imp_map).sort_values(ascending=False)
print("\n=== 聚合后特征重要性 ===")
print(imp.round(4).to_string())

# ---------------------------------------------------------------- 分组取消率（写给营收经理的证据）
print("\n=== 训练集分组取消率 ===")
for c in ["DepositType", "Channel", "lead_bin", "res_size"]:
    g = train_val.groupby(c)["Canceled"].agg(["mean", "count"])
    g["mean"] = (g["mean"] * 100).round(1)
    print(f"[{c}]")
    print(g.to_string())
    print()
pc = train_val["PriorCancellations"]
print("[PriorCancellations 0 vs >=2]")
print("0 次:", round(train_val.loc[pc == 0, "Canceled"].mean(), 4),
      " >=2 次:", round(train_val.loc[pc >= 2, "Canceled"].mean(), 4))
