"""湖滨度假村预订取消预测 —— XGBoost 方案。

目标：对 test.csv 的每条预订预测 Canceled (0/1)，评分标准为分类准确率。
基线（全预测 0）约 59.8%，需显著超过。
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

BASE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- 数据加载
train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")
sample = pd.read_csv(BASE / "sample_submission.csv")

y = train["Canceled"].values


# ---------------------------------------------------------------- 特征工程
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # BookingId -> 预订号 / 房号 / 预订内房间数
    df["reservation"] = df["BookingId"].str.extract(r"B-(\d+)-")
    df["room_no"] = df["BookingId"].str.extract(r"-(\d+)$").astype(int)
    res_size = df.groupby("reservation")["BookingId"].transform("count")
    df["res_size"] = res_size.astype(int)

    # 数值特征
    df["total_rate"] = df["RoomRateUSD"] * df["Nights"]  # 总房费
    df["leadtime_roomrate"] = df["LeadTimeDays"] * df["RoomRateUSD"]

    # 布尔 -> 0/1
    df["IsWeekendStay"] = df["IsWeekendStay"].astype(int)
    df["ParkingRequested"] = df["ParkingRequested"].astype(int)

    # 分类特征缺失值 -> "missing"
    for c in ["DepositType", "LoyaltyTier"]:
        df[c] = df[c].fillna("missing")

    # 提前预订天数分箱（分段单调性很强，分箱后更利于树模型捕捉）
    df["lead_bin"] = pd.cut(
        df["LeadTimeDays"],
        bins=[-1, 7, 30, 60, 90, 120, 180, 1000],
        labels=["<=7", "8-30", "31-60", "61-90", "91-120", "121-180", ">180"],
    ).astype(str)

    return df


train = engineer(train)
test = engineer(test)

# ---------------------------------------------------------------- 特征清单
cat_features = [
    "Channel",
    "DepositType",
    "LoyaltyTier",
    "GuestRegion",
    "lead_bin",
]
num_features = [
    "LeadTimeDays",
    "RoomRateUSD",
    "PartySize",
    "Nights",
    "ArrivalMonth",
    "SpecialRequests",
    "PriorStays",
    "PriorCancellations",
    "room_no",
    "res_size",
    "total_rate",
    "leadtime_roomrate",
    "IsWeekendStay",
    "ParkingRequested",
]

FEATURES = num_features + cat_features

# 分类特征转 category dtype，供 XGBoost 原生处理
X_train = train[FEATURES].copy()
X_test = test[FEATURES].copy()
for c in cat_features:
    cats = pd.CategoricalDtype(
        categories=sorted(set(X_train[c]) | set(X_test[c])), ordered=False
    )
    X_train[c] = X_train[c].astype(cats)
    X_test[c] = X_test[c].astype(cats)

# ---------------------------------------------------------------- 交叉验证
groups = train["reservation"].values
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(y))

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 5,
    "learning_rate": 0.04,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 10,
    "reg_lambda": 1.0,
    "n_estimators": 2500,
    "early_stopping_rounds": 100,
    "random_state": 0,
}

fold_accs = []
best_iters = []
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y, groups)):
    Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]
    model = xgb.XGBClassifier(
        **params,
        enable_categorical=True,
    )
    model.fit(
        Xtr,
        ytr,
        eval_set=[(Xva, yva)],
        verbose=False,
    )
    p = model.predict_proba(Xva)[:, 1]
    oof[va_idx] = p
    acc = accuracy_score(yva, (p >= 0.5).astype(int))
    fold_accs.append(acc)
    best_iters.append(model.best_iteration)
    print(f"fold {fold}: acc={acc:.4f}  best_iter={model.best_iteration}")

print(f"\nOOF accuracy @0.5 = {accuracy_score(y, (oof >= 0.5).astype(int)):.4f}")
print(f"fold accs mean = {np.mean(fold_accs):.4f}")

# ---------------------------------------------------------------- 阈值调优
best_t, best_acc = 0.5, accuracy_score(y, (oof >= 0.5).astype(int))
for t in np.arange(0.40, 0.60, 0.01):
    a = accuracy_score(y, (oof >= t).astype(int))
    if a > best_acc:
        best_acc, best_t = a, t
print(f"best threshold = {best_t:.2f} -> OOF acc = {best_acc:.4f}")

# ---------------------------------------------------------------- 全量训练 + 预测
full_params = {**params, "n_estimators": int(np.median(best_iters))}
full_params.pop("early_stopping_rounds", None)
full_model = xgb.XGBClassifier(**full_params, enable_categorical=True)
full_model.fit(X_train, y, verbose=False)
pred_prob = full_model.predict_proba(X_test)[:, 1]
pred = (pred_prob >= best_t).astype(int)

sub = sample.copy()
sub["Canceled"] = pred
sub.to_csv(BASE / "submission.csv", index=False)
print(f"\nSaved submission.csv  ({len(sub)} rows)")
print("pred distribution:", np.bincount(pred))
print("pred mean (cancel rate):", pred.mean().round(4))

# 特征重要性
imp = pd.Series(full_model.feature_importances_, index=FEATURES).sort_values(
    ascending=False
)
print("\nfeature importance:")
print(imp.to_string())
