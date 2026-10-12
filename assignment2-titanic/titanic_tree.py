"""Titanic 决策树预测（Assignment 2, Part 2）。

按讲义流程，两步提交：
  1. 默认决策树 DecisionTreeClassifier()（不改任何参数）
  2. 限深决策树 DecisionTreeClassifier(max_depth=5)

运行前：把 Titanic 的 train.csv / test.csv / gender_submission.csv 放到本文件同一目录。
"""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).resolve().parent

train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# ---- 讲义做法：train 直接丢缺失行，test 填缺失 ----
train = train[features + ["Survived"]].dropna()
test = test[features].copy()

# test 里的缺失值：用 train 的统计量填（不能用 test 自己的，见写稿 Q1）
test["Age"] = test["Age"].fillna(train["Age"].mean())
test["Fare"] = test["Fare"].fillna(train["Fare"].mean())
test["Embarked"] = test["Embarked"].fillna(train["Embarked"].mode()[0])

X = train[features]
y = train["Survived"]
X_test = test[features]

# ---- 一次热编码（Sex / Embarked 是文字）----
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)

# 让 test 的列对齐 train（讲义 Q2 要讨论的问题点）
X_test = X_test.reindex(columns=X.columns, fill_value=0)

print("训练集形状（dropna 后）:", X.shape)
print("测试集形状:", X_test.shape)
print("X.columns     :", list(X.columns))
print("X_test.columns:", list(X_test.columns))
print("两列集合是否一致:", set(X.columns) == set(X_test.columns), "\n")

# ---- 提交 1：默认决策树 ----
m1 = DecisionTreeClassifier()
m1.fit(X, y)
pred1 = m1.predict(X_test)
sub1 = pd.DataFrame({"PassengerId": pd.read_csv(BASE / "test.csv")["PassengerId"],
                     "Survived": pred1})
sub1.to_csv(BASE / "submission_default_tree.csv", index=False)
print(f"[默认树] 深度={m1.get_depth()}  预测生还率={pred1.mean():.4f}  -> submission_default_tree.csv")

# ---- 提交 2：限深决策树 ----
m2 = DecisionTreeClassifier(max_depth=5)
m2.fit(X, y)
pred2 = m2.predict(X_test)
sub2 = pd.DataFrame({"PassengerId": pd.read_csv(BASE / "test.csv")["PassengerId"],
                     "Survived": pred2})
sub2.to_csv(BASE / "submission_depth5_tree.csv", index=False)
print(f"[限深树] 深度={m2.get_depth()}  预测生还率={pred2.mean():.4f}  -> submission_depth5_tree.csv")
