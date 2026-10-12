# Assignment 2 — Prediction Without and With a Model

**Team name:** [Team 12]
**Groupmates:** [张祉谦、朱航霖、何雨佳、刘泽宇]

---

## Part 1 — Lakeside Competition (Model-Free Rules)

The goal is to flag which bookings will be canceled, using rules only. A booking that
predicts `Canceled = 0` for everyone scores about **0.596** — that is the floor. My rules
below all clear it by a wide margin.

All accuracies below are on the training set (13,544 bookings, 40.2% cancellation rate);
the leaderboard scores are listed next to them.

### Rule 1 — one column (LeadTimeDays)

> **Flag a booking as a cancellation when it was made 90 or more days before arrival.**

**Evidence.** Lead time is the single strongest signal in the data. Cutting the data at 90
days splits it cleanly:

| Lead time | Bookings | Cancellation rate |
|---|---:|---:|
| `< 90 days` | 9,921 | 29.1% |
| `>= 90 days` | 3,623 | **70.8%** |

A booking made 90+ days out cancels more than twice as often as one made inside 90 days.
I picked the threshold by scanning every possible cutoff and taking the one that maximized
accuracy — 90 days is where the rate first clearly crosses 50%.

**Score:** training accuracy **0.7089** · leaderboard **[填分数]**

### Rule 2 — add DepositType (NonRefundable override)

> **Flag a booking as a cancellation when it was made 90+ days before arrival *and* the
> guest did not pay a non-refundable deposit.**

**Evidence.** `NonRefundable` bookings almost never cancel — only **7.0%** overall, and
critically only **8.7%** even among those booked 90+ days out. Rule 1 was mislabeling these
381 far-out, non-refundable bookings as cancellations. Excluding them fixes that.

**What the change bought:** training accuracy **0.7322**, up **+0.0233** from Rule 1.
This matched what I expected before submitting: the non-refundable deposit is a strong
commitment signal that overrides lead time.

**Score:** leaderboard **[填分数]**

### Rule 3 — add PriorCancellations (best rule)

> **Flag a booking as a cancellation when (it was made 90+ days before arrival and not
> non-refundable), *or* the guest has 2 or more prior cancellations.**

**Evidence.** Guests with a history of cancelling keep cancelling:

| Prior cancellations | Bookings | Cancellation rate |
|---:|---:|---:|
| `0` | 12,340 | 37.4% |
| `1` | 298 | 46.0% |
| `>= 2` | 906 | **76.9%** |

A guest with 2+ prior cancellations cancels 77% of the time regardless of lead time, so it
is worth flagging them even when they booked close to arrival (which Rule 2 would miss).

**What the change bought:** training accuracy **0.7544**, up **+0.0222** from Rule 2.

**Score:** leaderboard **[填分数]**

### The write-up questions

**1. Which column mattered most, and how do you know?**

`LeadTimeDays`. A rule using *only* that column (Rule 1) reaches **0.7089** accuracy against
the 0.596 floor. No other single column comes close — the next best are `Channel = OTA`
(0.663) and `PriorCancellations >= 2` (0.634). The cancellation rate jumps from 29.1% to
70.8% when lead time crosses 90 days, which is the largest gap any column produces.

**2. Which column looked promising and turned out to be worthless?**

`GuestRegion`. On its own it *looked* like `Overseas` guests cancel more (41.4%) than everyone
else (40.0–40.1%):

| Region | Cancellation rate | Bookings |
|---|---:|---:|
| East | 40.1% | 4,102 |
| North | 40.1% | 2,997 |
| Overseas | **41.4%** | 1,370 |
| South | 40.1% | 2,760 |
| West | 40.0% | 2,315 |

The table that fooled me is this one, ranked by mean. But every region sits within ~1
percentage point of the overall 40.2% baseline, and `Overseas` is the *smallest* group
(1,370 rows) — its "lead" is noise, not signal. What I should have looked at first is the
*spread relative to the baseline and the sample size*, not just the rank of the means. A
1-point difference on a small bucket is not actionable.

**3. Does any column only work in combination with another?**

Yes — `DepositType = NonRefundable` reverses the effect of `LeadTimeDays`. Lead time says
"booked far out → cancel", but that rule flips for non-refundable guests:

| | Lead time < 90 | Lead time >= 90 |
|---|---:|---:|
| **NonRefundable** | ~7% cancel | **8.7% cancel** (stays low) |
| **Other deposits** | low | **~74% cancel** (jumps up) |

For guests with other deposit types, being far out nearly triples the cancellation rate; for
non-refundable guests it barely moves. So `NonRefundable` is worthless as a standalone flag
(it scores 0.469 alone — below the 0.596 floor) but earns its place only as a *condition on
top of* lead time. I found it by cross-tabulating `LeadTimeDays` buckets against each
`DepositType` value, which is where the reversal shows up.

---

## Part 2 — Titanic Decision Tree

Two submissions: the default `DecisionTreeClassifier()` and
`DecisionTreeClassifier(max_depth=5)`.

**Scores:** default tree **[填分数]** · depth-5 tree **[填分数]**

### Question 1 — missing data on the two sides

The lecture drops rows with missing values from `train.csv` but *fills* them in `test.csv`.

You cannot drop rows from the test set because Kaggle requires a prediction for **every**
`PassengerId` in `test.csv` — the submission must have exactly 418 rows. Dropping a test row
would mean refusing to predict that passenger, producing a submission with the wrong number
of rows, which Kaggle rejects.

For the single missing `Fare` in the test set, the **training-set mean is more defensible**
than the test-set mean. The model was trained on the training data, so "a typical fare" in
the model's world is defined by the training distribution. Using the test mean would (a)
peek at the test set — information you are not supposed to use at prediction time, and (b)
be inconsistent with what the model learned. I filled it with `train["Fare"].mean()`.

### Question 2 — one-hot encoding done twice

`X` and `X_test` are encoded with `pd.get_dummies` separately, so their columns are only
guaranteed to match if the two sets happen to contain the same categories. Printing them:

```
X.columns     : ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
X_test.columns: ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
```

If `test.csv` had contained a category the training set did not (say an `Embarked` value
the training set never saw), `X_test` would gain an extra column `X` lacks. Without any
check, `model.predict(X_test)` either **raises an error** (feature count mismatch) or — more
dangerously — if the columns get **reindexed into a different order, silently predicts on
misaligned columns** and returns wrong answers with no warning. The line that catches it
before submitting:

```python
assert list(X.columns) == list(X_test.columns), "train/test columns differ!"
```

I also defensively reindex the test columns to the training columns with
`X_test.reindex(columns=X.columns, fill_value=0)`, so any extra category is dropped to zero
rather than silently shifting.

### Question 3 — information gain by hand

Using the training data after dropping missing rows (712 passengers, 40.45% survived).

**Entropy of the target.** 424 died, 288 survived:

```
p0 = 424/712 = 0.5955,   p1 = 288/712 = 0.4045
H(Survived) = -p0·log2(p0) - p1·log2(p1)
            = -(0.5955·log2 0.5955 + 0.4045·log2 0.4045)
            = -(0.5955·(-0.7478) + 0.4045·(-1.3058))
            = -(-0.4453 - 0.5282)
            = 0.9735
```

**Information gain of `Sex`.** Frequency table:

| Sex | Count | Survived | H(bucket) |
|---|---|---:|---:|
| female | 259 | 75.3% | 0.8067 |
| male | 453 | 20.5% | 0.7324 |

```
IG(Sex) = H - [ (259/712)·0.8067 + (453/712)·0.7324 ]
        = 0.9735 - [ 0.3638·0.8067 + 0.6362·0.7324 ]
        = 0.9735 - [ 0.2935 + 0.4660 ]
        = 0.9735 - 0.7595
        = 0.2141
```

**Information gain of `Pclass`.** Frequency table:

| Pclass | Count | Survived | H(bucket) |
|---:|---:|---:|---:|
| 1 | 184 | 65.2% | 0.9321 |
| 2 | 173 | 48.0% | 0.9988 |
| 3 | 355 | 23.9% | 0.7941 |

```
IG(Pclass) = H - [ (184/712)·0.9321 + (173/712)·0.9988 + (355/712)·0.7941 ]
           = 0.9735 - [ 0.2584·0.9321 + 0.2430·0.9988 + 0.4986·0.7941 ]
           = 0.9735 - [ 0.2409 + 0.2427 + 0.3959 ]
           = 0.9735 - 0.8795
           = 0.0940
```

Since `IG(Sex) = 0.2141 > IG(Pclass) = 0.0940`, **ID3 would pick `Sex` as the root**.
Plotting the fitted tree shows its actual root is also `Sex` (both the default Gini tree and
the `criterion='entropy'` tree split first on `Sex`), so it agrees with the ID3 result. (If
they had differed, it would be because scikit-learn's default `DecisionTreeClassifier` uses
the Gini impurity rather than entropy; with `criterion='entropy'` the tree matches ID3
exactly — here both happen to pick `Sex`.)

---

## Files produced

| File | What it is |
|---|---|
| `9-21-2026/rules.py` | Part 1 rule script (reproducible) |
| `9-21-2026/submission_rule1_leadtime90.csv` | Rule 1 submission |
| `9-21-2026/submission_rule2_deposit.csv` | Rule 2 submission |
| `9-21-2026/submission_rule3_best.csv` | Rule 3 (best) submission |
| `assignment2-titanic/titanic_tree.py` | Part 2 decision tree script |
| `assignment2-titanic/submission_default_tree.csv` | Part 2 default-tree submission |
| `assignment2-titanic/submission_depth5_tree.csv` | Part 2 depth-5 submission |
