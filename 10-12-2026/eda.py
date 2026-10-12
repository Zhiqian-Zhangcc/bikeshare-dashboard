import pandas as pd

day = pd.read_csv("day.csv")
hour = pd.read_csv("hour.csv")

print("=== day.csv ===")
print("shape:", day.shape)
print(day.dtypes)
print("nulls:\n", day.isna().sum().sum())
print("date range:", day.dteday.min(), "->", day.dteday.max())
print("cnt describe:\n", day.cnt.describe())

print("\n=== hour.csv ===")
print("shape:", hour.shape)
print("nulls:", hour.isna().sum().sum())
print("date range:", hour.dteday.min(), "->", hour.dteday.max())
print("cnt describe:\n", hour.cnt.describe())

# Yearly totals
print("\n=== total rides per year (day) ===")
print(day.groupby("yr").cnt.agg(["sum", "mean", "count"]))

print("\n=== total rides per season (day) ===")
print(day.groupby("season").cnt.agg(["sum", "mean"]))

print("\n=== average rides by weekday (day), 0=Sunday ===")
print(day.groupby("weekday").cnt.mean().round(1))

print("\n=== registered vs casual share ===")
reg = day.registered.sum()
cas = day.casual.sum()
tot = day.cnt.sum()
print(f"total={tot}, registered={reg} ({reg/tot*100:.1f}%), casual={cas} ({cas/tot*100:.1f}%)")

print("\n=== workingday vs non-workingday (day) ===")
print(day.groupby("workingday").cnt.agg(["sum", "mean"]))

# Hourly pattern: rush hours
print("\n=== average rides by hour (hour.csv), full day ===")
h = hour.groupby("hr").cnt.mean().round(1)
print(h.to_string())

print("\n=== average rides by hour: workingday vs weekend ===")
wd = hour[hour.workingday == 1].groupby("hr").cnt.mean().round(1)
we = hour[hour.workingday == 0].groupby("hr").cnt.mean().round(1)
print(pd.DataFrame({"workingday": wd, "non_working": we}).to_string())

print("\n=== peak hour overall ===")
print("peak hr:", hour.groupby('hr').cnt.mean().idxmax(),
      "avg:", round(hour.groupby('hr').cnt.mean().max(), 1))

print("\n=== weathersit distribution (day) ===")
print(day.weathersit.value_counts(normalize=True).round(3))

print("\n=== busiest day & lowest day ===")
print("max day:", day.loc[day.cnt.idxmax(), ["dteday", "cnt", "season", "weathersit"]].to_dict())
print("min day:", day.loc[day.cnt.idxmin(), ["dteday", "cnt", "season", "weathersit"]].to_dict())

print("\n=== Oct 30 2012 (Hurricane Sandy) ===")
sandy = day[day.dteday == "2012-10-30"]
print(sandy[["dteday", "cnt", "casual", "registered", "weathersit"]].to_string())

print("\n=== year over year monthly trend (day) ===")
day["ym"] = day.dteday.str[:7]
print(day.pivot_table(index="mnth", columns="yr", values="cnt", aggfunc="sum").round(0).to_string())
