# BikeShare Demand Monitor

**Team:** Team 8
**Members:** Ma Manxi, Gao Ya, Guan Zixuan, Zhang Zhiqian
**App:** [Streamlit Cloud URL] · **Repo:** [GitHub URL]

A Streamlit dashboard for the **Fleet Operations Manager** of Capital Bikeshare
(Washington, D.C.). It answers a single question that comes back *every week*:
**when is it safe to take bikes out of service for maintenance, so the fleet is
still full during the commute peaks?**

---

## Step 1 — Dataset and decision-maker

**Dataset:** [Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)
— the UCI "Bike Sharing" dataset (Fanaee-T & Gama, 2013). Real rental counts
from the **Capital Bikeshare** system in Washington, D.C., covering
**Jan 2011 – Dec 2012** (731 days, 17,379 hours).

- `day.csv` (58 KB): one row per day — total rentals plus weather/calendar fields.
- `hour.csv` (1.1 MB): one row per hour — same fields, broken out by hour of day.

Both are small and tabular, well under the 25 MB limit. Fields include `season`,
`weathersit`, `temp`/`atemp`/`hum`/`windspeed` (normalized 0–1), `holiday`,
`workingday`, `casual`, `registered`, and `cnt` (total rentals).

**Decision-maker:** the **Fleet Operations Manager**. Their job is to decide when
to pull bikes for maintenance and how many bikes to have on the street each
season. (Not "everyone" — a marketing lead would look at casual riders; this
dashboard is built for the person who schedules the fleet.)

## Step 2 — Justify the dashboard

1. **Who looks at it, how often?** The Fleet Ops Manager, **every Monday morning**.
2. **What decision does it support?**
   > *If the Fleet Operations Manager sees that demand is concentrated in the
   8 a.m. and 5 p.m. weekday peaks and collapses in winter, they will schedule
   maintenance in off-peak hours and low-demand months so the fleet is fully
   available during the commute.*
3. **What do they do today without it?** They schedule maintenance from a
   spreadsheet of last month's totals, or react to complaints ("empty dock at
   rush hour"). That costs **lost rides** when bikes sit in the shop during peak
   demand, and **rider churn** when commuters repeatedly find no bike.
4. **Why a dashboard** and not a weekly email or report? Because the question is
   the same every week but its answer changes with season and time of day — a
   reusable, filterable view beats a static email. The *one* thing that should
   not be a dashboard is a sudden outage; that becomes an alert (Step 4).

## Step 3 — Decide what matters

**EDA (Pandas + Matplotlib).** 731 days, no missing values, dates 2011-01-01 →
2012-12-31. What stood out:

| Finding | Number |
|---|---|
| Total rides (2 yrs) | **3.29M** (2011: 1.24M, 2012: 2.05M, **+65% YoY**) |
| Registered vs casual | **81.2% registered**, 18.8% casual |
| Seasonality (avg/day) | fall 5,644 · summer 4,992 · winter 4,728 · **spring 2,604** |
| Working-day hourly peak | **8 a.m. (477/hr)** and **5 p.m. (525/hr)**; overnight < 60/hr |
| Weekend hourly pattern | flat midday plateau (~370/hr), no rush-hour spike |
| Weather | 63% clear, 34% mist/cloudy, 3% rain/snow |

**What we show:**

- **Three headline metrics** (`st.metric`): *total rides* (scale, with YoY delta),
  *peak-hour demand* (the number maintenance must never starve), *registered
  share* (who the peaks serve — commuters).
- **Chart 1 — hourly demand, working day vs weekend.** Explains *why* maintenance
  must run overnight, not during the commute peaks.
- **Chart 2 — monthly trend (2011 vs 2012).** Shows the winter low season —
  *when* to schedule the heavy, multi-day maintenance.
- **One filter — year.** The manager compares this year against last; everything
  else is already visible in the two charts.

**What we chose *not* to show:**

1. **Weather variables (temp, humidity, wind).** They correlate with demand in
   the EDA, but the manager can't change the weather and can't act on it through
   this dashboard — they check a forecast separately. Weather explains a bad
   day; it doesn't change a maintenance schedule.
2. **The holiday flag.** Only a handful of days a year; the working-day/weekend
   split already captures almost all of it, and a holiday view wouldn't change a
   weekly maintenance decision.

**What could mislead the viewer, and how we handle it:**

- `cnt` is **city-wide across all stations**, so a near-zero day is almost never
  "no demand" — the 22-ride day on **2012-10-29 is Hurricane Sandy**, not a
  system failure. We caption the dashboard so an extreme low isn't misread as an
  outage.
- `temp`/`atemp` are **normalized 0–1** (divided by 41/50), not real °C — we
  don't plot them, so no one can misread "0.3" as 30°C.
- The data is a **frozen 2011–2012 snapshot**; the dashboard states the period
  and the intended production refresh.

**Order (top → bottom):** headline metrics first ("is anything wrong right
now?") → hourly chart ("why does the peak matter") → monthly trend ("when to do
maintenance") → data-period line. No raw-row table — a weekly reviewer doesn't
read individual rows, so we left it out.

## Step 4 — Refresh frequency

1. **How often should the data refresh?** **Nightly**, from the bike-share trip
   log. Each trip is recorded in real time, but the maintenance decision is made
   daily/weekly — real-time refresh would add cost without changing any decision.
2. **How often should someone review it?** The manager reviews **weekly, Monday
   morning**, to schedule that week's maintenance; the seasonal fleet-size
   decision is **monthly/quarterly**. Refresh and review are deliberately
   different: data arrives nightly, decisions happen weekly.
3. **What triggers action?** If the 7-day average of daily rides falls **more
   than 50% below the same-week seasonal norm**, someone investigates — a station
   outage, a weather event, or a system failure. (This is an *alert*, not a
   dashboard check.)
4. **What does stale data cost?** A day stale → bikes sit in the shop while
   commuters find empty docks (lost rides, churn). A week stale → maintenance
   gets scheduled during peak season, so the fleet is short exactly when demand
   is highest.

---

*Dashboard built with Streamlit + Pandas + Matplotlib. `app.py` runs the app;
`data/` holds the committed `day.csv` and `hour.csv`.*
