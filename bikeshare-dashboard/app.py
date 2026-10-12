"""BikeShare Demand Monitor — a Streamlit dashboard for the Fleet Operations
Manager of Capital Bikeshare (Washington, D.C.).

It answers one question, asked every week: *when is it safe to take bikes out
of service for maintenance, so the fleet is still full during the commute
peaks?* Built for the In-Class Group Exercise (data app), not as a full BI tool.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

plt.style.use("seaborn-v0_8")

# Fixed categorical colors (colorblind-safe blue/orange pair).
COMMUTE = "#1f77b4"   # working day
LEISURE = "#ff7f0e"   # weekend / holiday
YEAR_2011 = "#9ecae1"
YEAR_2012 = "#08519c"

TEAM_NAME = "Team 8"  # <- replace with your team name

st.set_page_config(page_title="BikeShare Demand Monitor", layout="wide")


@st.cache_data
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    day = pd.read_csv("data/day.csv")
    hour = pd.read_csv("data/hour.csv")
    day["dteday"] = pd.to_datetime(day["dteday"])
    hour["dteday"] = pd.to_datetime(hour["dteday"])
    return day, hour


day, hour = load()

# ---------------------------------------------------------------------------
# Title + caption: who this is for, and the decision it supports.
# ---------------------------------------------------------------------------
st.title(f"{TEAM_NAME} — BikeShare Demand Monitor")
st.caption(
    "For the **Fleet Operations Manager** of Capital Bikeshare: if they see "
    "demand concentrated in the 8 a.m. and 5 p.m. weekday peaks and collapsing "
    "in winter, they will schedule maintenance in off-peak hours and low-demand "
    "months so bikes stay on the street during the commute."
)

# ---------------------------------------------------------------------------
# Filter: the one thing the manager actually selects — which year to look at.
# ---------------------------------------------------------------------------
year_opt = st.selectbox("Year", ["Both years", "2011", "2012"])
if year_opt == "2011":
    yr_vals = [0]
elif year_opt == "2012":
    yr_vals = [1]
else:
    yr_vals = [0, 1]

d = day[day["yr"].isin(yr_vals)]
h = hour[hour["yr"].isin(yr_vals)]

# ---------------------------------------------------------------------------
# Headline metrics (top = "is anything wrong right now?").
# ---------------------------------------------------------------------------
total = int(d["cnt"].sum())
reg_share = d["registered"].sum() / d["cnt"].sum() * 100

peak = h[h["workingday"] == 1].groupby("hr")["cnt"].mean()
peak_hr = int(peak.idxmax())
peak_val = peak.max()

if year_opt == "2012":
    prev_total = int(day[day["yr"] == 0]["cnt"].sum())
    delta = f"+{(total - prev_total) / prev_total * 100:.0f}% vs 2011"
elif year_opt == "2011":
    delta = "first year"
else:
    delta = None

col1, col2, col3 = st.columns(3)
col1.metric("Total rides", f"{total / 1e6:,.2f}M", delta)
col2.metric("Peak-hour demand", f"{peak_val:.0f} rides/hr", f"{peak_hr}:00, working day")
col3.metric("Registered riders", f"{reg_share:.0f}%")

st.write("")
st.write("**Why these three?** Total rides = the scale of operation; peak-hour "
         "demand = the number maintenance must never starve; registered share = "
         "who the peaks serve (commuters, not tourists).")

# ---------------------------------------------------------------------------
# Chart 1 — hourly demand, working day vs weekend (WHY the peak matters).
# ---------------------------------------------------------------------------
st.subheader("When do people actually ride?")
wd = h[h["workingday"] == 1].groupby("hr")["cnt"].mean()
we = h[h["workingday"] == 0].groupby("hr")["cnt"].mean()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(wd.index, wd.values, color=COMMUTE, linewidth=2, label="Working day")
ax.plot(we.index, we.values, color=LEISURE, linewidth=2, label="Weekend / holiday")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Average rides per hour")
ax.set_xticks(range(0, 24, 2))
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend(frameon=False)
st.pyplot(fig)

st.caption(
    "Rides on a working day are a double spike — 8 a.m. and 5 p.m. — and near "
    "zero overnight. That is the window to take bikes off the street for "
    "maintenance: after ~8 p.m., before ~6 a.m."
)

# ---------------------------------------------------------------------------
# Chart 2 — monthly trend (WHEN to schedule heavy maintenance).
# ---------------------------------------------------------------------------
st.subheader("How does demand move through the year?")
monthly = d.groupby(["mnth", "yr"])["cnt"].sum().reset_index()
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(9, 4))
if year_opt == "Both years":
    for yr_val, color, label in [(0, YEAR_2011, "2011"), (1, YEAR_2012, "2012")]:
        m = monthly[monthly["yr"] == yr_val]
        ax.plot(m["mnth"], m["cnt"] / 1e3, color=color, linewidth=2, label=label)
    ax.legend(frameon=False)
else:
    ax.plot(monthly["mnth"], monthly["cnt"] / 1e3,
            color=YEAR_2012, linewidth=2, label=year_opt)
ax.set_xlabel("Month")
ax.set_ylabel("Total rides (thousands)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
st.pyplot(fig)

st.caption(
    "Demand is lowest in winter and spring and peaks in summer and fall — the "
    "low season is when to schedule the heavy, multi-day maintenance. The 2012 "
    "line also shows the fleet grew sharply year over year."
)

# ---------------------------------------------------------------------------
# Data period + intended production refresh (required).
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "**Data:** Jan 2011 – Dec 2012, Capital Bikeshare system (Kaggle snapshot). "
    "**Production refresh:** nightly from the trip log; reviewed weekly on "
    "Mondays. Extreme one-day drops (e.g. 22 rides on 2012-10-29) are weather "
    "events like Hurricane Sandy, not system outages."
)
