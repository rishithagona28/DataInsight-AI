"""
Generates sample_data/retail_sales.csv: a realistic, synthetic retail dataset
for trying DataInsight AI without your own file.

Run:  python sample_data/make_sample.py

What's baked in (so the dashboard, forecasts and insights have a story to find):
  - overall growth over time, plus seasonality (strong around June, weak around October)
  - regions growing at different speeds (South fastest, West shrinking)
  - an Online channel that is gaining share
  - realistic noise, a few missing values and a handful of duplicate rows to clean
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2025)
N = 2000
START = pd.Timestamp("2024-01-01")
DAYS = 608  # Jan 2024 .. end of Aug 2025

day = np.sort(rng.integers(0, DAYS, N))
order_date = START + pd.to_timedelta(day, unit="D")
t = day / DAYS  # 0 -> 1 across the period

regions = np.array(["North", "South", "East", "West"])
region = rng.choice(regions, N, p=[0.32, 0.28, 0.24, 0.16])
region_growth = pd.Series(region).map({"North": 0.35, "South": 0.9, "East": 0.45, "West": -0.25}).to_numpy()

products = {"Laptop": 55000, "Phone": 25000, "Tablet": 18000, "Monitor": 12000, "Headphones": 3500, "Keyboard": 1500}
product = rng.choice(list(products), N, p=[0.18, 0.24, 0.14, 0.14, 0.17, 0.13])
unit_price = pd.Series(product).map(products).to_numpy() * rng.normal(1, 0.04, N)

# Online grows from 25% to 50% of orders over the period
p_online = 0.25 + 0.25 * t
channel = np.where(rng.random(N) < p_online, "Online", rng.choice(["Store", "Distributor"], N, p=[0.6, 0.4]))

units = rng.integers(1, 9, N)
season = 1 + 0.22 * np.sin((order_date.month.to_numpy() - 3) / 12 * 2 * np.pi)
revenue = units * unit_price * season * (1 + region_growth * t) * rng.normal(1, 0.08, N)

df = pd.DataFrame({
    "order_id": [f"ORD{100000 + i}" for i in range(N)],
    "order_date": order_date.strftime("%Y-%m-%d"),
    "region": region,
    "channel": channel,
    "product": product,
    "units": units,
    "unit_price": unit_price.round(0),
    "revenue": revenue.round(0),
    "customer_rating": np.clip(rng.normal(4.1, 0.6, N), 1, 5).round(1),
})

# A little real-world mess for the cleaning tab
df.loc[rng.choice(N, 25, replace=False), "customer_rating"] = np.nan
df = pd.concat([df, df.sample(6, random_state=1)]).sort_values("order_date").reset_index(drop=True)

out = Path(__file__).with_name("retail_sales.csv")
df.to_csv(out, index=False)
print(f"Wrote {len(df):,} rows to {out}")
