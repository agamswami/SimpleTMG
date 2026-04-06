import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and set index explicitly to datetime
df = pd.read_csv("2019Floor6.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
df.index = pd.to_datetime(df.index, errors="coerce")

# Generate zonal totals and floor total
zones = [1, 2, 3, 4, 5]
for z in zones:
    kw_cols = [c for c in df.columns if c.startswith(f"z{z}_") and "(kW)" in c]
    if kw_cols:
        df[f"z{z}_Total(kW)"] = df[kw_cols].sum(axis=1, skipna=False)

all_kw_cols = [c for c in df.columns if "(kW)" in c and "Total" not in c]
df["Floor_Total(kW)"] = df[all_kw_cols].sum(axis=1, skipna=False)

# Average temperature and lux for the floor
temp_cols = [c for c in df.columns if "degC" in c]
df["Floor_Mean_Temp"] = df[temp_cols].mean(axis=1, skipna=False)

lux_cols = [c for c in df.columns if "lux" in c]
df["Floor_Mean_Lux"] = df[lux_cols].mean(axis=1, skipna=False)

# 1. Missing Data Handling
# Forward-fill and interpolate
df_clean = df.interpolate(method="linear", limit=60)
df_clean.ffill(inplace=True)
df_clean.bfill(inplace=True)

# Resample to Hourly
df_hourly = df_clean.resample("h").mean()

# 2. EDA: Cross-Correlation Heatmap (Channel Independence Check)
zonal_totals = [
    f"z{z}_Total(kW)" for z in zones if f"z{z}_Total(kW)" in df_hourly.columns
]
corr_cols = zonal_totals + ["Floor_Mean_Temp", "Floor_Mean_Lux"]
corr_matrix = df_hourly[corr_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Cross-Correlation Heatmap (Hourly Resampled)")
plt.tight_layout()
plt.savefig("cross_correlation.png")
plt.show()

# 3. EDA: Nonlinearity Check (Temperature vs Energy)
plt.figure(figsize=(8, 6))
plt.scatter(df_hourly["Floor_Mean_Temp"], df_hourly["Floor_Total(kW)"], alpha=0.3, s=5)
plt.title("Floor Mean Temperature vs Floor Total Energy (Hourly)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Total Energy Consumption (kW)")
plt.grid(True)
plt.tight_layout()
plt.savefig("nonlinearity_temp_energy.png")
plt.show()

# 4. EDA: Seasonality / Periodic Profile Check
plt.figure(figsize=(12, 4))
df_hourly["Floor_Total(kW)"].loc["2019-03-04":"2019-03-10"].plot()
plt.title("Floor Total Energy Consumption (One Week in March 2019)")
plt.ylabel("Total Energy (kW)")
plt.grid(True)
plt.tight_layout()
plt.savefig("weekly_profile.png")
plt.show()

# Evaluate output shapes and NaNs
print("Cleaned Hourly Shape:", df_hourly.shape)
print(
    "NaNs after handling:\n",
    df_hourly[["Floor_Total(kW)", "Floor_Mean_Temp"]].isna().sum(),
)