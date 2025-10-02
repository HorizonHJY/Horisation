import numpy as np
import pandas as pd


def simulate_price(S0, vol_annual, days, seed=None, basis=252):
    """
    """
    if seed is not None:
        np.random.seed(seed)

    dt = days / basis
    sigma = vol_annual
    Z = np.random.normal()
    # GBM公式：S_T = S0 * exp(-0.5σ²T + σ√T * Z)
    ST = S0 * np.exp(-0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * Z)
    return ST


def simulate_price(S0, vol_annual, T, seed=None, basis=252):
    """
    模拟到期价格（单次GBM路径）
    """
    if seed is not None:
        np.random.seed(seed)
    Z = np.random.normal()
    ST = S0 * np.exp(-0.5 * vol_annual**2 * T + vol_annual * np.sqrt(T) * Z)
    return ST

df = pd.read_excel("PFE_Results.xlsx", sheet_name="PFE Results")
df["GBM Price"] = df.apply(
    lambda row: simulate_price(
        row["Contract Price (USD/MT)"],     # 当前价格
        row["Ann Volatility (%)"] / 100.0,  # 波动率转小数
        row["Time to Expiry"]               # 年数
    ),
    axis=1
)

print(df[["Contract Price (USD/MT)", "Ann Volatility (%)", "Time to Expiry", "GBM Price"]])
