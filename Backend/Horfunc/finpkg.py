import numpy as np


def simulate_price(S0, vol_annual, days, seed=None, basis=252):
    """
    用GBM模拟未来合约价格（单路径）。

    参数:
    S0 : float
        初始价格（当前市场价格）
    vol_annual : float
        年化波动率，例如 0.3 表示 30%
    days : int
        要模拟的交易日数
    seed : int, 可选
        随机种子（用于可重复结果）
    basis : int, 默认252
        年化基数，常用252表示交易日

    返回:
    float : 在 days 天后的模拟价格
    """
    if seed is not None:
        np.random.seed(seed)

    dt = days / basis
    sigma = vol_annual
    # 生成标准正态随机数
    Z = np.random.normal()
    # GBM公式：S_T = S0 * exp(-0.5σ²T + σ√T * Z)
    ST = S0 * np.exp(-0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * Z)
    return ST


# 示例
# price = simulate_price(S0=100, vol_annual=0.3, days=21, seed=42)
# print(price)



def simulate_price(S0, vol_annual, T, seed=None, basis=252):
    """
    模拟到期价格（单次GBM路径）
    """
    if seed is not None:
        np.random.seed(seed)
    Z = np.random.normal()
    ST = S0 * np.exp(-0.5 * vol_annual**2 * T + vol_annual * np.sqrt(T) * Z)
    return ST

# 读取Excel
df = pd.read_excel("PFE_Results.xlsx", sheet_name="PFE Results")

# 取需要的列，注意列名要和Excel里一致
df["GBM Price"] = df.apply(
    lambda row: simulate_price(
        row["Contract Price (USD/MT)"],     # 当前价格
        row["Ann Volatility (%)"] / 100.0,  # 波动率转小数
        row["Time to Expiry"]               # 年数
    ),
    axis=1
)

print(df[["Contract Price (USD/MT)", "Ann Volatility (%)", "Time to Expiry", "GBM Price"]])
