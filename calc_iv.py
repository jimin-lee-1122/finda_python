import pandas as pd
import numpy as np
from calcbsimpvol import calcbsimpvol
import matplotlib.pyplot as plt
from datetime import datetime


def parse_strike_from_stk_cd(stk_cd: str) -> float:
    """Extract strike price from option code string."""
    import re
    match = re.search(r"[CP](\d+)", stk_cd)
    if match:
        return float(match.group(1))
    return np.nan


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe columns for implied volatility calculation."""
    df = df.copy()
    df["Strike"] = df["STK_CD"].apply(parse_strike_from_stk_cd)
    df["S"] = df["BASE_CLPRC"].astype(float)
    df["tau"] = (
        pd.to_datetime(df["EXR_DT"], format="%Y%m%d")
        - pd.to_datetime(df["STD_DT"], format="%Y%m%d")
    ).dt.days / 365
    df["cp"] = df["STK_TP_CD"].apply(lambda x: 1 if x.upper() == "C" else -1)
    df["P"] = df["STK_CLPRC"].astype(float)
    return df.dropna(subset=["Strike"])


def compute_implied_volatility(df: pd.DataFrame, r: float = 0.05, q: float = 0.0) -> pd.DataFrame:
    """Compute implied volatility using calcbsimpvol."""
    params = {
        "cp": df["cp"].to_numpy(),
        "P": df["P"].to_numpy(),
        "S": df["S"].to_numpy(),
        "K": df["Strike"].to_numpy(),
        "tau": df["tau"].to_numpy(),
        "r": r,
        "q": q,
    }
    iv = calcbsimpvol(params)
    df = df.copy()
    df["implied_vol"] = iv
    return df


def plot_iv_surface(df: pd.DataFrame) -> None:
    """Plot implied volatility surface."""
    surface = df.pivot_table(index="Strike", columns="tau", values="implied_vol", aggfunc="mean")
    plt.imshow(surface.values, aspect="auto", origin="lower", cmap="viridis")
    plt.xticks(range(len(surface.columns)), [f"{x:.2f}" for x in surface.columns])
    plt.yticks(range(len(surface.index)), [f"{x:.0f}" for x in surface.index])
    plt.xlabel("Time to Expiry (years)")
    plt.ylabel("Strike Price")
    plt.colorbar(label="Implied Volatility")
    plt.title("Implied Volatility Surface")
    plt.tight_layout()
    plt.show()


def main(file_path: str = "data/sp500_raw_data_20240103.csv") -> None:
    df_raw = pd.read_csv(file_path)
    df_prepared = prepare_data(df_raw)
    result = compute_implied_volatility(df_prepared)
    print(result[["STK_CD", "implied_vol"]].head())
    plot_iv_surface(result)
    result.to_csv("implied_volatility_results.csv", index=False)


if __name__ == "__main__":
    main()
