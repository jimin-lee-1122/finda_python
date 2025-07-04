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
    """Compute implied volatility for each option individually.

    ``calcbsimpvol`` expects scalar values for ``S``, ``tau``, ``r`` and ``q``.
    When these are shared across options, broadcasting can lead to errors.
    This function iterates over each option and calls ``calcbsimpvol`` one by one
    to avoid such issues.
    """

    ivs = []
    for idx, row in df.iterrows():
        params = {
            "cp": np.array([row["cp"]]),
            "P": np.array([row["P"]]),
            "S": row["S"],
            "K": np.array([row["Strike"]]),
            "tau": row["tau"],
            "r": r,
            "q": q,
        }
        try:
            iv = calcbsimpvol(params)[0]
        except Exception:
            iv = np.nan
        ivs.append(iv)

    df = df.copy()
    df["implied_vol"] = ivs
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


def main(file_path: str = None) -> None:
    # 여러 날의 데이터 파일을 모두 합쳐서 사용
    import glob
    import os
    data_dir = "data"
    csv_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]
    print(f"[DEBUG] CSV files to use: {csv_files}")
    df_list = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                df_list.append(df)
        except Exception as e:
            print(f"[SKIP] {f}: {e}")
    df_raw = pd.concat(df_list, ignore_index=True)
    df_prepared = prepare_data(df_raw)
    result = compute_implied_volatility(df_prepared)
    print(result[["Strike", "tau", "cp", "implied_vol"]].head())
    plot_iv_surface(result)
    result.to_csv("implied_volatility_results.csv", index=False)


if __name__ == "__main__":
    main()
