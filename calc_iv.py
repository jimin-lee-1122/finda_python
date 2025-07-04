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
    """S, tau가 같은 옵션끼리 그룹화하여 각각 calcbsimpvol을 호출하고 결과를 합칩니다."""
    results = []
    for (S, tau), group in df.groupby(['S', 'tau']):
        params = {
            "cp": group["cp"].to_numpy(),
            "P": group["P"].to_numpy(),
            "S": S,
            "K": group["Strike"].to_numpy(),
            "tau": tau,
            "r": r,
            "q": q,
        }
        print(f"[DEBUG] group S={S}, tau={tau}, size={len(group)}")
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}, first5={v[:5]}")
            else:
                print(f"  {k}: value={v}, type={type(v)}")
        iv = calcbsimpvol(params)
        group = group.copy()
        group["implied_vol"] = iv
        results.append(group)
    result_df = pd.concat(results, ignore_index=True)
    return result_df


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
