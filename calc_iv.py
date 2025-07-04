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
    """Compute implied volatility using calcbsimpvol with meshgrid surface."""
    # 1. unique strike, tau 추출
    strikes = np.sort(df["Strike"].unique())
    taus = np.sort(df["tau"].unique())
    S = df["S"].iloc[0]
    # 2. meshgrid 생성
    K_grid, tau_grid = np.meshgrid(strikes, taus)
    S_grid = np.full_like(K_grid, S, dtype=float)
    r_grid = np.full_like(K_grid, r, dtype=float)
    q_grid = np.full_like(K_grid, q, dtype=float)
    # 3. cp, P 2D surface 생성 (mean aggregation)
    cp_grid = np.full_like(K_grid, 1, dtype=int)  # 콜 옵션만 우선 예시
    P_grid = np.full_like(K_grid, np.nan, dtype=float)
    # 각 (tau, strike)에 해당하는 평균 P값 채우기
    for i, tau_val in enumerate(taus):
        for j, strike_val in enumerate(strikes):
            mask = (df["tau"] == tau_val) & (df["Strike"] == strike_val) & (df["cp"] == 1)
            if mask.any():
                P_grid[i, j] = df.loc[mask, "P"].mean()
    # 4. calcbsimpvol 호출
    params = dict(cp=cp_grid, P=P_grid, S=S_grid, K=K_grid, tau=tau_grid, r=r_grid, q=q_grid)
    print("[DEBUG] meshgrid shapes:")
    for k, v in params.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, sample={v.flatten()[:5]}")
    iv_grid = calcbsimpvol(params)
    # 5. 결과를 DataFrame으로 변환
    result = []
    for i, tau_val in enumerate(taus):
        for j, strike_val in enumerate(strikes):
            if not np.isnan(P_grid[i, j]):
                result.append({
                    "Strike": strike_val,
                    "tau": tau_val,
                    "S": S,
                    "P": P_grid[i, j],
                    "cp": 1,
                    "implied_vol": iv_grid[i, j],
                })
    return pd.DataFrame(result)


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
