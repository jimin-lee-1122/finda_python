import argparse
import pandas as pd
import numpy as np
from dataclasses import dataclass
from math import log, sqrt, exp
from typing import Optional

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- needed for 3D plotting
from scipy.stats import norm


def parse_strike(row: pd.Series) -> float:
    if 'STRIKE' in row and not pd.isna(row['STRIKE']):
        return float(row['STRIKE'])
    if 'STK_CD' in row and isinstance(row['STK_CD'], str):
        import re
        m = re.search(r"(\d+)$", row['STK_CD'])
        if m:
            return float(m.group(1)) / 1000.0  # assume last digits represent strike
    return np.nan


def compute_ttm(std_dt: str, exr_dt: str) -> float:
    std = pd.to_datetime(std_dt, format="%Y%m%d")
    exr = pd.to_datetime(exr_dt, format="%Y%m%d")
    return (exr - std).days / 365.0


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option: str) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if option.lower() == 'c':
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return S * norm.pdf(d1) * sqrt(T)


def implied_vol(price: float, S: float, K: float, T: float, r: float, option: str) -> float:
    if price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    sigma = 0.2
    for _ in range(100):
        price_est = bs_price(S, K, T, r, sigma, option)
        vega = bs_vega(S, K, T, r, sigma)
        if vega == 0:
            break
        diff = price_est - price
        if abs(diff) < 1e-6:
            return sigma
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 1e-4
    return np.nan


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Strike'] = df.apply(parse_strike, axis=1)
    df['TTM'] = df.apply(lambda row: compute_ttm(row['STD_DT'], row['EXR_DT']), axis=1)
    df = df.dropna(subset=['Strike', 'TTM', 'SETL_PRC', 'BASE_CLPRC'])
    df['Price'] = df['SETL_PRC'].astype(float)
    df['S'] = df['BASE_CLPRC'].astype(float)
    return df


def compute_iv(df: pd.DataFrame, option: str, r: float = 0.03) -> pd.DataFrame:
    ivs = []
    for _, row in df.iterrows():
        iv = implied_vol(row['Price'], row['S'], row['Strike'], row['TTM'], r, option)
        ivs.append(iv)
    df = df.copy()
    df['IV'] = ivs
    return df.dropna(subset=['IV'])


def plot_surface(df: pd.DataFrame, ax: Optional[plt.Axes] = None, label: str = 'Call') -> None:
    pivot = df.pivot_table(index='Strike', columns='TTM', values='IV', aggfunc='mean')
    X, Y = np.meshgrid(pivot.index.values, pivot.columns.values)
    Z = pivot.values.T
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7, label=label)
    ax.set_xlabel('Strike')
    ax.set_ylabel('TTM')
    ax.set_zlabel('IV')
    ax.set_title(f'IV Surface ({label})')


def save_output(df: pd.DataFrame, path: str) -> None:
    """Save IV data to CSV or Excel depending on the file extension."""
    if path.lower().endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Compute IV surface from option data")
    parser.add_argument("csv", help="CSV file with option data")
    parser.add_argument("--output", "-o", default="iv_surface.csv", help="path to save calculated IV data (CSV)")
    parser.add_argument("--put", action="store_true", help="use put options instead of call")
    parser.add_argument("--compare", action="store_true", help="compare call and put surfaces")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.compare:
        call_df = prepare(df[df['STK_TP_CD'].str.upper() == 'C'])
        put_df = prepare(df[df['STK_TP_CD'].str.upper() == 'P'])
        call_iv = compute_iv(call_df, 'c')
        call_iv['Option'] = 'Call'
        put_iv = compute_iv(put_df, 'p')
        put_iv['Option'] = 'Put'
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        plot_surface(call_iv, ax=ax, label='Call')
        plot_surface(put_iv, ax=ax, label='Put')
        plt.legend()
        save_output(pd.concat([call_iv, put_iv]), args.output)
    else:
        opt_type = 'p' if args.put else 'c'
        df_opt = df[df['STK_TP_CD'].str.upper() == ('P' if args.put else 'C')]
        df_opt = prepare(df_opt)
        iv_df = compute_iv(df_opt, opt_type)
        iv_df['Option'] = 'Put' if args.put else 'Call'
        plot_surface(iv_df, label='Put' if args.put else 'Call')
        save_output(iv_df, args.output)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()