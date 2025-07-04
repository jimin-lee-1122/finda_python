import argparse
import logging
from math import exp, log, sqrt
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- needed for 3D plotting
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
import seaborn as sns
from scipy.optimize import curve_fit

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", filename='iv_debug.log', filemode='w')
logger = logging.getLogger(__name__)


def parse_strike(row: pd.Series) -> float:
    if 'STRIKE' in row and not pd.isna(row['STRIKE']):
        return float(row['STRIKE'])
    if 'STK_CD' in row and isinstance(row['STK_CD'], str):
        import re
        m = re.search(r"[CP](\d+)", row['STK_CD'])
        if m:
            return float(m.group(1))
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


def implied_vol_optimized(price, S, K, T, r, option):
    if price <= 0.001 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    # 시간가치가 너무 작으면 NaN 처리 (조건 완화)
    intrinsic = max(S - K, 0) if option.lower() == 'c' else max(K - S, 0)
    if price <= intrinsic + 1e-3:
        return np.nan

    sigma = 0.3  # 초기 추정값
    for _ in range(50):
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-8:  # Vega가 너무 작으면 빠르게 종료 (더 낮게 허용)
            return np.nan
        price_est = bs_price(S, K, T, r, sigma, option)
        diff = price_est - price
        if abs(diff) < 1e-4:
            return sigma
        sigma -= diff / vega
        sigma = np.clip(sigma, 0.001, 5.0)
    return np.nan


def svi_raw(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def fit_svi_for_ttm(df_ttm):
    F = df_ttm['S'].iloc[0] * np.exp(0.03 * df_ttm['TTM'].iloc[0])
    k = np.log(df_ttm['Strike'] / F)
    w = (df_ttm['IV'] ** 2) * df_ttm['TTM']
    p0 = [0.01, 0.1, 0.0, 0.0, 0.1]
    try:
        popt, _ = curve_fit(svi_raw, k, w, p0=p0, maxfev=10000)
        return popt
    except Exception:
        return None

def svi_surface(df):
    svi_iv = []
    for ttm, group in df.groupby('TTM'):
        popt = fit_svi_for_ttm(group)
        if popt is None:
            svi_iv.extend([np.nan]*len(group))
            continue
        F = group['S'].iloc[0] * np.exp(0.03 * ttm)
        k = np.log(group['Strike'] / F)
        w_svi = svi_raw(k, *popt)
        iv_svi = np.sqrt(np.maximum(w_svi / ttm, 0))
        svi_iv.extend(iv_svi)
    df = df.copy()
    df['IV_SVI'] = svi_iv
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 전처리 최적화"""
    df = df.copy()
    
    # 벡터화된 연산 사용
    df['Strike'] = df.apply(parse_strike, axis=1)
    df['TTM'] = df.apply(lambda row: compute_ttm(row['STD_DT'], row['EXR_DT']), axis=1)
    logger.debug("[DEBUG] Strike, TTM 생성 후 샘플:")
    logger.debug(df[['STK_CD', 'Strike', 'STD_DT', 'EXR_DT', 'TTM', 'SETL_PRC', 'BASE_CLPRC', 'STK_TP_CD']].head(10).to_string())
    price_col = 'SETL_PRC'
    if price_col not in df.columns:
        if 'MID_PRC' in df.columns:
            price_col = 'MID_PRC'
        else:
            raise KeyError('SETL_PRC or MID_PRC column required')
    logger.debug(f"[DEBUG] dropna 전: {len(df)}")
    df = df.dropna(subset=['Strike', 'TTM', price_col, 'BASE_CLPRC'])
    logger.debug(f"[DEBUG] dropna 후: {len(df)}")
    logger.debug("[DEBUG] dropna 후 샘플:")
    logger.debug(df[['STK_CD', 'Strike', 'TTM', price_col, 'BASE_CLPRC', 'STK_TP_CD']].head(10).to_string())
    df['Price'] = df[price_col].astype(float)
    df['S'] = df['BASE_CLPRC'].astype(float)
    
    # 비현실적인 값들 필터링
    df = df[(df['Price'] > 0) & (df['S'] > 0) & (df['Strike'] > 0) & (df['TTM'] > 0)]
    logger.debug(f"[DEBUG] 최종 필터 후: {len(df)}")
    logger.debug("[DEBUG] 최종 필터 후 샘플:")
    logger.debug(df[['STK_CD', 'Strike', 'TTM', 'Price', 'S', 'STK_TP_CD']].head(10).to_string())
    
    return df


def compute_iv_vectorized(df: pd.DataFrame, option: str, r: float = 0.03, n_jobs: int = 4) -> pd.DataFrame:
    """병렬 처리를 통한 IV 계산"""
    def calc_iv_row(row):
        return implied_vol_optimized(row['Price'], row['S'], row['Strike'], row['TTM'], r, option)
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        ivs = list(executor.map(calc_iv_row, [row for _, row in df.iterrows()]))
    
    df = df.copy()
    df['IV'] = ivs
    return df.dropna(subset=['IV'])


def compute_iv(
    df: pd.DataFrame,
    option: str,
    r: float = 0.03,
    show_progress: bool = False,
) -> pd.DataFrame:
    """기존 순차 처리 방식 (최적화된 IV 계산 함수 사용)"""
    ivs = []
    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Calculating IV", leave=False)
    
    for _, row in iterator:
        iv = implied_vol_optimized(row['Price'], row['S'], row['Strike'], row['TTM'], r, option)
        ivs.append(iv)
    
    df = df.copy()
    df['IV'] = ivs
    return df.dropna(subset=['IV'])


def plot_surface(df: pd.DataFrame, ax: Optional[plt.Axes] = None, label: str = 'Call', iv_col: str = 'IV') -> None:
    pivot = df.pivot_table(index='Strike', columns='TTM', values=iv_col, aggfunc='mean')
    if pivot.empty:
        print(f"[WARN] {iv_col} 피벗 결과가 비어 있습니다. 스킕합니다.")
        return

    X, Y = np.meshgrid(pivot.index.values, pivot.columns.values)
    Z = pivot.values.T

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.set_xlabel('Strike')
    ax.set_ylabel('TTM')
    ax.set_zlabel(iv_col)
    ax.set_title(f'{iv_col} Surface - {label}')
    plt.show()


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
    parser.add_argument("--verbose", "-v", action="store_true", help="show progress information")
    parser.add_argument("--parallel", action="store_true", help="use parallel processing")
    parser.add_argument("--jobs", type=int, default=4, help="number of parallel jobs")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="number of rows to sample for quicker testing",
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    logger.info("Reading input CSV %s", args.csv)
    start_time = time.time()

    df = pd.read_csv(args.csv)
    
    if args.sample:
        logger.info("Sampling %d rows for quick processing", args.sample)
        df = df.sample(n=min(args.sample, len(df)), random_state=1)
    
    # 중복 샘플링 제거 (기존 코드의 df.head(100) 제거)

    if args.compare:
        logger.info("Preparing call and put data for comparison")
        call_df = prepare(df[df['STK_TP_CD'].str.upper() == 'C'])
        put_df = prepare(df[df['STK_TP_CD'].str.upper() == 'P'])
        
        # 샘플 데이터 5개와 IV 계산 결과 출력
        print("[샘플] Call 옵션 데이터 5개:")
        print(call_df[['Price', 'S', 'Strike', 'TTM']].head(5))
        for _, row in call_df.head(5).iterrows():
            print(implied_vol_optimized(row['Price'], row['S'], row['Strike'], row['TTM'], r=0.03, option='c'))
        
        compute_func = compute_iv_vectorized if args.parallel else compute_iv
        
        if args.parallel:
            call_iv = compute_func(call_df, 'c', r=0.03, n_jobs=args.jobs)
            put_iv = compute_func(put_df, 'p', r=0.03, n_jobs=args.jobs)
        else:
            call_iv = compute_func(call_df, 'c', show_progress=args.verbose)
            put_iv = compute_func(put_df, 'p', show_progress=args.verbose)
        
        call_iv['Option'] = 'Call'
        put_iv['Option'] = 'Put'
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        plot_surface(call_iv, ax=ax, label='Call', iv_col='IV')
        plot_surface(put_iv, ax=ax, label='Put', iv_col='IV')
        plt.legend()
        out_df = pd.concat([call_iv, put_iv])
        save_output(out_df, args.output)
        # SVI 보정 surface (콜/풋 각각)
        call_iv_svi = svi_surface(call_iv)
        put_iv_svi = svi_surface(put_iv)
        fig2 = plt.figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111, projection='3d')
        plot_surface(call_iv_svi, ax=ax2, label='Call SVI', iv_col='IV_SVI')
        plot_surface(put_iv_svi, ax=ax2, label='Put SVI', iv_col='IV_SVI')
        plt.legend()
    else:
        opt_type = 'p' if args.put else 'c'
        logger.info("Preparing %s option data", "put" if args.put else "call")
        df_opt = df[df['STK_TP_CD'].str.upper() == ('P' if args.put else 'C')]
        df_opt = prepare(df_opt)
        
        # 샘플 데이터 5개와 IV 계산 결과 출력
        print("[샘플] 옵션 데이터 5개:")
        print(df_opt[['Price', 'S', 'Strike', 'TTM', 'STK_TP_CD']].head(5))

        for _, row in df_opt.head(5).iterrows():
            print(implied_vol_optimized(row['Price'], row['S'], row['Strike'], row['TTM'], r=0.03, option='c'))
        
        compute_func = compute_iv_vectorized if args.parallel else compute_iv
        
        if args.parallel:
            iv_df = compute_func(df_opt, opt_type, r=0.03, n_jobs=args.jobs)
        else:
            iv_df = compute_func(df_opt, opt_type, show_progress=args.verbose)
        iv_df['Option'] = 'Put' if args.put else 'Call'
        plot_surface(iv_df, label='Put' if args.put else 'Call', iv_col='IV')
        save_output(iv_df, args.output)
        # SVI 보정 surface
        iv_df_svi = svi_surface(iv_df)
        plot_surface(iv_df_svi, label='SVI Fitted', iv_col='IV_SVI')

    end_time = time.time()
    logger.info("Computation completed in %.2f seconds", end_time - start_time)
    logger.info("Saving results to %s", args.output)
    plt.tight_layout()
    plt.show()

    print(f"총 옵션 수: {len(df_opt)}")
    print(f"IV 계산 후 유효한 데이터 수: {len(iv_df)}")
    print(f"IV 값 예시: {iv_df['IV'].describe()}")

    sns.histplot(iv_df['IV'].dropna(), bins=50, kde=True)
    plt.title("Implied Volatility Distribution")
    plt.show()


if __name__ == '__main__':
    main()