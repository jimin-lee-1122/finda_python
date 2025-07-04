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

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def implied_vol_optimized(price, S, K, T, r, option):
    if price <= 0.01 or S <= 0 or K <= 0 or T <= 0:
        return np.nan

    intrinsic = max(S - K, 0) if option.lower() == 'c' else max(K - S, 0)
    if price <= intrinsic:
        return np.nan

    sigma = 0.3
    for i in range(50):
        vega = bs_vega(S, K, T, r, sigma)
        if vega < 1e-6:
            print(f"[DEBUG] Vega too small at iter {i}, sigma={sigma:.4f}")
            return np.nan
        price_est = bs_price(S, K, T, r, sigma, option)
        diff = price_est - price
        if np.isnan(price_est) or abs(diff) > 10:
            print(f"[DEBUG] price_est abnormal at iter {i}, sigma={sigma:.4f}, diff={diff:.4f}")
            return np.nan
        if abs(diff) < 1e-4:
            return sigma
        if i % 10 == 0:
            print(f"[DEBUG] Iter {i} | sigma: {sigma:.4f} | diff: {diff:.4f} | vega: {vega:.4f}")
        sigma -= diff / vega
        sigma = np.clip(sigma, 0.001, 5.0)
    print(f"[DEBUG] Max iter reached, returning NaN")
    return np.nan


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """데이터 전처리 최적화"""
    df = df.copy()
    
    # 벡터화된 연산 사용
    df['Strike'] = df.apply(parse_strike, axis=1)
    df['TTM'] = df.apply(lambda row: compute_ttm(row['STD_DT'], row['EXR_DT']), axis=1)
    print("[DEBUG] Strike, TTM 생성 후 샘플:")
    print(df[['STK_CD', 'Strike', 'STD_DT', 'EXR_DT', 'TTM', 'SETL_PRC', 'BASE_CLPRC']].head(10))
    price_col = 'SETL_PRC'
    if price_col not in df.columns:
        if 'MID_PRC' in df.columns:
            price_col = 'MID_PRC'
        else:
            raise KeyError('SETL_PRC or MID_PRC column required')
    print(f"[DEBUG] dropna 전: {len(df)}")
    df = df.dropna(subset=['Strike', 'TTM', price_col, 'BASE_CLPRC'])
    print(f"[DEBUG] dropna 후: {len(df)}")
    df['Price'] = df[price_col].astype(float)
    df['S'] = df['BASE_CLPRC'].astype(float)
    
    # 비현실적인 값들 필터링
    df = df[(df['Price'] > 0) & (df['S'] > 0) & (df['Strike'] > 0) & (df['TTM'] > 0)]
    
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


def plot_surface(df: pd.DataFrame, ax: Optional[plt.Axes] = None, label: str = 'Call') -> None:
    print(f"[INFO] 유효한 IV 데이터 수: {df['IV'].notna().sum()}")
    sns.histplot(df['IV'].dropna(), bins=40, kde=True)
    plt.title(f"Implied Volatility Distribution ({label})")
    plt.show()

    pivot = df.pivot_table(index='Strike', columns='TTM', values='IV', aggfunc='mean')
    if pivot.empty:
        print("[WARN] IV pivot 테이블이 비어 있습니다. 시각화 생략.")
        return
    X, Y = np.meshgrid(pivot.index.values, pivot.columns.values)
    Z = pivot.values.T
    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.7)
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
        plot_surface(call_iv, ax=ax, label='Call')
        plot_surface(put_iv, ax=ax, label='Put')
        plt.legend()
        out_df = pd.concat([call_iv, put_iv])
        save_output(out_df, args.output)
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
        plot_surface(iv_df, label='Put' if args.put else 'Call')
        save_output(iv_df, args.output)

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