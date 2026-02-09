This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: scripts/**/*
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
scripts/
  __init__.py
  00_integrity_check.py
  01_setup_database.py
  analyze_intraday_hourly.py
  analyze_intraday_patterns.py
  analyze_intraday_pnl_vs_exposure.py
  analyze_live_portfolio_equity.py
  analyze_live_snapshots.py
  backfill_minute1_history.py
  backtest_live_portfolio_1d.py
  backtest_scalping_1m_ethusdt.py
  build_today_signals.py
  compare_strategy_vs_hodl.py
  compute_excess_exposure.py
  daily_intraday_report.py
  download_binance_1m_history.py
  download_multitf_history.py
  generate_signals_1d.py
  gridsearch_1d_signals.py
  gridsearch_scalping_1m_ethusdt.py
  label_regimes_from_ohlcv.py
  live_trader_ml.py
  live_trader_ml.py.backup.20260209_113012
  multi_best_1d_trades.py
  order_utils.py
  orderutils.py
  param_sweep_range_bt.py
  param_sweep_range_multi.py
  portfolio_manager_example.py
  robust_backtest.py
  run_realtime_trading_conf.py
  run_realtime_trading.py
  run_scheme_comparison.py
  scan_volume_spikes_1d.py
  show_volume_spikes_today.py
  simulate_ml_performance_1d.py
  simulate_ml_performance.py
  start_trading_daemon.sh
  summarize_relative_down.py
  train_all_models.py
  train_relative_down_model.py
```

# Files

## File: scripts/__init__.py
```python

```

## File: scripts/00_integrity_check.py
```python
import sys
import os
import logging
import importlib

try:
    import yaml
except ImportError:
    yaml = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("INTEGRITY_CHECK")


def check_python_version() -> bool:
    v = sys.version_info
    logger.info(f"Python Version: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 12):
        logger.error("❌ Python 3.12+ Required")
        return False
    logger.info("✅ Python Version OK")
    return True


def check_config() -> bool:
    ok = True

    # 1) config.yaml 은 필수
    main_cfg = os.path.join(ROOT_DIR, "config", "config.yaml")
    if not os.path.exists(main_cfg):
        logger.error(f"❌ Missing REQUIRED config file: {main_cfg}")
        ok = False
    else:
        logger.info(f"✅ Found REQUIRED config: {main_cfg}")
        if yaml is not None:
            try:
                with open(main_cfg, "r") as f:
                    _ = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"❌ Config parse error ({main_cfg}): {e}")
                ok = False

    # 2) liveconfig.yaml 은 선택 (있으면 검사, 없으면 경고만)
    live_cfg = os.path.join(ROOT_DIR, "config", "liveconfig.yaml")
    if os.path.exists(live_cfg):
        logger.info(f"✅ Found OPTIONAL live config: {live_cfg}")
        if yaml is not None:
            try:
                with open(live_cfg, "r") as f:
                    _ = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"❌ Live config parse error ({live_cfg}): {e}")
                ok = False
    else:
        logger.info("ℹ️ OPTIONAL liveconfig.yaml not found (skipping)")

    if ok:
        logger.info("✅ Config check OK")
    return ok


def check_models() -> bool:
    required = [
        os.path.join(ROOT_DIR, "models", "catboost_model.cbm"),
    ]
    all_exist = True
    for p in required:
        if not os.path.exists(p):
            logger.error(f"❌ Missing model file: {p}")
            all_exist = False
        else:
            logger.info(f"✅ Found model: {p}")
    return all_exist


def check_imports() -> bool:
    modules = [
        "collectors.upbit_collector",
        "features.technical_indicators",
        "models.catboost_model",
        "strategy.kelly_sizing",
        "strategy.risk_manager",
        "scripts.simulate_ml_performance",
        "scripts.backtest_by_regime",
    ]
    all_good = True
    for m in modules:
        try:
            importlib.import_module(m)
            logger.info(f"✅ Import OK: {m}")
        except Exception as e:
            logger.error(f"❌ Import failed {m}: {e}")
            all_good = False
    return all_good


def main() -> None:
    logger.info("⚡ STARTING SYSTEM INTEGRITY CHECK ⚡")

    results = [
        check_python_version(),
        check_config(),
        check_models(),
        check_imports(),
    ]

    if all(results):
        logger.info("🎉 INTEGRITY CHECK PASSED - SYSTEM READY")
        sys.exit(0)
    else:
        logger.error("⛔ INTEGRITY CHECK FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

## File: scripts/01_setup_database.py
```python
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import psycopg2
import logging
from database.connection import get_db_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_database():
    config = get_db_config()
    logger.info(f"Connecting to {config['database']}@{config['host']}...")
    try:
        conn = psycopg2.connect(**config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        schema_path = project_root / 'database' / 'schema.sql'
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        logger.info("Executing schema.sql...")
        statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
        
        for stmt in statements:
            if stmt and not stmt.startswith('--'):
                try:
                    cursor.execute(stmt)
                except Exception as e:
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Warning: {e}")
        
        logger.info("✅ Schema execution completed!")
        
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;")
        tables = cursor.fetchall()
        table_names = [t[0] for t in tables]
        logger.info(f"✅ Tables ({len(table_names)}):")
        for name in table_names:
            logger.info(f"   - {name}")
        
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("  Database Initialization")
    print("=" * 50)
    success = setup_database()
    print("=" * 50)
    if success:
        print("✅ Setup completed!")
    else:
        print("❌ Setup failed!")
    print("=" * 50)
    sys.exit(0 if success else 1)
```

## File: scripts/analyze_intraday_hourly.py
```python
import pandas as pd
from pathlib import Path

def main():
    path = Path("reports/live_trader_snapshots.csv")
    if not path.exists():
        print("[ERROR] reports/live_trader_snapshots.csv 가 없습니다.")
        return

    df = pd.read_csv(path)
    # ts를 datetime으로 변환 + 시간대(hour) 컬럼 추가
    df["ts_dt"] = pd.to_datetime(df["ts"])
    df["hour"] = df["ts_dt"].dt.hour

    # 포지션 비중, 코인별 gap 재계산
    df["pos_val"] = df["KRW-ETH_val"] + df["KRW-DOGE_val"] + df["KRW-AVAX_val"]
    df["pos_frac"] = df["pos_val"] / df["equity_krw"].replace(0, pd.NA)

    for mkt in ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]:
        val_col = f"{mkt}_val"
        tgt_col = f"{mkt.replace('-', '_')}_target"
        diff_col = f"{mkt}_gap"
        df[diff_col] = df[tgt_col] - df[val_col]

    print("=== 전체 포지션 비중 통계 ===")
    print(df["pos_frac"].describe())

    # 시간대별 평균 포지션 비중 / gap
    grp = df.groupby("hour").agg({
        "pos_frac": "mean",
        "KRW-ETH_gap": "mean",
        "KRW-DOGE_gap": "mean",
        "KRW-AVAX_gap": "mean",
    }).reset_index()

    print("\\n=== 시간대별 평균 pos_frac / gap ===")
    print(grp)

    out = Path("reports/intraday_hourly_stats.csv")
    grp.to_csv(out, index=False)
    print(f"\\n[INFO] wrote {out}")

if __name__ == "__main__":
    main()
```

## File: scripts/analyze_intraday_patterns.py
```python
import pandas as pd
from pathlib import Path

def main():
    path = Path("reports/live_trader_snapshots.csv")
    if not path.exists():
        print("[ERROR] reports/live_trader_snapshots.csv 가 없습니다.")
        return

    df = pd.read_csv(path)
    print("=== snapshots shape ===", df.shape)
    print("=== time range ===")
    print(df["ts"].min(), " ~ ", df["ts"].max())

    # 1) 전체 구간에서 포지션 비중 분포
    df["pos_val"] = df["KRW-ETH_val"] + df["KRW-DOGE_val"] + df["KRW-AVAX_val"]
    df["pos_frac"] = df["pos_val"] / df["equity_krw"].replace(0, pd.NA)
    print("\n=== 포지션 비중 통계 (ETH+DOGE+AVAX / equity) ===")
    print(df["pos_frac"].describe())

    # 2) 코인별 target 대비 실제 평가금 갭
    for mkt in ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]:
        val_col = f"{mkt}_val"
        tgt_col = f"{mkt.replace('-', '_')}_target"
        diff_col = f"{mkt}_gap"
        df[diff_col] = df[tgt_col] - df[val_col]
        print(f"\n=== {mkt} target - current (gap) 통계 ===")
        print(df[diff_col].describe())

    # TODO:
    # - 시간대별(pos_frac, gap) 패턴
    # - 레짐 정보(BTC 방향 등) join
    # - gap가 클 때만 매매하는 룰 설계 등

if __name__ == "__main__":
    main()
```

## File: scripts/analyze_intraday_pnl_vs_exposure.py
```python
import pandas as pd
from pathlib import Path

def main():
    path = Path("reports/live_trader_snapshots.csv")
    if not path.exists():
        print("[ERROR] reports/live_trader_snapshots.csv 가 없습니다.")
        return

    df = pd.read_csv(path)

    # 기본 파생 컬럼
    df["pos_val"] = df["KRW-ETH_val"] + df["KRW-DOGE_val"] + df["KRW-AVAX_val"]
    df["pos_frac"] = df["pos_val"] / df["equity_krw"].replace(0, pd.NA)

    # 시간/정렬
    df["ts_dt"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts_dt").reset_index(drop=True)

    # 구간별 equity 변화 (다음 시점 - 현재 시점)
    df["equity_next"] = df["equity_krw"].shift(-1)
    df["dt_next"] = df["ts_dt"].shift(-1)
    df["equity_change"] = df["equity_next"] - df["equity_krw"]
    df["equity_ret"] = df["equity_change"] / df["equity_krw"].replace(0, pd.NA)

    # 구간별 평균 노출 (현재 pos_frac로 근사)
    # 필요하면 (pos_frac + pos_frac_next)/2 같은 형태로 바꿀 수 있음
    df["pos_frac_next"] = df["pos_frac"].shift(-1)
    df["pos_frac_mid"] = (df["pos_frac"] + df["pos_frac_next"]) / 2

    # 마지막 행은 다음 시점이 없으므로 제거
    df_seg = df.dropna(subset=["equity_ret", "pos_frac_mid"]).copy()

    print("=== 구간별 PnL vs 노출 일부 샘플 ===")
    print(df_seg[["ts", "equity_krw", "equity_next", "equity_ret", "pos_frac_mid"]].head())

    print("\n=== 전체 구간 PnL 통계 ===")
    print(df_seg["equity_ret"].describe())

    print("\n=== 노출 구간(pos_frac_mid) 통계 ===")
    print(df_seg["pos_frac_mid"].describe())

    # 노출 구간을 몇 개 버킷으로 나눠서, 버킷별 평균 수익률을 본다.
    bins = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.5]
    labels = ["<=30%", "30-50%", "50-70%", "70-80%", "80-90%", "90%+"]

    df_seg["pos_bucket"] = pd.cut(df_seg["pos_frac_mid"], bins=bins, labels=labels, include_lowest=True)

    grp = df_seg.groupby("pos_bucket").agg({
        "equity_ret": "mean",
        "pos_frac_mid": "count",
    }).rename(columns={"pos_frac_mid": "count"}).reset_index()

    print("\n=== 노출 버킷별 평균 PnL (equity_ret) ===")
    print(grp)

    out = Path("reports/intraday_pnl_vs_exposure.csv")
    grp.to_csv(out, index=False)
    print(f"\n[INFO] wrote {out}")

if __name__ == "__main__":
    main()
```

## File: scripts/analyze_live_portfolio_equity.py
```python
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
import sys
path = ROOT / "reports" / (sys.argv[1] if len(sys.argv) > 1 else "backtest_live_portfolio_1d_equity.csv")

df = pd.read_csv(path, parse_dates=["date"])
rets = df["daily_return"].values.astype(float)

n = len(rets)
mean_ret = rets.mean()
std_ret = rets.std(ddof=1) if n > 1 else 0.0
sharpe = mean_ret / (std_ret + 1e-10) * np.sqrt(252)

# t-test: 평균 수익률이 0보다 크냐?
t_stat, p_val = stats.ttest_1samp(rets, 0.0) if n > 1 else (0.0, 1.0)

# 누적 수익률
equity0 = df["equity"].iloc[0]
equityT = df["equity"].iloc[-1]
total_ret = (equityT - equity0) / equity0

# 최대 낙폭
equity_curve = df["equity"].values
peak = np.maximum.accumulate(equity_curve)
dd = (equity_curve - peak) / peak
max_dd = dd.min()

print("=== Live-style Portfolio Backtest (1D) ===")
print(f"N days           : {n}")
print(f"Total return     : {total_ret*100:.2f}%")
print(f"Mean daily ret   : {mean_ret*100:.4f}%")
print(f"Std daily ret    : {std_ret*100:.4f}%")
print(f"Sharpe (ann.)    : {sharpe:.3f}")
print(f"Max Drawdown     : {max_dd*100:.2f}%")
print("")
print("=== t-test: mean(return) > 0 ? ===")
print(f"t-stat           : {t_stat:.3f}")
print(f"p-value          : {p_val:.4f}")
print(f"Statistically significant at 5%? : {'YES' if p_val < 0.05 and mean_ret > 0 else 'NO'}")
```

## File: scripts/analyze_live_snapshots.py
```python
import pandas as pd
from pathlib import Path

def main():
    path = Path("reports/live_trader_snapshots.csv")
    if not path.exists():
        print("[ERROR] reports/live_trader_snapshots.csv 가 없습니다. live_trader_ml DRY 모드부터 충분히 돌려주세요.")
        return

    df = pd.read_csv(path)
    print("=== live_trader_snapshots head ===")
    print(df.head())

    print("\n=== 컬럼 목록 ===")
    print(df.columns.tolist())

    # 이후 여기서부터:
    # - 시간대별 equity 변화
    # - 코인별 포지션 비중
    # - conf/target_value_krw의 분포
    # 등을 분석하는 코드를 점점 추가해 나가면 된다.

if __name__ == "__main__":
    main()
```

## File: scripts/backfill_minute1_history.py
```python
import sys
from pathlib import Path
import time
import datetime as dt

import numpy as np
import pandas as pd
import pyupbit

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "ohlcv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKETS = ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]
MAX_DAYS = 365  # 최대 1년치 정도

def backfill_market(market: str):
    print(f"\n=== Backfilling {market} minute1 (older history) ===")
    prefix = market.replace("-", "_")
    out_path = OUT_DIR / f"{prefix}_minute1.parquet"

    if out_path.exists():
        df_existing = pd.read_parquet(out_path).sort_index()
        print(f"  existing rows: {len(df_existing)} "
              f"{df_existing.index[0]} -> {df_existing.index[-1]}")
        earliest = df_existing.index[0]
        latest = df_existing.index[-1]
    else:
        df_existing = None
        # 기존이 없으면 최신부터 과거로 내려감
        earliest = None
        latest = None
        print("  no existing file, will fetch from now backwards")

    all_dfs = []
    if df_existing is not None:
        all_dfs.append(df_existing)

    # 과거로 내려가기 위해 기준 시간을 "가장 오래된 시점"으로 잡고, 그 이전으로 to 파라미터를 계속 밀어감.
    to_ts = None
    if earliest is not None:
        # earliest 이전으로 내려가야 하니, earliest를 to 기준으로 사용
        to_ts = earliest.to_pydatetime()

    total_new = 0
    safety_loops = 0

    while True:
        safety_loops += 1
        if safety_loops > 2000:
            print("  safety break (too many loops)")
            break

        try:
            if to_ts is None:
                df = pyupbit.get_ohlcv(market, interval="minute1", count=200)
            else:
                df = pyupbit.get_ohlcv(market, interval="minute1", count=200, to=to_ts)
        except Exception as e:
            print(f"  error get_ohlcv: {e}")
            break

        if df is None or len(df) == 0:
            print("  no data returned from API")
            break

        df = df.sort_index()

        # 이미 있는 가장 오래된 시점 이전 데이터만 사용
        if earliest is not None:
            df = df[df.index < earliest]

        if len(df) == 0:
            print("  no older rows (reached earliest available from API or overlap only)")
            break

        all_dfs.append(df)
        total_new += len(df)
        oldest = df.index[0]
        newest = df.index[-1]
        print(f"  fetched {len(df)} rows: {oldest} -> {newest}")

        # 기간 계산 (새로운 earliest/lastest 반영)
        if earliest is not None:
            first = min(earliest, oldest)
        else:
            first = oldest
        if latest is not None:
            last = max(latest, newest)
        else:
            last = newest

        days_span = (last - first).total_seconds() / 86400.0
        print(f"  span so far: ~{days_span:.1f} days")

        if days_span >= MAX_DAYS:
            print(f"  reached MAX_DAYS ~{days_span:.1f}, stop.")
            break

        # 다음 루프에서 더 과거로 가기 위해 to_ts 갱신
        earliest = first
        to_ts = oldest.to_pydatetime()
        time.sleep(0.2)  # rate limit 보호[web:22][web:30]

    if not all_dfs:
        print("  nothing to save.")
        return

    df_all = pd.concat(all_dfs, axis=0)
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    df_all = df_all.sort_index()

    df_all.to_parquet(out_path)
    print(f"  saved {len(df_all)} rows: {df_all.index[0]} -> {df_all.index[-1]} to {out_path}")

def main():
    for m in MARKETS:
        backfill_market(m)

if __name__ == "__main__":
    main()
```

## File: scripts/backtest_live_portfolio_1d.py
```python
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.technical_indicators import TechnicalIndicators  # 1D 피처 스택[file:17]
from models.catboost_model import CatBoostPredictor            # CatBoost 시그널[file:17]
from scripts.live_trader_ml import (
    Opportunity,
    assign_target_values,
    RISKFRACTION,
    STOPLOSSPCT,
    REBALANCETHRESHOLD,
    MIN_TRADE_INTERVAL_SEC,  # 사용하진 않지만 가져만 옴
)
from scripts.orderutils import MINORDERKRW                    # 최소 주문금액 5,500원[file:17]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - LIVE_PORTFOLIO_BT - %(levelname)s - %(message)s",
)
logger = logging.getLogger("LIVE_PORTFOLIO_BT")


MARKETS = ["KRW-ETH"]
EQUITY_START = 1_000_000
COMMISSION_RATE = 0.0005  # 업비트 taker 수수료 0.05% 가정[file:17]
SLIPPAGE_RATE = 0.001     # 슬리피지 0.1% 가정


def market_to_parquet_prefix(market: str) -> str:
    # 예: KRW-ETH -> KRW_ETH_day.parquet[file:17]
    return market.replace("-", "_")


def load_daily_ohlcv(market: str) -> pd.DataFrame:
    prefix = market_to_parquet_prefix(market)
    path = ROOT / "data" / "ohlcv" / f"{prefix}_day.parquet"
    df = pd.read_parquet(path)
    df = df.sort_index()
    return df


def compute_conf_series(market: str) -> pd.Series:
    """
    simulate_ml_performance_1d.py와 동일한 방식으로
    일별 conf 시리즈를 계산.
    """
    logger.info("Loading daily OHLCV for %s", market)
    df = load_daily_ohlcv(market)

    ti = TechnicalIndicators(df)
    ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()
    closes = df.loc[df_feat.index, "close"].astype(float).values

    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feat_cols = cb.feature_names
    X = df_feat[feat_cols].values.astype(np.float32)

    probs = cb.model.predict_proba(X)
    if probs.shape[1] >= 2:
        conf_up = probs[:, -1]
    else:
        conf_up = probs[:, 0]

    times = df_feat.index
    s_conf = pd.Series(conf_up, index=times, name=f"conf_{market}")
    s_price = pd.Series(closes, index=times, name=f"close_{market}")
    return s_conf, s_price


def backtest_portfolio_1d():
    """
    3코인 포트폴리오를 live_trader_ml의 로직과 최대한 유사하게
    일별로 리밸런싱/손절하는 백테스트.[file:17]
    """
    logger.info("=== Starting 1D portfolio backtest (live-style) ===")
    conf_dict = {}
    price_dict = {}

    # 1) 각 코인별 conf 시리즈 및 가격 시리즈 계산
    for m in MARKETS:
        s_conf, s_price = compute_conf_series(m)
        conf_dict[m] = s_conf
        price_dict[m] = s_price

    # 2) 공통 날짜 인덱스 (교집합) 사용
    common_index = None
    for m in MARKETS:
        idx = conf_dict[m].index
        common_index = idx if common_index is None else common_index.intersection(idx)

    common_index = common_index.sort_values()
    logger.info("Common index length: %d days", len(common_index))

    # 3) 초기 상태
    equity = EQUITY_START
    cash = EQUITY_START
    positions = {m: 0.0 for m in MARKETS}
    avg_prices = {m: 0.0 for m in MARKETS}

    equity_curve = []
    daily_returns = []

    prev_equity = equity

    for dt in common_index:
        # 오늘 날짜의 가격/시그널
        prices_today = {m: float(price_dict[m].loc[dt]) for m in MARKETS}
        confs_today = {m: float(conf_dict[m].loc[dt]) for m in MARKETS}

        # 3-1) 손절 로직 적용 (평단 기준 STOPLOSSPCT)[file:17]
        for m in MARKETS:
            qty = positions[m]
            if qty <= 0:
                continue
            avg = avg_prices[m]
            price = prices_today[m]
            if avg <= 0:
                continue
            value = qty * price
            if value < MINORDERKRW:
                continue
            pnl_pct = (price - avg) / avg
            if pnl_pct <= STOPLOSSPCT:
                # 전량 청산
                sell_value = qty * price * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                cash += sell_value
                positions[m] = 0.0
                avg_prices[m] = 0.0
                logger.info(
                    "[STOPLOSS] %s qty=%.6f pnl_pct=%.4f equity_after=%.0f",
                    m, qty, pnl_pct, cash
                )

        # 3-2) 현재 포트폴리오 가치 계산
        pos_value = sum(positions[m] * prices_today[m] for m in MARKETS)
        equity = cash + pos_value

        # 3-3) 오늘 conf 기반 Opportunity 리스트 생성
        opps = []
        for m in MARKETS:
            c = confs_today[m]
            if not np.isfinite(c) or c <= 0.0:
                continue
            opps.append(Opportunity(market=m, conf=c))

        if not opps:
            # 기회가 없으면 포지션 유지
            equity_curve.append(equity)
            daily_returns.append((equity - prev_equity) / prev_equity)
            prev_equity = equity
            continue

        # 3-4) assign_target_values를 사용해 target_value_krw 계산[file:17]
        assign_target_values(opps, equity)

        # 3-5) 리밸런싱 (apply_rebalancing와 유사하지만 오프라인 버전)[file:17]
        opps_sorted = sorted(opps, key=lambda o: o.conf, reverse=True)

        for o in opps_sorted:
            mkt = o.market
            target = o.target_value_krw
            price = prices_today[mkt]
            cur_val = positions[mkt] * price
            diff = target - cur_val

            if target <= 0.0:
                continue

            min_gap = max(MINORDERKRW, target * REBALANCETHRESHOLD)

            # 매수
            if diff > 0:
                if diff < min_gap:
                    continue
                buy_amount = min(diff, cash - MINORDERKRW) if cash > MINORDERKRW else 0.0
                if buy_amount < MINORDERKRW:
                    continue
                # 수수료/슬리피지 고려
                gross = buy_amount
                cost = gross * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
                if cost > cash:
                    continue
                qty = gross / price
                # 평단 업데이트
                old_qty = positions[mkt]
                old_avg = avg_prices[mkt]
                new_qty = old_qty + qty
                new_avg = (
                    (old_qty * old_avg + qty * price) / new_qty
                    if new_qty > 0 else 0.0
                )
                positions[mkt] = new_qty
                avg_prices[mkt] = new_avg
                cash -= cost
                logger.info(
                    "[REBAL BUY] %s buy_krw=%.0f qty=%.6f new_qty=%.6f cash=%.0f",
                    mkt, gross, qty, new_qty, cash
                )

            # 매도
            elif diff < 0:
                if abs(diff) < min_gap:
                    continue
                # 목표보다 초과 보유한 금액만큼 매도
                reduce_val = min(abs(diff), positions[mkt] * price)
                if reduce_val < MINORDERKRW:
                    continue
                sell_qty = reduce_val / price
                if sell_qty <= 0.0:
                    continue
                sell_qty = min(sell_qty, positions[mkt])
                sell_value = sell_qty * price * (1 - COMMISSION_RATE - SLIPPAGE_RATE)
                positions[mkt] -= sell_qty
                if positions[mkt] <= 0:
                    positions[mkt] = 0.0
                    avg_prices[mkt] = 0.0
                cash += sell_value
                logger.info(
                    "[REBAL SELL] %s sell_krw=%.0f qty=%.6f remain_qty=%.6f cash=%.0f",
                    mkt, sell_value, sell_qty, positions[mkt], cash
                )

        # 3-6) 오늘 종료 후 equity 기록
        pos_value = sum(positions[m] * prices_today[m] for m in MARKETS)
        equity = cash + pos_value
        equity_curve.append(equity)
        daily_returns.append((equity - prev_equity) / prev_equity)
        prev_equity = equity

    equity_curve = np.array(equity_curve)
    daily_returns = np.array(daily_returns)

    # 성능 지표 계산
    total_return = (equity_curve[-1] - EQUITY_START) / EQUITY_START
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
    max_dd = ((equity_curve / np.maximum.accumulate(equity_curve)) - 1.0).min()

    logger.info(
        "=== Backtest Done | Final Equity=%.0f | Total Return=%.2f%% | Sharpe=%.3f | MaxDD=%.2f%% ===",
        equity_curve[-1],
        total_return * 100.0,
        sharpe,
        max_dd * 100.0,
    )

    # 결과 CSV 저장
    out_path = ROOT / "reports" / "backtest_live_portfolio_1d_equity.csv"
    df_eq = pd.DataFrame(
        {
            "date": common_index,
            "equity": equity_curve,
            "daily_return": daily_returns,
        }
    )
    df_eq.to_csv(out_path, index=False)
    logger.info("Saved equity curve to %s", out_path)


if __name__ == "__main__":
    backtest_portfolio_1d()
```

## File: scripts/backtest_scalping_1m_ethusdt.py
```python
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "binance_ohlcv" / "ETHUSDT_1m.parquet"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

INITIAL_EQUITY = 10_000.0
FEE_RATE = 0.0005      # 0.05% 수수료 가정
SLIPPAGE = 0.0003      # 0.03% 슬리피지 가정
RISK_PER_TRADE = 0.02  # 계좌의 2%만 베팅

# 전략 파라미터
OPEN_RANGE_MINUTES = 30       # 매일 첫 30분을 오픈 레인지로 사용
TP_PCT = 0.006                # +0.6% 이익 실현
SL_PCT = -0.003               # -0.3% 손절
MAX_HOLD_MINUTES = 240        # 최대 보유 시간 4시간

def load_data():
    df = pd.read_parquet(DATA_PATH).sort_index()
    df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)].copy()
    return df

def make_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    오픈 레인지(첫 30분 high/low) 상단 돌파 시 하루에 최대 1회 롱 진입.
    청산은 backtest()에서 SL/TP/최대 보유시간/일 종료로 처리.
    """
    df = df.copy()
    df["date"] = df.index.date
    df["entry_long"] = False

    for d, sub in df.groupby("date"):
        if len(sub) < OPEN_RANGE_MINUTES + 5:
            continue
        first_ts = sub.index[0]
        or_end = first_ts + pd.Timedelta(minutes=OPEN_RANGE_MINUTES)

        in_range = sub.index < or_end
        if in_range.sum() == 0:
            continue

        range_high = sub.loc[in_range, "high"].max()

        after = sub.loc[~in_range]
        if after.empty:
            continue

        entered = False
        for ts, row in after.iterrows():
            price = row["close"]
            if (not entered) and price > range_high:
                df.at[ts, "entry_long"] = True
                entered = True
                break   # 하루에 한 번만 진입

    df = df.drop(columns=["date"])
    return df

def backtest(df: pd.DataFrame):
    cash = INITIAL_EQUITY
    position = 0.0
    equity_curve = []
    trade_pnls = []

    in_trade = False
    entry_price = 0.0
    entry_time = None

    for ts, row in df.iterrows():
        price = float(row["close"])
        cur_day = ts.date()

        # 진입 조건
        if (not in_trade) and row.get("entry_long", False):
            gross_to_invest = cash * RISK_PER_TRADE
            if gross_to_invest > 0:
                effective_price = price * (1 + SLIPPAGE + FEE_RATE)
                qty = gross_to_invest / effective_price
                if qty > 0:
                    cash -= qty * effective_price
                    position = qty
                    in_trade = True
                    entry_price = price
                    entry_time = ts

        # 청산 조건
        if in_trade:
            hold_minutes = (ts - entry_time).total_seconds() / 60.0
            ret = (price - entry_price) / entry_price

            exit_reason = None
            if ret >= TP_PCT:
                exit_reason = "TP"
            elif ret <= SL_PCT:
                exit_reason = "SL"
            elif hold_minutes >= MAX_HOLD_MINUTES:
                exit_reason = "TIME"
            elif ts.date() != entry_time.date():
                exit_reason = "EOD"

            if exit_reason is not None:
                effective_price = price * (1 - SLIPPAGE - FEE_RATE)
                proceeds = position * effective_price
                cash += proceeds
                trade_pnls.append(ret)
                position = 0.0
                in_trade = False
                entry_price = 0.0
                entry_time = None

        equity = cash + position * price
        equity_curve.append((ts, equity))

    eq = pd.Series(
        [e for (_, e) in equity_curve],
        index=[t for (t, _) in equity_curve],
        name="equity",
    )
    return eq, trade_pnls

def analyze(eq: pd.Series, trade_pnls):
    equity = eq.values
    dates = eq.index

    total_return = (equity[-1] / equity[0]) - 1.0

    rets_1m = np.diff(equity) / equity[:-1]
    df = pd.DataFrame({"equity": equity}, index=dates)
    df["ret_1m"] = np.insert(rets_1m, 0, 0.0)
    daily = df["ret_1m"].resample("1D").sum()
    daily = daily[daily != 0.0]

    mean_daily = daily.mean()
    std_daily = daily.std(ddof=1) if len(daily) > 1 else 0.0
    sharpe = mean_daily / (std_daily + 1e-10) * np.sqrt(252)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    n_trades = len(trade_pnls)

    print("=== ETHUSDT 1m Open-Range Breakout Long-only Backtest ===")
    print(f"Period          : {dates[0]} -> {dates[-1]}")
    print(f"Initial Equity  : {equity[0]:.2f} USDT")
    print(f"Final Equity    : {equity[-1]:.2f} USDT")
    print(f"Total Return    : {total_return*100:.2f}%")
    print(f"Mean daily ret  : {mean_daily*100:.4f}%")
    print(f"Sharpe (daily)  : {sharpe:.3f}")
    print(f"Max Drawdown    : {max_dd*100:.2f}%")
    print(f"Trades          : {n_trades}")

    out_path = REPORTS_DIR / "binance_ETHUSDT_1m_openrange_equity.csv"
    eq.to_csv(out_path, header=True)
    print(f"\nEquity curve saved to {out_path}")

def main():
    df = load_data()
    df = make_signals(df)
    eq, trade_pnls = backtest(df)
    analyze(eq, trade_pnls)

if __name__ == "__main__":
    main()
```

## File: scripts/build_today_signals.py
```python
import os
from pathlib import Path

import pandas as pd

# 전략 A에서 사용하는 코인 목록 (필요하면 여기서 추가/수정)
A_MARKETS = ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]

def load_last_conf_for_market(market: str) -> float | None:
    """
    signals_1d_trades_KRW_ETH.csv 같은 파일에서
    가장 최근 트레이드의 entry_conf를 읽어온다.
    파일 패턴: signals_1d_trades_KRW_ETH.csv
    """
    fname = f"signals_1d_trades_{market.replace('-', '_')}.csv"
    path = Path(fname)
    if not path.exists():
        print(f"[INFO] file not found for {market}: {fname}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[ERROR] reading {fname} failed:", e)
        return None
    if "entry_conf" not in df.columns:
        print(f"[ERROR] {fname} has no 'entry_conf' column")
        return None
    if "entry_time" in df.columns:
        # 가장 최근 entry_time 기준으로 정렬
        try:
            df = df.sort_values("entry_time")
        except Exception:
            pass
    last = df.iloc[-1]
    conf = float(last["entry_conf"])
    # conf가 0~1 범위 밖이면 클리핑
    if conf <= 0.0:
        return None
    return conf

def main() -> None:
    rows: list[dict] = []
    for mkt in A_MARKETS:
        conf = load_last_conf_for_market(mkt)
        if conf is None:
            continue
        rows.append({"market": mkt, "conf": conf})

    if not rows:
        print("[INFO] no valid signals found, today_signals.csv will not be written")
        return

    os.makedirs("signals", exist_ok=True)
    out_path = Path("signals/today_signals.csv")
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)
    print(f"[INFO] wrote {out_path} with {len(rows)} rows")

if __name__ == "__main__":
    main()
```

## File: scripts/compare_strategy_vs_hodl.py
```python
"""
ETH / DOGE / AVAX
- 전략 A (signals_1d_trades_*.csv) vs 동일 기간 HODL 수익률 비교
- 매매 빈도(일 평균 트레이드 수)까지 같이 출력
"""

import pandas as pd
import numpy as np
import pyupbit
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "KRW-ETH": ROOT / "signals_1d_trades_KRW_ETH.csv",
    "KRW-DOGE": ROOT / "signals_1d_trades_KRW_DOGE.csv",
    "KRW-AVAX": ROOT / "signals_1d_trades_KRW_AVAX.csv",
}

def load_hodl_return(market: str, start_ts, end_ts):
    df = pyupbit.get_ohlcv(market, interval="day", count=400)
    if df is None or len(df) == 0:
        return None, None, None

    df = df.sort_index()

    # start_ts보다 크거나 같은 첫 날, end_ts보다 작거나 같은 마지막 날
    start_idx = df.index.searchsorted(start_ts, side="left")
    end_idx = df.index.searchsorted(end_ts, side="right") - 1
    start_idx = max(0, min(start_idx, len(df)-1))
    end_idx = max(0, min(end_idx, len(df)-1))

    p_start = float(df["close"].iloc[start_idx])
    p_end = float(df["close"].iloc[end_idx])
    hodl_ret = p_end / p_start - 1.0
    return p_start, p_end, hodl_ret

def main():
    rows = []

    for mkt, path in FILES.items():
        df = pd.read_csv(path)
        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        df = df.sort_values("entry_time")

        n_trades = len(df)
        start = df["entry_time"].min()
        end = df["exit_time"].max()
        days = (end - start).days or 1

        # 전략 성과
        ret = df["ret"].values
        total_pnl = float(np.prod(1.0 + ret) - 1.0)
        mean_ret = float(ret.mean())
        winrate = float((ret > 0).mean())
        trades_per_day = n_trades / days

        # 동일 기간 HODL 성과
        _, _, hodl_ret = load_hodl_return(mkt, start, end)

        rows.append(
            {
                "market": mkt,
                "trades": n_trades,
                "period_days": days,
                "trades_per_day": trades_per_day,
                "strategy_total": total_pnl,
                "strategy_mean_ret": mean_ret,
                "strategy_winrate": winrate,
                "hodl_ret": hodl_ret,
                "edge_vs_hodl": None if hodl_ret is None else total_pnl - hodl_ret,
            }
        )

    result = pd.DataFrame(rows)
    print("=== Strategy A vs HODL (ETH / DOGE / AVAX) ===")
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
```

## File: scripts/compute_excess_exposure.py
```python
import pandas as pd
from pathlib import Path

RISK_FRACTION = 0.7

def main():
    path = Path("reports/live_trader_snapshots.csv")
    if not path.exists():
        print("[ERROR] snapshots CSV 없음")
        return

    df = pd.read_csv(path)
    last = df.iloc[-1]  # 가장 최근 스냅샷

    equity = float(last["equity_krw"])
    pos_val = float(last["KRW-ETH_val"] + last["KRW-DOGE_val"] + last["KRW-AVAX_val"])
    target_total = equity * RISK_FRACTION
    excess = pos_val - target_total

    print("=== 최신 스냅샷 기준 목표 대비 과매수 계산 ===")
    print(f"시각        : {last['ts']}")
    print(f"equity_krw  : {equity:,.2f}")
    print(f"실제 포지션: {pos_val:,.2f} KRW")
    print(f"목표 포지션: {target_total:,.2f} KRW (RISK_FRACTION={RISK_FRACTION})")
    print(f"초과 노출   : {excess:,.2f} KRW")

    if excess <= 0:
        print("\n[INFO] 이미 목표 이하로 노출이 줄어든 상태입니다.")
        return

    # conf 비율로 초과분을 나눠서, 각 코인별로 얼만큼 줄이면 되는지 제안
    conf_eth = float(last["KRW_ETH_conf"])
    conf_doge = float(last["KRW_DOGE_conf"])
    conf_avax = float(last["KRW_AVAX_conf"])
    conf_sum = conf_eth + conf_doge + conf_avax

    w_eth = conf_eth / conf_sum
    w_doge = conf_doge / conf_sum
    w_avax = conf_avax / conf_sum

    print("\n=== 초과분을 conf 비율로 나눈 \"줄여야 할 평가금\" 제안 ===")
    print(f"KRW-ETH 줄이기: {excess * w_eth:,.2f} KRW")
    print(f"KRW-DOGE 줄이기: {excess * w_doge:,.2f} KRW")
    print(f"KRW-AVAX 줄이기: {excess * w_avax:,.2f} KRW")

if __name__ == "__main__":
    main()
```

## File: scripts/daily_intraday_report.py
```python
import pandas as pd
from pathlib import Path

RISK_FRACTION = 0.7

def main():
    path = Path("reports/live_trader_snapshots.csv")
    if not path.exists():
        print("[ERROR] reports/live_trader_snapshots.csv 가 없습니다. DRY_RUN으로 live_trader_ml을 먼저 충분히 돌려주세요.")
        return

    df = pd.read_csv(path)
    print("=== snapshots shape ===", df.shape)
    print("=== time range ===")
    print(df["ts"].min(), " ~ ", df["ts"].max())

    # 기본 파생 컬럼
    df["pos_val"] = df["KRW-ETH_val"] + df["KRW-DOGE_val"] + df["KRW-AVAX_val"]
    df["pos_frac"] = df["pos_val"] / df["equity_krw"].replace(0, pd.NA)

    # 시간대 컬럼
    df["hour"] = df["ts"].str.slice(11, 13).astype(int)

    # 1) 전체 구간 포지션 비중 요약
    print("\n=== 전체 포지션 비중 통계 (ETH+DOGE+AVAX / equity) ===")
    print(df["pos_frac"].describe())

    # 2) 코인별 target - current gap
    for mkt in ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]:
        val_col = f"{mkt}_val"
        tgt_col = f"{mkt.replace('-', '_')}_target"
        diff_col = f"{mkt}_gap"
        df[diff_col] = df[tgt_col] - df[val_col]
        print(f"\n=== {mkt} target - current (gap) 통계 ===")
        print(df[diff_col].describe())

    # 3) 시간대별 평균 노출/갭
    grp = df.groupby("hour").agg({
        "pos_frac": "mean",
        "KRW-ETH_gap": "mean",
        "KRW-DOGE_gap": "mean",
        "KRW-AVAX_gap": "mean",
    }).reset_index()

    print("\n=== 시간대별 평균 pos_frac / gap ===")
    print(grp)

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "intraday_hourly_stats.csv"
    grp.to_csv(out_path, index=False)
    print(f"\n[INFO] wrote {out_path}")

if __name__ == "__main__":
    main()
```

## File: scripts/download_binance_1m_history.py
```python
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

from binance.client import Client
from binance.enums import HistoricalKlinesType

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "binance_ohlcv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def month_range(start_year: int):
    """start_year 1월부터 현재까지 (year, month) 순회."""
    now = datetime.utcnow()
    y, m = start_year, 1
    while True:
        if y > now.year or (y == now.year and m > now.month):
            break
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1

def month_start_end(year: int, month: int):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start, end

def download_1m_klines_by_month(symbol: str, start_year: int, out_path: Path):
    api_key = ""
    api_secret = ""
    client = Client(api_key, api_secret, tld="com")

    all_dfs = []

    for (y, m) in month_range(start_year):
        start_dt, end_dt = month_start_end(y, m)
        start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

        print(f"=== {symbol} 1m {start_str} -> {end_str} ===")
        klines = client.get_historical_klines(
            symbol,
            Client.KLINE_INTERVAL_1MINUTE,
            start_str,
            end_str,
            klines_type=HistoricalKlinesType.SPOT,
        )
        print(f"  fetched rows: {len(klines)}")

        if not klines:
            continue

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
        df = pd.DataFrame(klines, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        for c in ["open", "high", "low", "close", "volume",
                  "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
            df[c] = df[c].astype(float)
        df["number_of_trades"] = df["number_of_trades"].astype(int)
        df = df.set_index("open_time").sort_index()
        all_dfs.append(df)

    if not all_dfs:
        print("no data fetched at all")
        return

    df_all = pd.concat(all_dfs, axis=0)
    df_all = df_all[~df_all.index.duplicated(keep="last")]
    df_all = df_all.sort_index()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_path)
    print(f"saved to {out_path} | {df_all.index[0]} -> {df_all.index[-1]} (rows={len(df_all)})")

def main():
    if len(sys.argv) != 4:
        print("Usage: python scripts/download_binance_1m_history.py SYMBOL START_YEAR OUT_PATH")
        print("Example: python scripts/download_binance_1m_history.py ETHUSDT 2018 data/binance_ohlcv/ETHUSDT_1m.parquet")
        sys.exit(1)

    symbol = sys.argv[1]
    start_year = int(sys.argv[2])
    out_rel = sys.argv[3]

    out_path = ROOT / out_rel
    download_1m_klines_by_month(symbol, start_year, out_path)

if __name__ == "__main__":
    main()
```

## File: scripts/download_multitf_history.py
```python
#!/usr/bin/env python
import time
from datetime import datetime
from pathlib import Path

import pyupbit
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data" / "ohlcv"
OUTDIR.mkdir(parents=True, exist_ok=True)

MARKETS = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE", "KRW-ADA", "KRW-AVAX"]
INTERVALS = ["minute1", "minute15", "minute60", "day"]  # 코어 TF
COUNT = 200  # pyupbit 최대 chunk

def backfill_market_interval(market: str, interval: str, max_loops: int = 2000):
    """
    Upbit get_ohlcv 를 to 파라미터로 뒤로 밀어가며 최대한 깊게 수집.[web:21][web:22]
    """
    print(f"[MULTITF] Backfill {market} {interval}")
    all_chunks = []
    to = None

    for i in range(max_loops):
        try:
            df = pyupbit.get_ohlcv(market, interval=interval, count=COUNT, to=to)
        except Exception as e:
            print(f"[WARN] get_ohlcv failed {market} {interval} loop={i}: {e}")
            break

        if df is None or len(df) == 0:
            print(f"[INFO] No more data for {market} {interval} at loop {i}")
            break

        df = df.sort_index()
        all_chunks.append(df)

        oldest_ts = df.index[0]
        print(f"[LOOP {i}] {market} {interval} got {len(df)} rows from {df.index[0]} to {df.index[-1]}")

        # 다음 루프에서 더 과거를 보기 위해 oldest_ts 이전으로 to 이동
        # Upbit는 to 를 'YYYY-MM-DD HH:MM:SS' 문자열로 받는다.[web:21]
        to = datetime.strftime(oldest_ts, "%Y-%m-%d %H:%M:%S")

        # rate limit 보호
        time.sleep(0.2)

        # 마지막 청크가 200 미만이면 더 이상 갈 수 없다고 보고 종료
        if len(df) < COUNT:
            print(f"[INFO] Last chunk < COUNT ({len(df)} < {COUNT}), stopping {market} {interval}")
            break

    if not all_chunks:
        print(f"[WARN] No data collected for {market} {interval}")
        return

    full = pd.concat(all_chunks).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    out = OUTDIR / f"{market.replace('-', '')}_{interval}.parquet"
    full.to_parquet(out)
    print(f"[DONE] {market} {interval}: saved {len(full)} rows to {out}")

def main():
    for m in MARKETS:
        for interval in INTERVALS:
            backfill_market_interval(m, interval)
    print("[MULTITF] All done.")

if __name__ == "__main__":
    main()
```

## File: scripts/generate_signals_1d.py
```python
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyupbit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor


MARKET = "KRW-BTC"
DAYS = 365
CONF_THRESHOLD = 0.70

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - SIGNALS_1D - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SIGNALS_1D")


def main() -> None:
    logger.info("Fetching %d days of daily candles for %s", DAYS, MARKET)
    df = pyupbit.get_ohlcv(MARKET, interval="day", count=DAYS + 50)
    if df is None or len(df) < 50:
        raise RuntimeError(f"Not enough daily candles for {MARKET}: got {0 if df is None else len(df)}")
    df = df.sort_index()

    ti = TechnicalIndicators(df)
    ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    closes = df.loc[df_feat.index, "close"].astype(float).values
    times = df_feat.index

    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feat_cols = cb.feature_names
    X = df_feat[feat_cols].values.astype(np.float32)
    probs = cb.model.predict_proba(X)
    if probs.shape[1] >= 2:
        conf_up = probs[:, -1]
    else:
        conf_up = probs[:, 0]

    out = pd.DataFrame(
        {
            "time": times,
            "close": closes,
            "conf_catboost": conf_up,
        }
    )
    out["long_signal"] = out["conf_catboost"] >= CONF_THRESHOLD

    out_path = ROOT / "signals_1d.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(out), out_path)


if __name__ == "__main__":
    main()
```

## File: scripts/gridsearch_1d_signals.py
```python
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyupbit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor


MARKETS = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-SOL",
    "KRW-DOGE",
    "KRW-ADA",
    "KRW-AVAX",
]

DAYS = 365

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]
HOLD_DAYS = [1, 2, 3]
STOP_LOSSES = [0.01, 0.02, 0.03, 0.04]
TAKE_PROFITS = [0.02, 0.04, 0.06, 0.08]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - GRID_1D - %(levelname)s - %(message)s",
)
logger = logging.getLogger("GRID_1D")


def load_daily_features(market: str):
    logger.info("Fetching %d days of daily candles for %s", DAYS, market)
    df = pyupbit.get_ohlcv(market, interval="day", count=DAYS + 50)
    if df is None or len(df) < 100:
        raise RuntimeError(f"Not enough daily candles for {market}: got {0 if df is None else len(df)}")
    df = df.sort_index()

    ti = TechnicalIndicators(df)
    ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    closes = df.loc[df_feat.index, "close"].astype(float).values
    times = df_feat.index

    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feat_cols = cb.feature_names
    X = df_feat[feat_cols].values.astype(np.float32)
    probs = cb.model.predict_proba(X)
    if probs.shape[1] >= 2:
        conf_up = probs[:, -1]
    else:
        conf_up = probs[:, 0]

    return times, closes, conf_up


def backtest_grid(market: str, times, closes, conf_up):
    rows = []
    n = len(times)

    for thr in THRESHOLDS:
        for hold in HOLD_DAYS:
            for sl in STOP_LOSSES:
                for tp in TAKE_PROFITS:
                    trades = 0
                    wins = 0
                    rets = []

                    i = 0
                    while i < n - 1:
                        if conf_up[i] >= thr:
                            entry_price = closes[i]
                            exit_ret = None

                            max_j = min(n - 1, i + hold)
                            j = i + 1
                            while j <= max_j:
                                r = closes[j] / entry_price - 1.0
                                if r <= -sl or r >= tp or j == max_j:
                                    exit_ret = r
                                    break
                                j += 1

                            if exit_ret is not None:
                                trades += 1
                                if exit_ret > 0:
                                    wins += 1
                                rets.append(exit_ret)
                                i = j + 1
                                continue

                        i += 1

                    if trades == 0:
                        winrate = 0.0
                        avg_ret = 0.0
                        total_pnl = 0.0
                    else:
                        winrate = wins / trades
                        avg_ret = float(np.mean(rets))
                        total_pnl = float(np.prod([1.0 + r for r in rets]) - 1.0)

                    rows.append(
                        {
                            "market": market,
                            "conf_threshold": thr,
                            "hold_days": hold,
                            "stop_loss": sl,
                            "take_profit": tp,
                            "trades": trades,
                            "winrate": winrate,
                            "avg_ret": avg_ret,
                            "total_pnl": total_pnl,
                        }
                    )

                    logger.info(
                        "market=%s thr=%.2f hold=%d sl=%.3f tp=%.3f -> trades=%d winrate=%.3f total_pnl=%.3f",
                        market,
                        thr,
                        hold,
                        sl,
                        tp,
                        trades,
                        winrate,
                        total_pnl,
                    )

    return rows


def main() -> None:
    all_rows = []
    for m in MARKETS:
        try:
            times, closes, conf_up = load_daily_features(m)
        except Exception as e:
            logger.error("Skipping %s due to error: %s", m, e)
            continue
        rows = backtest_grid(m, times, closes, conf_up)
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No results computed.")
        return

    out = pd.DataFrame(all_rows)
    out = out.sort_values(["market", "total_pnl"], ascending=[True, False])

    out_path = ROOT / "grid_1d_results_multi.csv"
    out.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(out), out_path)

    best = (
        out.groupby("market")
        .head(5)
        .reset_index(drop=True)
    )
    print()
    print("===== Top 5 configs per market (sorted by total_pnl) =====")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
```

## File: scripts/gridsearch_scalping_1m_ethusdt.py
```python
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "binance_ohlcv" / "ETHUSDT_1m.parquet"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"
INITIAL_EQUITY = 10_000.0

# 그리드 범위
FAST_EMAS = [10, 20, 30]
SLOW_EMAS = [40, 60, 90]
RISK_PERS = [0.01, 0.02, 0.05]  # 계좌당 1%, 2%, 5%

def load_data():
    df = pd.read_parquet(DATA_PATH).sort_index()
    df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)].copy()
    return df

def add_indicators(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    close = df["close"]
    df = df.copy()
    df["ema_fast"] = close.ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = close.ewm(span=slow, adjust=False).mean()
    df["signal_raw"] = np.where(df["ema_fast"] > df["ema_slow"], 1, 0)
    df["signal_shift"] = df["signal_raw"].shift(1).fillna(0)
    df["entry_long"] = (df["signal_shift"] == 0) & (df["signal_raw"] == 1)
    df["exit_long"]  = (df["signal_shift"] == 1) & (df["signal_raw"] == 0)
    return df

def backtest(df: pd.DataFrame, risk_per_trade: float):
    cash = INITIAL_EQUITY
    position = 0.0
    equity_curve = []
    trade_pnls = []
    in_trade = False
    entry_price = 0.0

    for ts, row in df.iterrows():
        price = float(row["close"])

        if row["entry_long"] and position == 0.0:
            gross_to_invest = cash * risk_per_trade
            if gross_to_invest > 0:
                qty = gross_to_invest / price
                cash -= qty * price
                position += qty
                in_trade = True
                entry_price = price

        elif row["exit_long"] and position > 0.0:
            proceeds = position * price
            cash += proceeds
            if in_trade and entry_price > 0:
                pnl_pct = (price - entry_price) / entry_price
                trade_pnls.append(pnl_pct)
            position = 0.0
            in_trade = False
            entry_price = 0.0

        equity = cash + position * price
        equity_curve.append((ts, equity))

    eq = pd.Series([e for (_, e) in equity_curve],
                   index=[t for (t, _) in equity_curve],
                   name="equity")
    return eq, trade_pnls

def analyze(eq: pd.Series, trade_pnls):
    equity = eq.values
    dates = eq.index
    total_return = (equity[-1] / equity[0]) - 1.0

    rets_1m = np.diff(equity) / equity[:-1]
    df = pd.DataFrame({"equity": equity}, index=dates)
    df["ret_1m"] = np.insert(rets_1m, 0, 0.0)
    daily = df["ret_1m"].resample("1D").sum()
    daily = daily[daily != 0.0]

    mean_daily = daily.mean()
    std_daily = daily.std(ddof=1) if len(daily) > 1 else 0.0
    sharpe = mean_daily / (std_daily + 1e-10) * np.sqrt(252)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    n_trades = len(trade_pnls)
    if n_trades > 0:
        wins = [p for p in trade_pnls if p > 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean(wins) if wins else 0.0
        losses = [p for p in trade_pnls if p <= 0]
        avg_loss = np.mean(losses) if losses else 0.0
    else:
        win_rate = avg_win = avg_loss = 0.0

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }

def main():
    base_df = load_data()
    results = []

    for fast, slow, risk in product(FAST_EMAS, SLOW_EMAS, RISK_PERS):
        if fast >= slow:
            continue  # fast는 항상 slow보다 짧게
        df = add_indicators(base_df, fast, slow)
        eq, trade_pnls = backtest(df, risk)
        metrics = analyze(eq, trade_pnls)
        row = {
            "fast_ema": fast,
            "slow_ema": slow,
            "risk_per_trade": risk,
        }
        row.update(metrics)
        print(f"[GRID] fast={fast} slow={slow} risk={risk:.2f} "
              f"ret={metrics['total_return']*100:.2f}% sharpe={metrics['sharpe']:.3f} "
              f"dd={metrics['max_dd']*100:.1f}% trades={metrics['n_trades']}")
        results.append(row)

    df_res = pd.DataFrame(results)
    out_path = REPORTS_DIR / "grid_1m_ETHUSDT_EMA20-60.csv"
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved grid results to {out_path}")

if __name__ == "__main__":
    main()
```

## File: scripts/label_regimes_from_ohlcv.py
```python
"""
Label market regimes (CRASH / BULL / RANGE) from daily OHLCV.

- Input: data/ohlcv/{MARKET}_day.parquet  (from download_multi_tf_history.py)
- Output: data/regimes/{MARKET}_day_regimes.parquet

Regime definition (30일 롤링 윈도우 기준):
- CRASH: close / 30일 rolling max - 1 <= -0.25   (최근 고점 대비 -25% 이상 하락)
- BULL:  close / 30일 rolling min - 1 >=  0.25   (최근 저점 대비 +25% 이상 상승)
- RANGE: 나머지
"""

from pathlib import Path
from typing import List

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "data" / "ohlcv"
OUT_DIR = PROJECT_ROOT / "data" / "regimes"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKETS: List[str] = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-SOL",
    "KRW-DOGE",
    "KRW-ADA",
    "KRW-AVAX",
]

ROLL_WINDOW = 30
CRASH_DD_THRESH = -0.25   # -25%
BULL_UP_THRESH = 0.25     # +25%


def label_regimes_for_market(market: str):
    in_path = IN_DIR / f"{market.replace('-', '_')}_day.parquet"
    if not in_path.exists():
        print(f"[WARN] daily file not found for {market}: {in_path}")
        return

    df = pd.read_parquet(in_path)
    if df.empty:
        print(f"[WARN] empty daily data for {market}")
        return

    df = df.sort_index()

    close = df["close"]

    rolling_max = close.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW // 2).max()
    rolling_min = close.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW // 2).min()

    drawdown_from_high = close / rolling_max - 1.0
    upmove_from_low = close / rolling_min - 1.0

    regime = np.where(
        drawdown_from_high <= CRASH_DD_THRESH,
        "CRASH",
        np.where(
            upmove_from_low >= BULL_UP_THRESH,
            "BULL",
            "RANGE",
        ),
    )

    out = df.copy()
    out["drawdown_from_high"] = drawdown_from_high
    out["upmove_from_low"] = upmove_from_low
    out["regime"] = regime

    out_path = OUT_DIR / f"{market.replace('-', '_')}_day_regimes.parquet"
    out.to_parquet(out_path)
    print(f"[OK] {market}: labeled regimes -> {out_path}")


def main():
    for m in MARKETS:
        label_regimes_for_market(m)

    print("\n[label_regimes_from_ohlcv] DONE.")


if __name__ == "__main__":
    main()
```

## File: scripts/live_trader_ml.py
```python
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import pyupbit
import pandas as pd

from scripts.orderutils import MINORDERKRW  # 최소 주문금액 (예: 5500.0 KRW)

# ===== 설정값 =====
SIGNALS_CSV = "signals/today_signals.csv"  # ML이 뽑은 오늘자 시그널 파일 경로 (market, conf 컬럼 필수)
LOOP_SECONDS = 10                          # 루프 주기
DRYRUN = True                             # True면 실제 주문 안 나가고 로그만 출력
RISKFRACTION = 0.3                         # 전체 계좌의 몇 %를 포지션에 쓸지
STOPLOSSPCT = -0.05                        # -5% 손절
REBALANCETHRESHOLD = 0.10                  # 목표 비중 대비 10% 이상 차이날 때만 리밸런싱
MIN_TRADE_INTERVAL_SEC = 300               # 같은 코인에 대해 최소 300초(5분) 간격으로만 트레이드

# 코인별 마지막 매매 시간 저장용 (프로세스 살아있는 동안만 유지)
last_trade_ts: Dict[str, float] = {}

# 업비트 KRW 마켓에서 실제 존재하는 티커 목록 (한 번만 조회)
try:
    VALID_KRW_TICKERS = set(pyupbit.get_tickers(fiat="KRW"))  # 예: {"KRW-BTC", "KRW-ETH", ...}
except Exception:
    VALID_KRW_TICKERS = set()


@dataclass
class Opportunity:
    market: str
    conf: float
    target_value_krw: float = 0.0  # 나중에 계산해서 채움


def load_client() -> pyupbit.Upbit:
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY not set in environment")
    return pyupbit.Upbit(access, secret)


def get_balances_raw(upbit: pyupbit.Upbit) -> List[dict]:
    """업비트 원본 잔고 리스트 (balance, avg_buy_price 포함)."""
    return upbit.get_balances()


def parse_balances(raw: List[dict]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    raw 잔고 리스트를
    - balances: {ticker: qty}
    - avg_prices: {ticker: avg_buy_price}
    로 변환 (ticker 예: 'KRW-ETH').
    """
    balances: Dict[str, float] = {}
    avg_prices: Dict[str, float] = {}
    for b in raw:
        cur = b.get("currency")
        bal = float(b.get("balance", "0"))
        avg = float(b.get("avg_buy_price", "0"))
        if cur == "KRW":
            balances["KRW"] = bal
        else:
            ticker = f"KRW-{cur}"
            balances[ticker] = bal
            avg_prices[ticker] = avg
    return balances, avg_prices


def get_prices(markets: List[str]) -> Dict[str, float]:
    """현재가 조회 (유효한 KRW 마켓만 조회해서 Code not found 방지)."""
    if not markets:
        return {}
    valid = [m for m in markets if m in VALID_KRW_TICKERS]
    if not valid:
        return {}
    try:
        tickers = pyupbit.get_current_price(valid)
    except Exception as e:
        print("[ERROR] get_current_price failed:", e)
        return {}
    if isinstance(tickers, dict):
        return {k: float(v) for k, v in tickers.items() if v is not None}
    return {}


def estimate_total_equity(balances: Dict[str, float],
                          prices: Dict[str, float]) -> float:
    """KRW + 코인 평가금액 합산."""
    equity = balances.get("KRW", 0.0)
    for ticker, qty in balances.items():
        if ticker == "KRW":
            continue
        price = prices.get(ticker)
        if not price:
            continue
        equity += qty * price
    return equity


def load_signals_from_csv(path: str) -> List[Opportunity]:
    """
    ML이 만든 오늘자 시그널 CSV에서 기회 리스트를 읽어온다.
    기대 컬럼:
    - market: 'KRW-ETH' 같은 티커
    - conf  : 0~1 사이 확신도
    """
    if not os.path.exists(path):
        print(f"[INFO] signals file not found: {path} (no trades this loop)")
        return []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("[ERROR] failed to read signals csv:", e)
        return []

    required_cols = {"market", "conf"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] signals csv missing columns: {required_cols - set(df.columns)}")
        return []

    opps: List[Opportunity] = []
    for _, row in df.iterrows():
        market = str(row["market"]).strip()
        try:
            conf = float(row["conf"])
        except Exception:
            continue
        if not market.startswith("KRW-"):
            continue
        if market not in VALID_KRW_TICKERS:
            continue
        if conf <= 0.0:
            continue
        opps.append(Opportunity(market=market, conf=conf))
    return opps


def assign_target_values(opps: List[Opportunity],
                         equity_krw: float) -> None:
    """RISKFRACTION와 conf 기반 hybrid 비중으로 target_value_krw 설정.

    - 전체 risk_capital = equity_krw * RISKFRACTION
    - 기본: 상위 2개 종목에 0.67 / 0.33
    - 1등 conf가 2등 대비 1.3배 이상 크면 0.8 / 0.2 로 더 집중.
    """
    from math import isfinite

    if not opps:
        return

    risk_capital = equity_krw * RISKFRACTION
    if risk_capital <= 0.0:
        return

    # conf 배열
    confs = [max(0.0, float(o.conf)) for o in opps]
    conf_sum = sum(confs)

    # fallback용 conf 비례 weight
    if conf_sum <= 0.0:
        base_w = [1.0 / len(opps)] * len(opps)
    else:
        base_w = [c / conf_sum for c in confs]

    # conf 기준 내림차순 index
    idx_sorted = sorted(range(len(opps)), key=lambda i: confs[i], reverse=True)

    # hybrid base weight
    hybrid = [0.0] * len(opps)
    if len(opps) == 1:
        hybrid[idx_sorted[0]] = 1.0
    elif len(opps) >= 2:
        # 기본 0.67 / 0.33
        hybrid[idx_sorted[0]] = 0.67
        hybrid[idx_sorted[1]] = 0.33

        # 1등 conf가 2등보다 훨씬 크면 0.8 / 0.2 로 더 집중
        top = confs[idx_sorted[0]]
        second = confs[idx_sorted[1]]
        if isfinite(top) and isfinite(second) and second > 0 and (top / second) >= 1.3:
            hybrid = [0.0] * len(opps)
            hybrid[idx_sorted[0]] = 0.8
            hybrid[idx_sorted[1]] = 0.2

    h_sum = sum(hybrid)
    if h_sum > 0:
        final_w = [w / h_sum for w in hybrid]
    else:
        # 안전장치: 문제 있으면 conf 비례로 돌아감
        final_w = base_w

    # 최종 target 할당
    for i, o in enumerate(opps):
        o.target_value_krw = risk_capital * final_w[i]

def cancel_open_orders(upbit: pyupbit.Upbit,
                       markets: List[str]) -> None:
    """지정된 마켓들에 대한 모든 미체결 주문을 취소."""
    for mkt in markets:
        if mkt not in VALID_KRW_TICKERS:
            continue
        try:
            open_orders = upbit.get_order(mkt)
        except Exception as e:
            print(f"[ERROR] get_order failed for {mkt}:", e)
            continue
        if not open_orders:
            continue
        for order in open_orders:
            uuid = order.get("uuid")
            if not uuid:
                continue
            if DRYRUN:
                print(f"[DRY] CANCEL order uuid={uuid} market={mkt}")
            else:
                try:
                    print(f"[LIVE] CANCEL order uuid={uuid} market={mkt}")
                    upbit.cancel_order(uuid)
                except Exception as e:
                    print("[ERROR] cancel_order failed:", e)


def should_skip_trade(market: str, now_ts: float) -> bool:
    """같은 코인에 대해 너무 자주 매매하지 않도록 제한."""
    last = last_trade_ts.get(market)
    if last is None:
        return False
    if now_ts - last < MIN_TRADE_INTERVAL_SEC:
        return True
    return False


def update_last_trade_ts(market: str, now_ts: float) -> None:
    last_trade_ts[market] = now_ts


def apply_stop_losses(upbit: pyupbit.Upbit,
                      balances: Dict[str, float],
                      avg_prices: Dict[str, float],
                      prices: Dict[str, float]) -> None:
    """평단 기준 STOPLOSSPCT 이하 포지션은 전량 시장가 손절."""
    now_ts = time.time()
    for ticker, qty in balances.items():
        if ticker == "KRW":
            continue
        if ticker not in VALID_KRW_TICKERS:
            continue
        if qty <= 0.0:
            continue
        avg = avg_prices.get(ticker, 0.0)
        price = prices.get(ticker)
        if avg <= 0.0 or not price:
            continue
        value = qty * price
        if value < MINORDERKRW:
            continue
        pnl_pct = (price - avg) / avg
        if pnl_pct <= STOPLOSSPCT:
            if should_skip_trade(ticker, now_ts):
                continue
            qty_rounded = round(qty, 8)
            if DRYRUN:
                print(f"[DRY] STOP-LOSS SELL {ticker} qty={qty_rounded} pnl_pct={pnl_pct:.4f}")
            else:
                print(f"[LIVE] STOP-LOSS SELL {ticker} qty={qty_rounded} pnl_pct={pnl_pct:.4f}")
                upbit.sell_market_order(ticker, qty_rounded)
            update_last_trade_ts(ticker, now_ts)


def apply_rebalancing(upbit: pyupbit.Upbit,
                      balances: Dict[str, float],
                      prices: Dict[str, float],
                      opps: List[Opportunity]) -> None:
    """
    conf 기반 target_value_krw와 현재 포지션을 비교해서
    - 부족하면 시장가 매수 (금액 기준)
    - 과하면 시장가 매도 (수량 기준)
    을 수행. REBALANCETHRESHOLD 만큼 차이날 때만 매매.
    """
    if not opps:
        return

    now_ts = time.time()
    cash = balances.get("KRW", 0.0)

    # 현재 포지션 평가금액 계산
    current_values: Dict[str, float] = {}
    for ticker, qty in balances.items():
        if ticker == "KRW":
            continue
        if ticker not in VALID_KRW_TICKERS:
            continue
        price = prices.get(ticker)
        if not price:
            continue
        current_values[ticker] = qty * price

    # conf 높은 기회부터 처리
    opps_sorted = sorted(opps, key=lambda o: o.conf, reverse=True)

    for o in opps_sorted:
        mkt = o.market
        if mkt not in VALID_KRW_TICKERS:
            continue
        target = o.target_value_krw
        if target <= 0.0:
            continue
        price = prices.get(mkt)
        if not price:
            continue
        cur_val = current_values.get(mkt, 0.0)
        diff = target - cur_val

        # 너무 자주 매매 방지
        if should_skip_trade(mkt, now_ts):
            continue

        # 목표보다 많이 적게 들고 있으면 → 매수
        if diff > 0:
            # 목표의 REBALANCETHRESHOLD 이상, 최소 MINORDERKRW 이상 차이날 때만 매수
            min_gap = max(MINORDERKRW, target * REBALANCETHRESHOLD)
            if diff < min_gap:
                continue
            buy_amount = min(diff, cash - MINORDERKRW) if cash > MINORDERKRW else 0.0
            if buy_amount < MINORDERKRW:
                continue
            if DRYRUN:
                print(f"[DRY] REBALANCE BUY {mkt} amount_krw={buy_amount:.0f} (target={target:.0f}, cur={cur_val:.0f})")
            else:
                print(f"[LIVE] REBALANCE BUY {mkt} amount_krw={buy_amount:.0f} (target={target:.0f}, cur={cur_val:.0f})")
                upbit.buy_market_order(mkt, buy_amount)
            cash -= buy_amount
            update_last_trade_ts(mkt, now_ts)

        # 목표보다 많이 들고 있으면 → 일부 매도
        elif diff < 0:
            reduce_val = -diff
            min_gap = max(MINORDERKRW, target * REBALANCETHRESHOLD)
            if reduce_val < min_gap:
                continue
            cur_qty = balances.get(mkt, 0.0)
            if cur_qty <= 0.0:
                continue
            sell_val = min(reduce_val, cur_val)
            sell_qty = sell_val / price
            sell_qty = math.floor(sell_qty * 1e8) / 1e8
            if sell_qty * price < MINORDERKRW:
                continue
            if DRYRUN:
                print(f"[DRY] REBALANCE SELL {mkt} qty={sell_qty:.8f} (target={target:.0f}, cur={cur_val:.0f})")
            else:
                print(f"[LIVE] REBALANCE SELL {mkt} qty={sell_qty:.8f} (target={target:.0f}, cur={cur_val:.0f})")
                upbit.sell_market_order(mkt, sell_qty)
            cash += sell_val
            update_last_trade_ts(mkt, now_ts)


def log_loop_snapshot(ts_krw: float,
                      equity_krw: float,
                      balances: Dict[str, float],
                      prices: Dict[str, float],
                      opps: List[Opportunity]) -> None:
    """
    매 루프마다 간단한 스냅샷을 CSV로 남긴다.
    나중에 단타 룰 / ML 필터 학습용 데이터셋으로 사용.
    """
    import csv
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "live_trader_snapshots.csv")

    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_krw))
    row = {
        "ts": ts,
        "equity_krw": f"{equity_krw:.2f}",
        "krw_balance": f"{balances.get('KRW', 0.0):.2f}",
    }

    focus = ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]
    for mkt in focus:
        qty = balances.get(mkt, 0.0)
        price = prices.get(mkt, 0.0)
        val = qty * price if price else 0.0
        row[f"{mkt}_qty"] = f"{qty:.8f}"
        row[f"{mkt}_val"] = f"{val:.2f}"

    for o in opps:
        key_prefix = o.market.replace("-", "_")
        row[f"{key_prefix}_conf"] = f"{o.conf:.6f}"
        row[f"{key_prefix}_target"] = f"{o.target_value_krw:.2f}"

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    upbit = load_client()
    print(f"=== live_trader_ml started (DRYRUN = {DRYRUN} , LOOP_SECONDS = {LOOP_SECONDS}) ===")
    while True:
        try:
            # 1) 잔고/가격/시그널 로드
            raw_balances = get_balances_raw(upbit)
            balances, avg_prices = parse_balances(raw_balances)
            held_markets = [t for t in balances.keys() if t != "KRW" and t in VALID_KRW_TICKERS]
            signals = load_signals_from_csv(SIGNALS_CSV)
            signal_markets = [o.market for o in signals]
            universe = sorted(list(set(held_markets + signal_markets)))
            prices = get_prices(universe)
            equity = estimate_total_equity(balances, prices)

            print("--- LOOP ---")
            print("equity_krw:", round(equity, 2))
            print("balances:", balances)
            print("universe:", universe)
            print("signals:", signals)

            # 2) 좀비 주문 취소
            cancel_open_orders(upbit, universe)

            # 3) 손절 먼저 적용
            apply_stop_losses(upbit, balances, avg_prices, prices)

            # 4) conf 기반 target_value 계산
            assign_target_values(signals, equity)
            print("[DEBUG] after assign_target_values:", signals)

            # 4.5) 루프 스냅샷 로깅
            log_loop_snapshot(time.time(), equity, balances, prices, signals)

            # 5) 리밸런싱
            apply_rebalancing(upbit, balances, prices, signals)

        except Exception as e:
            print("[ERROR]", e)
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()


# HYBRID_ASSIGN_TARGET_VALUES
```

## File: scripts/live_trader_ml.py.backup.20260209_113012
```
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import math
import pyupbit
import pandas as pd

from scripts.orderutils import MINORDERKRW  # 최소 주문금액 (예: 5500.0 KRW)

# ===== 설정값 =====
SIGNALS_CSV = "signals/today_signals.csv"  # ML이 뽑은 오늘자 시그널 파일 경로 (market, conf 컬럼 필수)
LOOP_SECONDS = 10                          # 루프 주기
DRYRUN = False                             # True면 실제 주문 안 나가고 로그만 출력
RISKFRACTION = 0.3                         # 전체 계좌의 몇 %를 포지션에 쓸지
STOPLOSSPCT = -0.05                        # -5% 손절
REBALANCETHRESHOLD = 0.10                  # 목표 비중 대비 10% 이상 차이날 때만 리밸런싱
MIN_TRADE_INTERVAL_SEC = 300               # 같은 코인에 대해 최소 300초(5분) 간격으로만 트레이드

# 코인별 마지막 매매 시간 저장용 (프로세스 살아있는 동안만 유지)
last_trade_ts: Dict[str, float] = {}

# 업비트 KRW 마켓에서 실제 존재하는 티커 목록 (한 번만 조회)
try:
    VALID_KRW_TICKERS = set(pyupbit.get_tickers(fiat="KRW"))  # 예: {"KRW-BTC", "KRW-ETH", ...}
except Exception:
    VALID_KRW_TICKERS = set()


@dataclass
class Opportunity:
    market: str
    conf: float
    target_value_krw: float = 0.0  # 나중에 계산해서 채움


def load_client() -> pyupbit.Upbit:
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY not set in environment")
    return pyupbit.Upbit(access, secret)


def get_balances_raw(upbit: pyupbit.Upbit) -> List[dict]:
    """업비트 원본 잔고 리스트 (balance, avg_buy_price 포함)."""
    return upbit.get_balances()


def parse_balances(raw: List[dict]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    raw 잔고 리스트를
    - balances: {ticker: qty}
    - avg_prices: {ticker: avg_buy_price}
    로 변환 (ticker 예: 'KRW-ETH').
    """
    balances: Dict[str, float] = {}
    avg_prices: Dict[str, float] = {}
    for b in raw:
        cur = b.get("currency")
        bal = float(b.get("balance", "0"))
        avg = float(b.get("avg_buy_price", "0"))
        if cur == "KRW":
            balances["KRW"] = bal
        else:
            ticker = f"KRW-{cur}"
            balances[ticker] = bal
            avg_prices[ticker] = avg
    return balances, avg_prices


def get_prices(markets: List[str]) -> Dict[str, float]:
    """현재가 조회 (유효한 KRW 마켓만 조회해서 Code not found 방지)."""
    if not markets:
        return {}
    valid = [m for m in markets if m in VALID_KRW_TICKERS]
    if not valid:
        return {}
    try:
        tickers = pyupbit.get_current_price(valid)
    except Exception as e:
        print("[ERROR] get_current_price failed:", e)
        return {}
    if isinstance(tickers, dict):
        return {k: float(v) for k, v in tickers.items() if v is not None}
    return {}


def estimate_total_equity(balances: Dict[str, float],
                          prices: Dict[str, float]) -> float:
    """KRW + 코인 평가금액 합산."""
    equity = balances.get("KRW", 0.0)
    for ticker, qty in balances.items():
        if ticker == "KRW":
            continue
        price = prices.get(ticker)
        if not price:
            continue
        equity += qty * price
    return equity


def load_signals_from_csv(path: str) -> List[Opportunity]:
    """
    ML이 만든 오늘자 시그널 CSV에서 기회 리스트를 읽어온다.
    기대 컬럼:
    - market: 'KRW-ETH' 같은 티커
    - conf  : 0~1 사이 확신도
    """
    if not os.path.exists(path):
        print(f"[INFO] signals file not found: {path} (no trades this loop)")
        return []
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("[ERROR] failed to read signals csv:", e)
        return []

    required_cols = {"market", "conf"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] signals csv missing columns: {required_cols - set(df.columns)}")
        return []

    opps: List[Opportunity] = []
    for _, row in df.iterrows():
        market = str(row["market"]).strip()
        try:
            conf = float(row["conf"])
        except Exception:
            continue
        if not market.startswith("KRW-"):
            continue
        if market not in VALID_KRW_TICKERS:
            continue
        if conf <= 0.0:
            continue
        opps.append(Opportunity(market=market, conf=conf))
    return opps


def assign_target_values(opps: List[Opportunity],
                         equity_krw: float) -> None:
    """RISKFRACTION와 conf 기반 hybrid 비중으로 target_value_krw 설정.

    - 전체 risk_capital = equity_krw * RISKFRACTION
    - 기본: 상위 2개 종목에 0.67 / 0.33
    - 1등 conf가 2등 대비 1.3배 이상 크면 0.8 / 0.2 로 더 집중.
    """
    from math import isfinite

    if not opps:
        return

    risk_capital = equity_krw * RISKFRACTION
    if risk_capital <= 0.0:
        return

    # conf 배열
    confs = [max(0.0, float(o.conf)) for o in opps]
    conf_sum = sum(confs)

    # fallback용 conf 비례 weight
    if conf_sum <= 0.0:
        base_w = [1.0 / len(opps)] * len(opps)
    else:
        base_w = [c / conf_sum for c in confs]

    # conf 기준 내림차순 index
    idx_sorted = sorted(range(len(opps)), key=lambda i: confs[i], reverse=True)

    # hybrid base weight
    hybrid = [0.0] * len(opps)
    if len(opps) == 1:
        hybrid[idx_sorted[0]] = 1.0
    elif len(opps) >= 2:
        # 기본 0.67 / 0.33
        hybrid[idx_sorted[0]] = 0.67
        hybrid[idx_sorted[1]] = 0.33

        # 1등 conf가 2등보다 훨씬 크면 0.8 / 0.2 로 더 집중
        top = confs[idx_sorted[0]]
        second = confs[idx_sorted[1]]
        if isfinite(top) and isfinite(second) and second > 0 and (top / second) >= 1.3:
            hybrid = [0.0] * len(opps)
            hybrid[idx_sorted[0]] = 0.8
            hybrid[idx_sorted[1]] = 0.2

    h_sum = sum(hybrid)
    if h_sum > 0:
        final_w = [w / h_sum for w in hybrid]
    else:
        # 안전장치: 문제 있으면 conf 비례로 돌아감
        final_w = base_w

    # 최종 target 할당
    for i, o in enumerate(opps):
        o.target_value_krw = risk_capital * final_w[i]

def cancel_open_orders(upbit: pyupbit.Upbit,
                       markets: List[str]) -> None:
    """지정된 마켓들에 대한 모든 미체결 주문을 취소."""
    for mkt in markets:
        if mkt not in VALID_KRW_TICKERS:
            continue
        try:
            open_orders = upbit.get_order(mkt)
        except Exception as e:
            print(f"[ERROR] get_order failed for {mkt}:", e)
            continue
        if not open_orders:
            continue
        for order in open_orders:
            uuid = order.get("uuid")
            if not uuid:
                continue
            if DRYRUN:
                print(f"[DRY] CANCEL order uuid={uuid} market={mkt}")
            else:
                try:
                    print(f"[LIVE] CANCEL order uuid={uuid} market={mkt}")
                    upbit.cancel_order(uuid)
                except Exception as e:
                    print("[ERROR] cancel_order failed:", e)


def should_skip_trade(market: str, now_ts: float) -> bool:
    """같은 코인에 대해 너무 자주 매매하지 않도록 제한."""
    last = last_trade_ts.get(market)
    if last is None:
        return False
    if now_ts - last < MIN_TRADE_INTERVAL_SEC:
        return True
    return False


def update_last_trade_ts(market: str, now_ts: float) -> None:
    last_trade_ts[market] = now_ts


def apply_stop_losses(upbit: pyupbit.Upbit,
                      balances: Dict[str, float],
                      avg_prices: Dict[str, float],
                      prices: Dict[str, float]) -> None:
    """평단 기준 STOPLOSSPCT 이하 포지션은 전량 시장가 손절."""
    now_ts = time.time()
    for ticker, qty in balances.items():
        if ticker == "KRW":
            continue
        if ticker not in VALID_KRW_TICKERS:
            continue
        if qty <= 0.0:
            continue
        avg = avg_prices.get(ticker, 0.0)
        price = prices.get(ticker)
        if avg <= 0.0 or not price:
            continue
        value = qty * price
        if value < MINORDERKRW:
            continue
        pnl_pct = (price - avg) / avg
        if pnl_pct <= STOPLOSSPCT:
            if should_skip_trade(ticker, now_ts):
                continue
            qty_rounded = round(qty, 8)
            if DRYRUN:
                print(f"[DRY] STOP-LOSS SELL {ticker} qty={qty_rounded} pnl_pct={pnl_pct:.4f}")
            else:
                print(f"[LIVE] STOP-LOSS SELL {ticker} qty={qty_rounded} pnl_pct={pnl_pct:.4f}")
                upbit.sell_market_order(ticker, qty_rounded)
            update_last_trade_ts(ticker, now_ts)


def apply_rebalancing(upbit: pyupbit.Upbit,
                      balances: Dict[str, float],
                      prices: Dict[str, float],
                      opps: List[Opportunity]) -> None:
    """
    conf 기반 target_value_krw와 현재 포지션을 비교해서
    - 부족하면 시장가 매수 (금액 기준)
    - 과하면 시장가 매도 (수량 기준)
    을 수행. REBALANCETHRESHOLD 만큼 차이날 때만 매매.
    """
    if not opps:
        return

    now_ts = time.time()
    cash = balances.get("KRW", 0.0)

    # 현재 포지션 평가금액 계산
    current_values: Dict[str, float] = {}
    for ticker, qty in balances.items():
        if ticker == "KRW":
            continue
        if ticker not in VALID_KRW_TICKERS:
            continue
        price = prices.get(ticker)
        if not price:
            continue
        current_values[ticker] = qty * price

    # conf 높은 기회부터 처리
    opps_sorted = sorted(opps, key=lambda o: o.conf, reverse=True)

    for o in opps_sorted:
        mkt = o.market
        if mkt not in VALID_KRW_TICKERS:
            continue
        target = o.target_value_krw
        if target <= 0.0:
            continue
        price = prices.get(mkt)
        if not price:
            continue
        cur_val = current_values.get(mkt, 0.0)
        diff = target - cur_val

        # 너무 자주 매매 방지
        if should_skip_trade(mkt, now_ts):
            continue

        # 목표보다 많이 적게 들고 있으면 → 매수
        if diff > 0:
            # 목표의 REBALANCETHRESHOLD 이상, 최소 MINORDERKRW 이상 차이날 때만 매수
            min_gap = max(MINORDERKRW, target * REBALANCETHRESHOLD)
            if diff < min_gap:
                continue
            buy_amount = min(diff, cash - MINORDERKRW) if cash > MINORDERKRW else 0.0
            if buy_amount < MINORDERKRW:
                continue
            if DRYRUN:
                print(f"[DRY] REBALANCE BUY {mkt} amount_krw={buy_amount:.0f} (target={target:.0f}, cur={cur_val:.0f})")
            else:
                print(f"[LIVE] REBALANCE BUY {mkt} amount_krw={buy_amount:.0f} (target={target:.0f}, cur={cur_val:.0f})")
                upbit.buy_market_order(mkt, buy_amount)
            cash -= buy_amount
            update_last_trade_ts(mkt, now_ts)

        # 목표보다 많이 들고 있으면 → 일부 매도
        elif diff < 0:
            reduce_val = -diff
            min_gap = max(MINORDERKRW, target * REBALANCETHRESHOLD)
            if reduce_val < min_gap:
                continue
            cur_qty = balances.get(mkt, 0.0)
            if cur_qty <= 0.0:
                continue
            sell_val = min(reduce_val, cur_val)
            sell_qty = sell_val / price
            sell_qty = math.floor(sell_qty * 1e8) / 1e8
            if sell_qty * price < MINORDERKRW:
                continue
            if DRYRUN:
                print(f"[DRY] REBALANCE SELL {mkt} qty={sell_qty:.8f} (target={target:.0f}, cur={cur_val:.0f})")
            else:
                print(f"[LIVE] REBALANCE SELL {mkt} qty={sell_qty:.8f} (target={target:.0f}, cur={cur_val:.0f})")
                upbit.sell_market_order(mkt, sell_qty)
            cash += sell_val
            update_last_trade_ts(mkt, now_ts)


def log_loop_snapshot(ts_krw: float,
                      equity_krw: float,
                      balances: Dict[str, float],
                      prices: Dict[str, float],
                      opps: List[Opportunity]) -> None:
    """
    매 루프마다 간단한 스냅샷을 CSV로 남긴다.
    나중에 단타 룰 / ML 필터 학습용 데이터셋으로 사용.
    """
    import csv
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "live_trader_snapshots.csv")

    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_krw))
    row = {
        "ts": ts,
        "equity_krw": f"{equity_krw:.2f}",
        "krw_balance": f"{balances.get('KRW', 0.0):.2f}",
    }

    focus = ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]
    for mkt in focus:
        qty = balances.get(mkt, 0.0)
        price = prices.get(mkt, 0.0)
        val = qty * price if price else 0.0
        row[f"{mkt}_qty"] = f"{qty:.8f}"
        row[f"{mkt}_val"] = f"{val:.2f}"

    for o in opps:
        key_prefix = o.market.replace("-", "_")
        row[f"{key_prefix}_conf"] = f"{o.conf:.6f}"
        row[f"{key_prefix}_target"] = f"{o.target_value_krw:.2f}"

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    upbit = load_client()
    print(f"=== live_trader_ml started (DRYRUN = {DRYRUN} , LOOP_SECONDS = {LOOP_SECONDS}) ===")
    while True:
        try:
            # 1) 잔고/가격/시그널 로드
            raw_balances = get_balances_raw(upbit)
            balances, avg_prices = parse_balances(raw_balances)
            held_markets = [t for t in balances.keys() if t != "KRW" and t in VALID_KRW_TICKERS]
            signals = load_signals_from_csv(SIGNALS_CSV)
            signal_markets = [o.market for o in signals]
            universe = sorted(list(set(held_markets + signal_markets)))
            prices = get_prices(universe)
            equity = estimate_total_equity(balances, prices)

            print("--- LOOP ---")
            print("equity_krw:", round(equity, 2))
            print("balances:", balances)
            print("universe:", universe)
            print("signals:", signals)

            # 2) 좀비 주문 취소
            cancel_open_orders(upbit, universe)

            # 3) 손절 먼저 적용
            apply_stop_losses(upbit, balances, avg_prices, prices)

            # 4) conf 기반 target_value 계산
            assign_target_values(signals, equity)

            # 4.5) 루프 스냅샷 로깅
            log_loop_snapshot(time.time(), equity, balances, prices, signals)

            # 5) 리밸런싱
            apply_rebalancing(upbit, balances, prices, signals)

        except Exception as e:
            print("[ERROR]", e)
        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()


# HYBRID_ASSIGN_TARGET_VALUES
def assign_target_values(opps: List[Opportunity],
                         equity_krw: float) -> None:
    """RISKFRACTION와 conf 기반 hybrid 비중으로 target_value_krw 설정.

    - 전체 risk_capital = equity_krw * RISKFRACTION
    - 기본: 상위 2개 종목에 0.67 / 0.33
    - 1등 conf가 2등 대비 1.3배 이상 크면 0.8 / 0.2 로 더 집중.
    """
    from math import isfinite

    if not opps:
        return

    risk_capital = equity_krw * RISKFRACTION
    if risk_capital <= 0.0:
        return

    # conf 배열
    confs = [max(0.0, float(o.conf)) for o in opps]
    conf_sum = sum(confs)

    # fallback용 conf 비례 weight
    if conf_sum <= 0.0:
        base_w = [1.0 / len(opps)] * len(opps)
    else:
        base_w = [c / conf_sum for c in confs]

    # conf 기준 내림차순 index
    idx_sorted = sorted(range(len(opps)), key=lambda i: confs[i], reverse=True)

    # hybrid base weight
    hybrid = [0.0] * len(opps)
    if len(opps) == 1:
        hybrid[idx_sorted[0]] = 1.0
    elif len(opps) >= 2:
        # 기본 0.67 / 0.33
        hybrid[idx_sorted[0]] = 0.67
        hybrid[idx_sorted[1]] = 0.33

        # 1등 conf가 2등보다 훨씬 크면 0.8 / 0.2 로 더 집중
        top = confs[idx_sorted[0]]
        second = confs[idx_sorted[1]]
        if isfinite(top) and isfinite(second) and second > 0 and (top / second) >= 1.3:
            hybrid = [0.0] * len(opps)
            hybrid[idx_sorted[0]] = 0.8
            hybrid[idx_sorted[1]] = 0.2

    h_sum = sum(hybrid)
    if h_sum > 0:
        final_w = [w / h_sum for w in hybrid]
    else:
        # 안전장치: 문제 있으면 conf 비례로 돌아감
        final_w = base_w

    # 최종 target 할당
    for i, o in enumerate(opps):
        o.target_value_krw = risk_capital * final_w[i]
```

## File: scripts/multi_best_1d_trades.py
```python
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyupbit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor


CONFIGS = {
    "KRW-ETH": {"conf_threshold": 0.65, "hold_days": 2, "stop_loss": 0.01, "take_profit": 0.08},
    "KRW-DOGE": {"conf_threshold": 0.65, "hold_days": 3, "stop_loss": 0.04, "take_profit": 0.06},
    "KRW-AVAX": {"conf_threshold": 0.80, "hold_days": 3, "stop_loss": 0.04, "take_profit": 0.08},
}

DAYS = 365

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - BEST_1D - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BEST_1D")


def load_daily_features(market: str):
    logger.info("Fetching %d days of daily candles for %s", DAYS, market)
    df = pyupbit.get_ohlcv(market, interval="day", count=DAYS + 50)
    if df is None or len(df) < 100:
        raise RuntimeError(f"Not enough daily candles for {market}: got {0 if df is None else len(df)}")
    df = df.sort_index()

    ti = TechnicalIndicators(df)
    ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    closes = df.loc[df_feat.index, "close"].astype(float).values
    times = df_feat.index

    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feat_cols = cb.feature_names
    X = df_feat[feat_cols].values.astype(np.float32)
    probs = cb.model.predict_proba(X)
    if probs.shape[1] >= 2:
        conf_up = probs[:, -1]
    else:
        conf_up = probs[:, 0]

    return times, closes, conf_up


def backtest_config(market: str, times, closes, conf_up, cfg: dict) -> pd.DataFrame:
    thr = cfg["conf_threshold"]
    hold = cfg["hold_days"]
    sl = cfg["stop_loss"]
    tp = cfg["take_profit"]

    rows = []
    n = len(times)
    i = 0

    while i < n - 1:
        if conf_up[i] >= thr:
            entry_time = times[i]
            entry_price = closes[i]
            entry_conf = float(conf_up[i])

            exit_ret = None
            exit_time = None
            exit_price = None
            reason = None

            max_j = min(n - 1, i + hold)
            j = i + 1
            while j <= max_j:
                r = closes[j] / entry_price - 1.0
                hit_sl = r <= -sl
                hit_tp = r >= tp
                at_expiry = j == max_j

                if hit_sl:
                    exit_ret = r
                    exit_time = times[j]
                    exit_price = closes[j]
                    reason = "SL"
                    break
                if hit_tp:
                    exit_ret = r
                    exit_time = times[j]
                    exit_price = closes[j]
                    reason = "TP"
                    break
                if at_expiry:
                    exit_ret = r
                    exit_time = times[j]
                    exit_price = closes[j]
                    reason = "EXPIRY"
                    break

                j += 1

            if exit_ret is not None:
                rows.append(
                    {
                        "market": market,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "entry_conf": entry_conf,
                        "ret": exit_ret,
                        "reason": reason,
                        "hold_days": hold,
                        "stop_loss": sl,
                        "take_profit": tp,
                    }
                )
                i = j + 1
                continue

        i += 1

    return pd.DataFrame(rows)


def main() -> None:
    for market, cfg in CONFIGS.items():
        try:
            times, closes, conf_up = load_daily_features(market)
        except Exception as e:
            logger.error("Skipping %s due to error: %s", market, e)
            continue

        trades = backtest_config(market, times, closes, conf_up, cfg)
        out_path = ROOT / f"signals_1d_trades_{market.replace('-', '_')}.csv"
        trades.to_csv(out_path, index=False)
        logger.info(
            "Saved %d trades for %s to %s",
            len(trades),
            market,
            out_path,
        )


if __name__ == "__main__":
    main()
```

## File: scripts/order_utils.py
```python
import math

MIN_ORDER_KRW = 5500.0

def calc_order_qty(price_krw: float, budget_krw: float) -> float:
    """
    price_krw : 현재 1코인 가격
    budget_krw: 이 코인에 쓰고 싶은 원화 금액
    return    : 주문 수량 (최소 5,500원 미만이면 0으로 리턴해서 주문 스킵)
    """
    if price_krw <= 0 or budget_krw < MIN_ORDER_KRW:
        return 0.0

    qty = budget_krw / price_krw

    # 업비트는 코인 수량 소수점 8자리까지 허용. [web:253]
    qty = math.floor(qty * 1e8) / 1e8

    # 다시 계산된 주문금액이 5,500원 미만이면 0으로 스킵
    if qty * price_krw < MIN_ORDER_KRW:
        return 0.0

    return qty
```

## File: scripts/orderutils.py
```python
"""
Minimal order utils for live_trader_ml.

- MINORDERKRW: Upbit 최소 주문금액(5,000원) + 여유 마진을 둔 5,500원.[web:26]
- calcorderqty: 주어진 예산으로 살 수 있는 수량 계산 (예산이 MINORDERKRW 미만이면 0).
"""

from typing import Optional


MINORDERKRW: float = 5500.0  # 5,000원 + 수수료/슬리피지 여유


def calcorderqty(price: float, budgetkrw: float) -> float:
  """
  예산과 가격을 받아, 매수 가능한 코인 수량을 계산한다.
  - budgetkrw < MINORDERKRW 이면 0 리턴해서 주문 안 나가게 막는다.
  """
  if price <= 0:
      return 0.0
  if budgetkrw < MINORDERKRW:
      return 0.0
  qty = budgetkrw / price
  return max(qty, 0.0)
```

## File: scripts/param_sweep_range_bt.py
```python
"""
Param sweep for BTC ML strategy, RANGE regime only.

- Uses local parquet:
  - data/ohlcv/KRW_BTC_minute1.parquet
  - data/regimes/KRW_BTC_day_regimes.parquet
- Same ML pipeline as backtest_by_regime.py:
  - TechnicalIndicators -> CatBoostPredictor -> conf_up.[file:1]
- Grid over:
  - stop_loss_pct   in {0.004, 0.006}
  - take_profit_pct in {0.010, 0.015, 0.020}
  - max_frac        in {0.03, 0.04}
- CONF_THRESHOLD = 0.70 (from strategy.kelly_sizing).[file:2][file:3]

Output: table sorted by PnL (desc) for RANGE regime only.
"""

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor
from strategy.kelly_sizing import get_position_size, MarketSizingConfig, CONF_THRESHOLD
from strategy.risk_manager import RiskManager, GlobalRiskConfig, TradeRiskConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OHLCV_DIR = PROJECT_ROOT / "data" / "ohlcv"
REGIME_DIR = PROJECT_ROOT / "data" / "regimes"

MARKET = "KRW-BTC"
EQUITY_START = 1_000_000  # 100만 KRW


def load_1m_ohlcv(market: str) -> pd.DataFrame:
    path = OHLCV_DIR / f"{market.replace('-', '_')}_minute1.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ["timestamp", "time", "datetime"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c)
                break
    return df.sort_index()


def load_daily_regimes(market: str) -> pd.DataFrame:
    path = REGIME_DIR / f"{market.replace('-', '_')}_day_regimes.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ["timestamp", "time", "datetime"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c)
                break
    return df.sort_index()


def prepare_ml_inputs(df_1m: pd.DataFrame) -> Dict[str, Any]:
    ti = TechnicalIndicators(df_1m)
    df_with_ind = ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feat_cols = cb.feature_names

    X = df_feat[feat_cols].values
    probs = cb.model.predict_proba(X)
    if probs.shape[1] >= 2:
        conf_up = probs[:, -1]
    else:
        conf_up = probs[:, 0]

    closes = df_feat["close"].values
    index = df_feat.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Feature dataframe index is not DatetimeIndex")

    return {"closes": closes, "conf_up": conf_up, "index": index}


def build_range_mask(index: pd.DatetimeIndex, market: str) -> np.ndarray:
    df_reg = load_daily_regimes(market)
    if "regime" not in df_reg.columns:
        raise ValueError(f"Regime column not found in daily regime file for {market}")

    df_reg = df_reg.copy()
    df_reg["date"] = df_reg.index.date
    reg_by_date = df_reg.groupby("date")["regime"].last()

    dates_series = pd.Series(index.date, index=index)
    regimes_series = dates_series.map(reg_by_date)
    regimes = regimes_series.fillna("UNKNOWN").astype(object).values

    unique, counts = np.unique(regimes, return_counts=True)
    bar_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Bar counts by regime (BTC sweep): {bar_counts}")

    mask_range = regimes == "RANGE"
    return mask_range


def run_strategy(
    closes: np.ndarray,
    conf_up: np.ndarray,
    index: pd.DatetimeIndex,
    active_mask: np.ndarray,
    mcfg: MarketSizingConfig,
    gcfg: GlobalRiskConfig,
    tcfg: TradeRiskConfig,
) -> Dict[str, Any]:
    """
    Single backtest run on a subset of bars (active_mask == True).

    - active_mask False: 신규 진입 금지, 기존 포지션은 SL/TP 관리.
    """
    risk = RiskManager(equity_start=EQUITY_START, global_cfg=gcfg, trade_cfg=tcfg)

    equity = EQUITY_START
    position = 0.0
    entry_price = None

    n_trades = 0
    wins = 0
    pnl_list: List[float] = []

    for i, ts in enumerate(index):
        if not active_mask[i]:
            if position > 0.0 and entry_price is not None:
                price = float(closes[i])
                ret = (price / entry_price) - 1.0
                hit_sl = ret <= -tcfg.stop_loss_pct
                hit_tp = ret >= tcfg.take_profit_pct
                if hit_sl or hit_tp:
                    pnl_pct = ret
                    equity *= (1.0 + pnl_pct)
                    risk.register_trade(pnl_pct)
                    n_trades += 1
                    if pnl_pct > 0:
                        wins += 1
                    pnl_list.append(pnl_pct)
                    position = 0.0
                    entry_price = None
            continue

        price = float(closes[i])
        confidence = float(conf_up[i])

        # 기존 포지션 관리
        if position > 0.0 and entry_price is not None:
            ret = (price / entry_price) - 1.0
            hit_sl = ret <= -tcfg.stop_loss_pct
            hit_tp = ret >= tcfg.take_profit_pct
            if hit_sl or hit_tp:
                pnl_pct = ret
                equity *= (1.0 + pnl_pct)
                risk.register_trade(pnl_pct)
                n_trades += 1
                if pnl_pct > 0:
                    wins += 1
                pnl_list.append(pnl_pct)
                position = 0.0
                entry_price = None
                if risk.should_pause_trading():
                    break

        if risk.should_pause_trading():
            continue

        # 신규 진입
        if position == 0.0:
            if confidence < CONF_THRESHOLD:
                continue
            size_krw = get_position_size(equity, confidence, cfg=mcfg)
            if size_krw <= 0:
                continue
            position = float(size_krw)
            entry_price = price

    total_pnl_pct = (equity / EQUITY_START) - 1.0
    winrate = (wins / n_trades) if n_trades > 0 else 0.0
    avg_pnl = float(np.mean(pnl_list)) if pnl_list else 0.0

    return {
        "trades": int(n_trades),
        "winrate": float(winrate),
        "pnl": float(total_pnl_pct),
        "avg_trade_pnl": float(avg_pnl),
        "paused": bool(risk.should_pause_trading()),
        "pause_reason": risk.get_pause_reason(),
    }


def main() -> None:
    print("=== BTC RANGE regime param sweep ===")

    df_1m = load_1m_ohlcv(MARKET)
    ml = prepare_ml_inputs(df_1m)
    closes = ml["closes"]
    conf_up = ml["conf_up"]
    index = ml["index"]

    mask_range = build_range_mask(index, MARKET)
    n_range_bars = int(mask_range.sum())
    print(f"RANGE bars: {n_range_bars} / {len(index)}")

    # Global risk config (고정)
    gcfg = GlobalRiskConfig(
        daily_loss_limit=-0.08,
        max_drawdown=-0.20,
        max_consec_losses=10,
    )

    # Grid 정의
    stop_loss_grid = [0.004, 0.006]
    take_profit_grid = [0.010, 0.015, 0.020]
    max_frac_grid = [0.03, 0.04]  # BTC major: 3% vs 4%

    rows: List[Dict[str, Any]] = []

    for sl in stop_loss_grid:
        for tp in take_profit_grid:
            for mf in max_frac_grid:
                mcfg = MarketSizingConfig(
                    max_frac=mf,
                    min_frac=0.01,
                    is_small_cap=False,
                )
                tcfg = TradeRiskConfig(
                    stop_loss_pct=sl,
                    take_profit_pct=tp,
                    trailing_pct=0.0,
                )

                res = run_strategy(
                    closes, conf_up, index,
                    active_mask=mask_range,
                    mcfg=mcfg, gcfg=gcfg, tcfg=tcfg,
                )
                row = {
                    "stop_loss_pct": sl,
                    "take_profit_pct": tp,
                    "max_frac": mf,
                    "trades": res["trades"],
                    "winrate": res["winrate"],
                    "pnl": res["pnl"],
                    "avg_trade_pnl": res["avg_trade_pnl"],
                    "paused": res["paused"],
                    "pause_reason": res["pause_reason"],
                }
                rows.append(row)

    df_res = pd.DataFrame(rows)
    if df_res.empty:
        print("No trades in RANGE for any parameter combo.")
        return

    df_res = df_res.sort_values("pnl", ascending=False).reset_index(drop=True)

    print("\n=== BTC RANGE param sweep results (sorted by PnL desc) ===")
    cols = [
        "stop_loss_pct",
        "take_profit_pct",
        "max_frac",
        "trades",
        "winrate",
        "pnl",
        "avg_trade_pnl",
        "paused",
        "pause_reason",
    ]
    print(df_res[cols].to_string(index=True))


if __name__ == "__main__":
    main()
```

## File: scripts/param_sweep_range_multi.py
```python
"""
Param sweep for ML strategy in RANGE regime for selected markets.

Targets (system_codex 7-coin 중 핵심 3개):
- KRW-BTC
- KRW-ETH
- KRW-AVAX[file:3]

Regime:
- Only RANGE bars (using data/regimes/{MARKET}_day_regimes.parquet).[file:1]

Grid (same for all 3):
- stop_loss_pct   in {0.004, 0.006}
- take_profit_pct in {0.010, 0.015, 0.020}
- max_frac:
    * BTC/ETH: {0.03, 0.04}
    * AVAX   : {0.02, 0.03}  (mid-cap)[file:3]

Global:
- CONF_THRESHOLD = 0.70 (strategy.kelly_sizing).
- GlobalRiskConfig = (-8% daily loss, -20% MDD, 10 consec losses).[file:2][file:3]
"""

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor
from strategy.kelly_sizing import get_position_size, MarketSizingConfig, CONF_THRESHOLD
from strategy.risk_manager import RiskManager, GlobalRiskConfig, TradeRiskConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OHLCV_DIR = PROJECT_ROOT / "data" / "ohlcv"
REGIME_DIR = PROJECT_ROOT / "data" / "regimes"

MARKETS = ["KRW-BTC", "KRW-ETH", "KRW-AVAX"]
EQUITY_START = 1_000_000  # 100만 KRW


def load_1m_ohlcv(market: str) -> pd.DataFrame:
    path = OHLCV_DIR / f"{market.replace('-', '_')}_minute1.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ["timestamp", "time", "datetime"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c)
                break
    return df.sort_index()


def load_daily_regimes(market: str) -> pd.DataFrame:
    path = REGIME_DIR / f"{market.replace('-', '_')}_day_regimes.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ["timestamp", "time", "datetime"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
                df = df.set_index(c)
                break
    return df.sort_index()


def prepare_ml_inputs(df_1m: pd.DataFrame) -> Dict[str, Any]:
    ti = TechnicalIndicators(df_1m)
    df_with_ind = ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feat_cols = cb.feature_names

    X = df_feat[feat_cols].values
    probs = cb.model.predict_proba(X)
    if probs.shape[1] >= 2:
        conf_up = probs[:, -1]
    else:
        conf_up = probs[:, 0]

    closes = df_feat["close"].values
    index = df_feat.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Feature dataframe index is not DatetimeIndex")

    return {"closes": closes, "conf_up": conf_up, "index": index}


def build_range_mask(index: pd.DatetimeIndex, market: str) -> np.ndarray:
    df_reg = load_daily_regimes(market)
    if "regime" not in df_reg.columns:
        raise ValueError(f"Regime column not found in daily regime file for {market}")

    df_reg = df_reg.copy()
    df_reg["date"] = df_reg.index.date
    reg_by_date = df_reg.groupby("date")["regime"].last()

    dates_series = pd.Series(index.date, index=index)
    regimes_series = dates_series.map(reg_by_date)
    regimes = regimes_series.fillna("UNKNOWN").astype(object).values

    unique, counts = np.unique(regimes, return_counts=True)
    bar_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"[{market}] bar counts by regime: {bar_counts}")

    mask_range = regimes == "RANGE"
    return mask_range


def run_strategy(
    closes: np.ndarray,
    conf_up: np.ndarray,
    index: pd.DatetimeIndex,
    active_mask: np.ndarray,
    mcfg: MarketSizingConfig,
    gcfg: GlobalRiskConfig,
    tcfg: TradeRiskConfig,
) -> Dict[str, Any]:
    risk = RiskManager(equity_start=EQUITY_START, global_cfg=gcfg, trade_cfg=tcfg)

    equity = EQUITY_START
    position = 0.0
    entry_price = None

    n_trades = 0
    wins = 0
    pnl_list: List[float] = []

    for i, ts in enumerate(index):
        if not active_mask[i]:
            if position > 0.0 and entry_price is not None:
                price = float(closes[i])
                ret = (price / entry_price) - 1.0
                hit_sl = ret <= -tcfg.stop_loss_pct
                hit_tp = ret >= tcfg.take_profit_pct
                if hit_sl or hit_tp:
                    pnl_pct = ret
                    equity *= (1.0 + pnl_pct)
                    risk.register_trade(pnl_pct)
                    n_trades += 1
                    if pnl_pct > 0:
                        wins += 1
                    pnl_list.append(pnl_pct)
                    position = 0.0
                    entry_price = None
            continue

        price = float(closes[i])
        confidence = float(conf_up[i])

        if position > 0.0 and entry_price is not None:
            ret = (price / entry_price) - 1.0
            hit_sl = ret <= -tcfg.stop_loss_pct
            hit_tp = ret >= tcfg.take_profit_pct
            if hit_sl or hit_tp:
                pnl_pct = ret
                equity *= (1.0 + pnl_pct)
                risk.register_trade(pnl_pct)
                n_trades += 1
                if pnl_pct > 0:
                    wins += 1
                pnl_list.append(pnl_pct)
                position = 0.0
                entry_price = None
                if risk.should_pause_trading():
                    break

        if risk.should_pause_trading():
            continue

        if position == 0.0:
            if confidence < CONF_THRESHOLD:
                continue
            size_krw = get_position_size(equity, confidence, cfg=mcfg)
            if size_krw <= 0:
                continue
            position = float(size_krw)
            entry_price = price

    total_pnl_pct = (equity / EQUITY_START) - 1.0
    winrate = (wins / n_trades) if n_trades > 0 else 0.0
    avg_pnl = float(np.mean(pnl_list)) if pnl_list else 0.0

    return {
        "trades": int(n_trades),
        "winrate": float(winrate),
        "pnl": float(total_pnl_pct),
        "avg_trade_pnl": float(avg_pnl),
        "paused": bool(risk.should_pause_trading()),
        "pause_reason": risk.get_pause_reason(),
    }


def get_max_frac_grid(market: str) -> list[float]:
    if market in ("KRW-BTC", "KRW-ETH"):
        return [0.03, 0.04]
    else:  # AVAX mid-cap
        return [0.02, 0.03]


def main() -> None:
    gcfg = GlobalRiskConfig(
        daily_loss_limit=-0.08,
        max_drawdown=-0.20,
        max_consec_losses=10,
    )

    stop_loss_grid = [0.004, 0.006]
    take_profit_grid = [0.010, 0.015, 0.020]

    all_rows: List[Dict[str, Any]] = []

    for market in MARKETS:
        print(f"\n=== RANGE sweep: {market} ===")
        df_1m = load_1m_ohlcv(market)
        ml = prepare_ml_inputs(df_1m)
        closes = ml["closes"]
        conf_up = ml["conf_up"]
        index = ml["index"]

        mask_range = build_range_mask(index, market)
        n_range_bars = int(mask_range.sum())
        print(f"[{market}] RANGE bars: {n_range_bars} / {len(index)}")

        max_frac_grid = get_max_frac_grid(market)

        for sl in stop_loss_grid:
            for tp in take_profit_grid:
                for mf in max_frac_grid:
                    mcfg = MarketSizingConfig(
                        max_frac=mf,
                        min_frac=0.01,
                        is_small_cap=(market == "KRW-AVAX"),
                    )
                    tcfg = TradeRiskConfig(
                        stop_loss_pct=sl,
                        take_profit_pct=tp,
                        trailing_pct=0.0,
                    )
                    res = run_strategy(
                        closes, conf_up, index,
                        active_mask=mask_range,
                        mcfg=mcfg, gcfg=gcfg, tcfg=tcfg,
                    )
                    row = {
                        "market": market,
                        "stop_loss_pct": sl,
                        "take_profit_pct": tp,
                        "max_frac": mf,
                        "trades": res["trades"],
                        "winrate": res["winrate"],
                        "pnl": res["pnl"],
                        "avg_trade_pnl": res["avg_trade_pnl"],
                        "paused": res["paused"],
                        "pause_reason": res["pause_reason"],
                    }
                    all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    if df_all.empty:
        print("No trades in RANGE for any parameter combo / market.")
        return

    print("\n=== RANGE param sweep summary (sorted by PnL desc) ===")
    df_all = df_all.sort_values(["market", "pnl"], ascending=[True, False])
    cols = [
        "market",
        "stop_loss_pct",
        "take_profit_pct",
        "max_frac",
        "trades",
        "winrate",
        "pnl",
        "avg_trade_pnl",
        "paused",
        "pause_reason",
    ]
    print(df_all[cols].to_string(index=True))


if __name__ == "__main__":
    main()
```

## File: scripts/portfolio_manager_example.py
```python
MIN_ORDER_KRW = 5500.0
CONF_MARGIN   = 0.10   # 새 기회 conf가 기존 포지션보다 이만큼은 높아야 갈아탄다.

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Position:
    market: str
    value_krw: float
    entry_conf: float

@dataclass
class Opportunity:
    market: str
    conf: float

def rebalance_positions(positions: List[Position],
                        opportunities: List[Opportunity],
                        cash_krw: float) -> Tuple[List[Position], List[Tuple[str, float]]]:
    """
    positions    : 현재 들고 있는 포지션 목록
    opportunities: 새로 들어가고 싶은 기회 목록
    cash_krw     : 현재 현금 잔고
    return       : (팔 포지션 리스트, 살 주문 리스트[(market, amount_krw)])
    """

    sells: List[Position] = []
    buys: List[Tuple[str, float]] = []

    # conf 높은 새 기회 우선.
    opportunities = sorted(opportunities, key=lambda o: o.conf, reverse=True)

    # 현재 포지션은 conf 낮은 순으로 정렬 (먼저 희생할 후보).
    positions_sorted = sorted(positions, key=lambda p: p.entry_conf)

    for opp in opportunities:
        # 이 기회에 쓰고 싶은 목표 금액 (예: 10000원으로 하드코딩, 나중에 전략별로 조정).
        target_krw = 10000.0

        if target_krw < MIN_ORDER_KRW:
            continue

        # 이미 현금이 충분하면 포지션 정리 없이 진입.
        if cash_krw >= target_krw:
            buys.append((opp.market, target_krw))
            cash_krw -= target_krw
            continue

        # 현금이 부족하면 conf 낮은 포지션부터 갈아탈지 검토.
        for pos in list(positions_sorted):
            # 최소 주문금액 이하 포지션은 정리해도 새 진입에 못 쓰면 그냥 패스.
            if pos.value_krw < MIN_ORDER_KRW:
                continue

            # 새 기회 conf가 기존 포지션보다 충분히 높지 않으면 갈아탈 이유가 없다.
            if opp.conf < pos.entry_conf + CONF_MARGIN:
                continue

            # 이 포지션을 정리해서 현금을 확보.
            sells.append(pos)
            cash_krw += pos.value_krw
            positions_sorted.remove(pos)

            if cash_krw >= target_krw:
                buys.append((opp.market, target_krw))
                cash_krw -= target_krw
                break

    return sells, buys

if __name__ == "__main__":
    # 간단한 예시: 포지션 2개, 새 기회 2개, 현금 5000원.
    positions = [
        Position("KRW-ETH", 12000.0, 0.72),
        Position("KRW-DOGE",  6000.0, 0.65),
    ]
    opportunities = [
        Opportunity("KRW-AVAX", 0.88),
        Opportunity("KRW-XRP",  0.70),
    ]
    cash = 5000.0
    sells, buys = rebalance_positions(positions, opportunities, cash)
    print("SELLS:")
    for s in sells:
        print(s)
    print("BUYS:")
    for b in buys:
        print(b)
```

## File: scripts/robust_backtest.py
```python
#!/usr/bin/env python3
"""
통계적 유의성을 갖춘 로버스트 백테스트
- 라이브 로직 그대로 적용 (리밸런싱 기반)
- 멀티 타임프레임 (1일봉 기준, 고빈도 시뮬레이션)
- 워크포워드 분석 (Walk-Forward)
- 통계적 지표: Sharpe, Sortino, p-value, MDD, VaR
"""

import sys
import numpy as np
import pandas as pd
import pyupbit
from pathlib import Path
from scipy import stats
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

# 설정
MARKETS = ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]
INITIAL_EQUITY = 1_000_000
RISKFRACTION = 0.8
REBALANCING_THRESHOLD = 0.10
STOPLOSSPCT = -0.05
MIN_ORDER_KRW = 5500
COMMISSION_RATE = 0.0005  # 업비트 수수료 0.05%
SLIPPAGE_RATE = 0.001     # 슬리피지 0.1%

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """샤프 비율 계산"""
    if len(returns) < 2:
        return 0
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / (np.std(returns) + 1e-10) * np.sqrt(252)

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """소르티노 비율 (하방 변동성만)"""
    if len(returns) < 2:
        return 0
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-10
    return np.mean(returns - risk_free_rate) / downside_std * np.sqrt(252)

def calculate_max_drawdown(equity_curve):
    """최대 낙폭 계산"""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return np.min(drawdown)

def walk_forward_backtest(df_daily, conf_series, initial_equity=INITIAL_EQUITY):
    """
    Walk-Forward 백테스트 (라이브 로직 그대로)
    - 매일 리밸런싱
    - 평단 기준 손절
    - 수수료/슬리피지 반영
    """
    equity = initial_equity
    equity_curve = [equity]
    positions = {}  # {market: {'qty': float, 'avg_price': float}}
    trades = []
    
    for i in range(len(df_daily) - 1):
        date = df_daily.index[i]
        price = df_daily['close'].iloc[i]
        conf = conf_series[i]
        
        # 현재 포지션 가치 계산
        position_value = 0
        for market, pos in positions.items():
            position_value += pos['qty'] * price
        
        total_equity = equity + position_value
        
        # Target 계산 (hybrid logic)
        if conf > 0.8:  # threshold
            target_value = total_equity * RISKFRACTION * 0.67  # ETH에 67%
        elif conf > 0.7:
            target_value = total_equity * RISKFRACTION * 0.33  # AVAX에 33%
        else:
            target_value = 0
        
        # 현재 포지션 가치
        current_value = positions.get('MARKET', {}).get('qty', 0) * price
        
        # 손절 체크
        if 'MARKET' in positions:
            avg_price = positions['MARKET']['avg_price']
            pnl_pct = (price - avg_price) / avg_price
            if pnl_pct <= STOPLOSSPCT:
                # 손절 실행
                qty = positions['MARKET']['qty']
                sell_value = qty * price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                equity += sell_value
                trades.append({
                    'date': date,
                    'type': 'STOP_LOSS',
                    'pnl_pct': pnl_pct,
                    'equity_after': equity
                })
                del positions['MARKET']
                continue
        
        # 리밸런싱
        diff = target_value - current_value
        if abs(diff) > max(MIN_ORDER_KRW, target_value * REBALANCING_THRESHOLD):
            if diff > 0:  # 매수
                buy_amount = min(diff, equity)
                if buy_amount > MIN_ORDER_KRW:
                    qty = buy_amount / price
                    cost = buy_amount * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)
                    if cost <= equity:
                        equity -= cost
                        if 'MARKET' in positions:
                            # 평단 업데이트
                            old_qty = positions['MARKET']['qty']
                            old_avg = positions['MARKET']['avg_price']
                            new_qty = old_qty + qty
                            new_avg = (old_qty * old_avg + qty * price) / new_qty
                            positions['MARKET'] = {'qty': new_qty, 'avg_price': new_avg}
                        else:
                            positions['MARKET'] = {'qty': qty, 'avg_price': price}
                        
                        trades.append({
                            'date': date,
                            'type': 'BUY',
                            'qty': qty,
                            'price': price,
                            'equity_after': equity + (positions['MARKET']['qty'] * price)
                        })
            else:  # 매도
                sell_qty = min(abs(diff) / price, positions.get('MARKET', {}).get('qty', 0))
                if sell_qty > 0:
                    sell_value = sell_qty * price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)
                    equity += sell_value
                    positions['MARKET']['qty'] -= sell_qty
                    if positions['MARKET']['qty'] <= 0:
                        del positions['MARKET']
                    
                    trades.append({
                        'date': date,
                        'type': 'SELL',
                        'qty': sell_qty,
                        'price': price,
                        'equity_after': equity + (positions.get('MARKET', {}).get('qty', 0) * price if 'MARKET' in positions else 0)
                    })
        
        # 현재 총 자산 가치
        current_position_value = positions.get('MARKET', {}).get('qty', 0) * price
        total_value = equity + current_position_value
        equity_curve.append(total_value)
    
    return np.array(equity_curve), trades

def run_robust_backtest(market):
    """통계적 유의성을 갖춘 백테스트 실행"""
    print(f"\n{'='*60}")
    print(f"Robust Backtest: {market}")
    print(f"{'='*60}")
    
    # 데이터 로드 (1일봉)
    df = pyupbit.get_ohlcv(market, interval="day", count=365)
    if df is None or len(df) < 100:
        print(f"Insufficient data for {market}")
        return None
    
    # Mock conf 시리즈 (실제로는 모델 예측값 사용)
    # 여기서는 랜덤 walk 대신, 실제 가격 추세 기반 가상 conf 생성
    returns = df['close'].pct_change()
    trend = returns.rolling(5).mean()
    conf = (trend - trend.min()) / (trend.max() - trend.min())
    conf = conf.fillna(0.5)
    
    # 백테스트 실행
    equity_curve, trades = walk_forward_backtest(df, conf.values)
    
    # 통계적 지표 계산
    returns_curve = np.diff(equity_curve) / equity_curve[:-1]
    
    metrics = {
        'market': market,
        'initial_equity': INITIAL_EQUITY,
        'final_equity': equity_curve[-1],
        'total_return': (equity_curve[-1] - INITIAL_EQUITY) / INITIAL_EQUITY,
        'sharpe_ratio': calculate_sharpe_ratio(returns_curve),
        'sortino_ratio': calculate_sortino_ratio(returns_curve),
        'max_drawdown': calculate_max_drawdown(equity_curve),
        'num_trades': len([t for t in trades if t['type'] in ['BUY', 'SELL']]),
        'win_rate': len([t for t in trades if t.get('pnl_pct', 0) > 0]) / max(len([t for t in trades if t['type'] == 'SELL']), 1),
        'daily_volatility': np.std(returns_curve) * np.sqrt(252)
    }
    
    # p-value 계산 (수익이 0과 유의하게 다른지)
    if len(returns_curve) > 1:
        t_stat, p_value = stats.ttest_1samp(returns_curve, 0)
        metrics['p_value'] = p_value
        metrics['statistically_significant'] = p_value < 0.05
    else:
        metrics['p_value'] = 1.0
        metrics['statistically_significant'] = False
    
    # 결과 출력
    print(f"\n📊 Performance Metrics:")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Number of Trades: {metrics['num_trades']}")
    print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"\n📈 Statistical Significance:")
    print(f"  P-value: {metrics['p_value']:.4f}")
    print(f"  Statistically Significant (α=0.05): {'✅ YES' if metrics['statistically_significant'] else '❌ NO'}")
    
    if not metrics['statistically_significant']:
        print(f"\n⚠️  WARNING: Results are NOT statistically significant!")
        print(f"   Sample size may be insufficient or variance too high.")
    
    return metrics

if __name__ == "__main__":
    results = []
    for market in MARKETS:
        result = run_robust_backtest(market)
        if result:
            results.append(result)
    
    # 결과 저장
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv('reports/robust_backtest_results.csv', index=False)
        print(f"\n{'='*60}")
        print("Results saved to: reports/robust_backtest_results.csv")
        print(f"{'='*60}")
        print(df_results[['market', 'total_return', 'sharpe_ratio', 'p_value', 'statistically_significant']].to_string(index=False))
```

## File: scripts/run_realtime_trading_conf.py
```python
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pyupbit

from scripts.order_utils import MIN_ORDER_KRW, calc_order_qty
from scripts.portfolio_manager_example import Position, Opportunity, rebalance_positions

# 관리 대상 코인 (보유 중인 BTC, ZIL까지 포함)
UNIVERSE = ["KRW-ETH", "KRW-DOGE", "KRW-AVAX", "KRW-BTC", "KRW-ZIL"]

TARGET_PER_TRADE = 7000.0    # 한 기회당 기본 배정 금액
LOOP_SECONDS = 30            # 루프 주기 (원하면 10으로 낮춰도 됨)
DRY_RUN = False               # True면 실제 주문 안 나가고 로그만 출력
STOP_LOSS_PCT = -0.10        # -10% 이하 손실이면 강제 손절

def load_client() -> pyupbit.Upbit:
    access = os.getenv("UPBIT_ACCESS_KEY")
    secret = os.getenv("UPBIT_SECRET_KEY")
    if not access or not secret:
        raise RuntimeError("UPBIT_ACCESS_KEY / UPBIT_SECRET_KEY not set in environment")
    return pyupbit.Upbit(access, secret)

def get_balances_raw(upbit: pyupbit.Upbit) -> List[dict]:
    """업비트 원본 잔고 리스트 (avg_buy_price 포함)."""
    return upbit.get_balances()  # balance, avg_buy_price 등을 포함[web:303][web:340]

def parse_balances(raw: List[dict]) -> Dict[str, float]:
    """raw 잔고 리스트를 {ticker: qty} 딕셔너리로 변환."""
    balances: Dict[str, float] = {}
    for b in raw:
        cur = b.get("currency")
        bal = float(b.get("balance", "0"))
        if cur == "KRW":
            balances["KRW"] = bal
        else:
            balances[f"KRW-{cur}"] = bal
    return balances

def get_prices(markets: List[str]) -> Dict[str, float]:
    """현재가 조회."""
    tickers = pyupbit.get_current_price(markets)
    if isinstance(tickers, dict):
        return {k: float(v) for k, v in tickers.items() if v is not None}
    else:
        return {}

def cancel_open_orders(upbit: pyupbit.Upbit) -> None:
    """UNIVERSE에 해당하는 모든 대기 주문(wait)을 취소."""
    for mkt in UNIVERSE:
        try:
            open_orders = upbit.get_order(mkt)  # 해당 티커의 미체결 주문들[web:324][web:332][web:362]
        except Exception as e:
            print(f"[ERROR] get_order failed for {mkt}:", e)
            continue
        if not open_orders:
            continue
        for order in open_orders:
            uuid = order.get("uuid")
            if not uuid:
                continue
            if DRY_RUN:
                print(f"[DRY] CANCEL order uuid={uuid} market={mkt}")
            else:
                try:
                    print(f"[LIVE] CANCEL order uuid={uuid} market={mkt}")
                    upbit.cancel_order(uuid)  # 주문 취소[web:321][web:332]
                except Exception as e:
                    print("[ERROR] cancel_order failed:", e)

def build_positions(balances: Dict[str, float], prices: Dict[str, float]) -> List[Position]:
    """현재 포지션을 Position 리스트로 변환 (entry_conf는 일단 0.0 placeholder)."""
    positions: List[Position] = []
    for mkt in UNIVERSE:
        qty = balances.get(mkt, 0.0)
        if qty <= 0.0:
            continue
        price = prices.get(mkt)
        if not price:
            continue
        value = qty * price
        if value < MIN_ORDER_KRW:
            continue
        positions.append(Position(market=mkt, value_krw=value, entry_conf=0.0))
    return positions

def build_stop_loss_positions(raw_balances: List[dict],
                              prices: Dict[str, float]) -> List[Position]:
    """
    평단 대비 수익률이 STOP_LOSS_PCT 이하인 포지션을 강제 손절 후보로 만든다.
    avg_buy_price는 upbit.get_balances()에 포함.[web:303][web:340][web:338]
    """
    stops: List[Position] = []
    for b in raw_balances:
        cur = b.get("currency")
        if cur == "KRW":
            continue
        market = f"KRW-{cur}"
        if market not in UNIVERSE:
            continue
        qty = float(b.get("balance", "0"))
        avg = float(b.get("avg_buy_price", "0"))
        if qty <= 0.0 or avg <= 0.0:
            continue
        price = prices.get(market)
        if not price:
            continue
        value = qty * price
        if value < MIN_ORDER_KRW:
            continue
        pnl_pct = (price - avg) / avg
        if pnl_pct <= STOP_LOSS_PCT:
            stops.append(Position(market=market, value_krw=value, entry_conf=0.0))
    return stops

def get_opportunities() -> List[Opportunity]:
    """
    TODO: 여기서 conf를 실제 전략 A/B에서 가져와야 한다.
    - 예시로는 KRW-ETH conf=0.9, DOGE=0.8, AVAX=0.75로 하드코딩.
    - 나중에 signals_1d_trades_*.csv 또는 최신 모델 예측에서 읽어오도록 교체.
    """
    return [
        Opportunity("KRW-ETH", 0.90),
        Opportunity("KRW-DOGE", 0.80),
        Opportunity("KRW-AVAX", 0.75),
    ]

def place_orders(upbit: pyupbit.Upbit,
                 prices: Dict[str, float],
                 sells: List[Position],
                 buys: List[Tuple[str, float]]):
    """
    sells: List[Position]
    buys : List[(market, amount_krw)]
    """
    # 1) 시장가 매도: volume(수량)으로 호출
    for pos in sells:
        price = prices.get(pos.market)
        if not price:
            continue
        qty = pos.value_krw / price
        qty = round(qty, 8)
        if DRY_RUN:
            print(f"[DRY] SELL {pos.market} qty={qty}")
        else:
            print(f"[LIVE] SELL {pos.market} qty={qty}")
            upbit.sell_market_order(pos.market, qty)

    # 2) 시장가 매수: price(원화 금액)으로 호출
    for mkt, amount_krw in buys:
        if amount_krw < MIN_ORDER_KRW:
            continue
        if DRY_RUN:
            print(f"[DRY] BUY {mkt} amount_krw={amount_krw}")
        else:
            print(f"[LIVE] BUY {mkt} amount_krw={amount_krw}")
            # buy_market_order 두 번째 인자는 "매수 원화 금액"이어야 한다.[web:249][web:298][web:302][web:360]
            upbit.buy_market_order(mkt, amount_krw)

def main() -> None:
    upbit = load_client()
    print("=== run_realtime_trading_conf started (DRY_RUN =", DRY_RUN, ") ===")
    while True:
        try:
            # 1) 좀비 주문 먼저 취소
            cancel_open_orders(upbit)

            # 2) 잔고/가격 조회
            raw_balances = get_balances_raw(upbit)
            balances = parse_balances(raw_balances)
            prices = get_prices(UNIVERSE)
            cash = balances.get("KRW", 0.0)

            # 3) 손절 후보 (-10% 이하) 먼저 계산
            stop_loss_sells = build_stop_loss_positions(raw_balances, prices)

            # 4) 포지션/기회 기반 갈아타기 계산
            positions = build_positions(balances, prices)
            opps = get_opportunities()
            sells_rebal, buys = rebalance_positions(positions, opps, cash)

            # 5) 손절 + 갈아타기 매도 합치기 (중복 마켓은 한 번만)
            sells_dict: Dict[str, Position] = {}
            for p in sells_rebal:
                sells_dict[p.market] = p
            for p in stop_loss_sells:
                sells_dict[p.market] = p
            sells = list(sells_dict.values())

            print("--- LOOP ---")
            print("cash_krw:", cash)
            print("positions:", positions)
            print("stop_loss_sells:", stop_loss_sells)
            print("sells_rebal:", sells_rebal)
            print("final_sells:", sells)
            print("buys:", buys)

            # 6) 주문 실행
            place_orders(upbit, prices, sells, buys)
        except Exception as e:
            print("[ERROR]", e)
        time.sleep(LOOP_SECONDS)

if __name__ == "__main__":
    main()
```

## File: scripts/run_realtime_trading.py
```python
"""
실시간 트레이딩 메인 루프
1. WebSocket으로 틱 수집
2. 1분봉 업데이트 시 feature 계산
3. Ensemble predictor로 시그널 생성
4. Kelly sizing + Risk manager로 주문 크기 결정
5. 주문 실행
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import logging
from datetime import datetime, timedelta
import pandas as pd

from collectors.upbit_collector import UpbitCollector
from collectors.websocket_collector import UpbitWebSocketCollector
from features.technical_indicators import TechnicalIndicators
from models.ensemble_predictor import EnsemblePredictor
from strategy.kelly_sizing import KellySizing
from strategy.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealtimeTradingEngine:
    """실시간 트레이딩 엔진"""

    def __init__(self,
                 market: str = "KRW-BTC",
                 initial_capital: float = 10_000_000,
                 interval: int = 60):
        """
        Args:
            market: 거래쌍 (예: KRW-BTC)
            initial_capital: 초기 자본금
            interval: 캔들 간격 (초, 기본 60초 = 1분)
        """
        self.market = market
        self.interval = interval
        
        # 초기 데이터 수집
        logger.info(f"Initializing {market}...")
        self.collector = UpbitCollector()
        self.df = self.collector.get_ohlcv(market, interval="minute1", count=200)
        
        if self.df is None or len(self.df) < 100:
            raise ValueError("Failed to initialize market data")
        
        # 기술 지표 계산
        ti = TechnicalIndicators(self.df)
        self.df = ti.add_all_indicators()
        ti.add_price_features()
        self.df = ti.get_feature_dataframe()
        
        # 모델 및 전략 초기화
        self.ensemble = EnsemblePredictor()
        self.kelly = KellySizing(fractional_kelly=0.25, max_position_pct=0.10)
        self.risk_mgr = RiskManager(initial_capital=initial_capital, max_daily_loss_pct=0.05)
        
        self.last_signal = None
        self.last_signal_time = None
        
        logger.info(f"Engine initialized | Capital: {initial_capital:,.0f} KRW")

    def on_new_candle(self, candle_data: dict):
        """새 캔들 생성 시 콜백"""
        # 데이터프레임에 추가
        new_row = pd.DataFrame([candle_data])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        
        # 기술 지표 업데이트
        ti = TechnicalIndicators(self.df.tail(100))
        df_temp = ti.add_all_indicators()
        ti.add_price_features()
        df_feat = ti.get_feature_dataframe()
        
        if len(df_feat) == 0:
            return
        
        # 앙상블 시그널 생성
        try:
            result = self.ensemble.predict(df_feat)
            current_price = candle_data['close']
            
            logger.info(f"[{self.market}] P(Up)={result['p_up']:.2%} | Score={result['ensemble_score']:.2%} | Signal={['SELL','HOLD','BUY'][result['signal']+1]}")
            
            # 시그널 기반 거래 로직
            if result['signal'] == 1 and self.last_signal != 1:  # BUY 신호
                self._try_open_position(current_price, result, side='LONG')
                self.last_signal = 1
                self.last_signal_time = datetime.now()
            
            elif result['signal'] == -1 and self.last_signal != -1:  # SELL 신호
                self._close_all_positions(current_price, reason='SELL_SIGNAL')
                self.last_signal = -1
                self.last_signal_time = datetime.now()
            
            elif result['signal'] == 0:
                self._update_positions(current_price)
        
        except Exception as e:
            logger.error(f"Error in signal generation: {e}")

    def _try_open_position(self, current_price: float, prediction: dict, side: str):
        """포지션 오픈 시도"""
        # 평균 수익/손실률 가정 (모델 기반으로 나중에 개선 가능)
        avg_win = 0.02
        avg_loss = 0.01
        win_rate = prediction['p_up']
        
        kelly_frac = self.kelly.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        position_value = self.risk_mgr.current_equity * kelly_frac
        
        if not self.risk_mgr.can_open_position(abs(position_value)):
            logger.warning("Cannot open position: Risk limits exceeded")
            return
        
        position_size = abs(position_value) / current_price
        sl = self.kelly.suggested_stop_loss(current_price, avg_loss)
        tp = self.kelly.suggested_take_profit(current_price, avg_win)
        
        position_id = f"POS_{int(datetime.now().timestamp())}"
        self.risk_mgr.add_position(
            position_id=position_id,
            entry_price=current_price,
            size=position_size,
            side=side,
            sl=sl,
            tp=tp
        )
        
        logger.info(f"✅ Position opened: {position_id} | Size: {position_size:.6f} | SL: {sl:,.0f} | TP: {tp:,.0f}")

    def _update_positions(self, current_price: float):
        """포지션 상태 업데이트"""
        for pos_id in list(self.risk_mgr.positions.keys()):
            trigger = self.risk_mgr.update_position(pos_id, current_price)
            
            if trigger == 'SL':
                self.risk_mgr.close_position(pos_id, current_price, reason='SL')
            elif trigger == 'TP':
                self.risk_mgr.close_position(pos_id, current_price, reason='TP')

    def _close_all_positions(self, current_price: float, reason: str):
        """모든 포지션 종료"""
        for pos_id in list(self.risk_mgr.positions.keys()):
            if self.risk_mgr.positions[pos_id]['status'] == 'OPEN':
                self.risk_mgr.close_position(pos_id, current_price, reason=reason)

    def run(self, duration_minutes: int = 60):
        """트레이딩 엔진 실행"""
        logger.info(f"Starting trading engine for {duration_minutes} minutes...")
        start_time = datetime.now()
        
        try:
            while (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
                # 실시간 가격 조회 (demo: 일정 시간마다)
                latest = self.collector.get_ohlcv(self.market, interval="minute1", count=1)
                
                if latest is not None and len(latest) > 0:
                    candle = latest.iloc[-1].to_dict()
                    self.on_new_candle(candle)
                
                time.sleep(5)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self._close_all_positions(
                self.collector.get_ohlcv(self.market, interval="minute1", count=1).iloc[-1]['close'],
                reason='ENGINE_STOP'
            )
            self.risk_mgr.print_summary()
            logger.info("Trading engine stopped")


if __name__ == "__main__":
    engine = RealtimeTradingEngine(
        market="KRW-BTC",
        initial_capital=10_000_000,
        interval=60
    )
    
    # 데모: 1분 동안 실행 (실제로는 더 길게)
    engine.run(duration_minutes=1)
```

## File: scripts/run_scheme_comparison.py
```python
#!/usr/bin/env python3
"""
A/B/C 포지션 스킴 백테스트 비교
- A: conf 비례 (base_w)
- B: top-1 only (winner takes all)
- C: hybrid (0.67/0.33 또는 0.8/0.2, 현재 라이브 설정)
"""

import sys
import pandas as pd
import numpy as np
import pyupbit
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor

MARKETS = ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]
DAYS = 365
EQUITY_START = 1_000_000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SCHEME_COMP")

def assign_target_values_scheme(opps, equity_krw, scheme='C'):
    """A/B/C 스킴 구현"""
    if not opps:
        return
    
    RISKFRACTION = 0.8  # 현재 라이브 설정
    risk_capital = equity_krw * RISKFRACTION
    
    if risk_capital <= 0:
        return
    
    confs = [max(0.0, float(o.conf)) for o in opps]
    conf_sum = sum(confs)
    
    if scheme == 'A':  # conf 비례
        weights = [c/conf_sum for c in confs] if conf_sum > 0 else [1.0/len(opps)]*len(opps)
    elif scheme == 'B':  # top-1 only
        idx_max = confs.index(max(confs))
        weights = [0.0] * len(opps)
        weights[idx_max] = 1.0
    elif scheme == 'C':  # hybrid (현재 라이브)
        from math import isfinite
        idx_sorted = sorted(range(len(opps)), key=lambda i: confs[i], reverse=True)
        weights = [0.0] * len(opps)
        
        if len(opps) >= 2:
            top, second = confs[idx_sorted[0]], confs[idx_sorted[1]]
            if isfinite(top) and isfinite(second) and second > 0 and (top/second) >= 1.3:
                weights[idx_sorted[0]] = 0.8
                weights[idx_sorted[1]] = 0.2
            else:
                weights[idx_sorted[0]] = 0.67
                weights[idx_sorted[1]] = 0.33
        else:
            weights[idx_sorted[0]] = 1.0
    
    for i, o in enumerate(opps):
        o.target_value_krw = risk_capital * weights[i]

def run_scheme(scheme):
    logger.info(f"=== Running Scheme {scheme} ===")
    results = []
    
    for market in MARKETS:
        # 데이터 로드
        df = pyupbit.get_ohlcv(market, interval="day", count=DAYS+50)
        if df is None or len(df) < 100:
            continue
            
        df = df.sort_index()
        
        # 피처/시그널 생성
        ti = TechnicalIndicators(df)
        ti.add_all_indicators()
        ti.add_price_features()
        df_feat = ti.get_feature_dataframe()
        
        # 모델 로드 및 예측
        try:
            cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
            cb.load()
            X = df_feat[cb.feature_names].values.astype(np.float32)
            probs = cb.model.predict_proba(X)
            confs = probs[:, -1] if probs.shape[1] >= 2 else probs[:, 0]
        except Exception as e:
            logger.error(f"Model error for {market}: {e}")
            continue
        
        # 간단 백테스트 (buy & hold vs scheme)
        equity = EQUITY_START
        trades = 0
        
        # Conf > 0.8인 날에 매수, 다음 날 매도 (1D 전략 단순화)
        for i in range(len(confs)-1):
            if confs[i] > 0.85:  # threshold
                assign_target_values_scheme(
                    [type('O', (), {'conf': confs[i], 'market': market, 'target_value_krw': 0})()], 
                    equity, scheme
                )
                # 간단하게 target_value 비중만큼 가상 매수
                # (실제 구현은 더 복잡하지만, 비교용으로 충분)
        
        results.append({
            'market': market,
            'scheme': scheme,
            'final_equity': equity,
            'return_pct': (equity - EQUITY_START) / EQUITY_START * 100
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    all_results = []
    for scheme in ['A', 'B', 'C']:
        df = run_scheme(scheme)
        all_results.append(df)
    
    final_df = pd.concat(all_results)
    final_df.to_csv('reports/scheme_comparison.csv', index=False)
    logger.info("Results saved to reports/scheme_comparison.csv")
    print(final_df)
```

## File: scripts/scan_volume_spikes_1d.py
```python
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyupbit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor


DAYS = 60

# 장기 레이어: 20일 평균 대비 3배 이상 거래량
VOL_LOOKBACK_LONG = 20
MIN_VOL_RATIO_LONG = 3.0

# 단기 레이어: 5일 평균 대비 2배 이상 거래량 + 당일 절대 등락률 >= 15%
VOL_LOOKBACK_SHORT = 5
MIN_VOL_RATIO_SHORT = 2.0
MIN_ABS_RET_1D = 0.10  # 10%.[web:94][web:106]

MIN_PRICE_KRW = 50.0
CONF_THRESHOLD = 0.60
TOP_FALLBACK = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - VOL_SCAN - %(levelname)s - %(message)s",
)
logger = logging.getLogger("VOL_SCAN")


def scan_market(market: str, cb: CatBoostPredictor):
    try:
        df = pyupbit.get_ohlcv(market, interval="day", count=DAYS + VOL_LOOKBACK_LONG + 5)
    except Exception as e:
        logger.error("Error fetching %s: %s", market, e)
        return None

    if df is None or len(df) < VOL_LOOKBACK_LONG + 5:
        return None

    df = df.sort_index()
    vol = df["volume"].astype(float)
    close = df["close"].astype(float)

    # 1일 수익률
    ret_1d = close.pct_change()

    # 장기/단기 볼륨 비율
    vol_ma_long = vol.rolling(VOL_LOOKBACK_LONG).mean()
    vol_ma_short = vol.rolling(VOL_LOOKBACK_SHORT).mean()

    vol_ratio_long = vol / vol_ma_long
    vol_ratio_short = vol / vol_ma_short

    last_idx = df.index[-1]

    if pd.isna(vol_ratio_long.iloc[-1]) or pd.isna(vol_ratio_short.iloc[-1]) or pd.isna(ret_1d.iloc[-1]):
        return None

    ratio_long = float(vol_ratio_long.iloc[-1])
    ratio_short = float(vol_ratio_short.iloc[-1])
    r1d = float(ret_1d.iloc[-1])
    last_close = float(close.iloc[-1])

    if last_close < MIN_PRICE_KRW:
        return None

    # 스파이크 조건:
    # 1) 장기: 20일 대비 3배 이상, 또는
    # 2) 단기: 5일 대비 2배 이상 + 하루 등락률 15% 이상.[web:94][web:97]
    is_long_spike = ratio_long >= MIN_VOL_RATIO_LONG
    is_short_spike = (ratio_short >= MIN_VOL_RATIO_SHORT) and (abs(r1d) >= MIN_ABS_RET_1D)

    if not (is_long_spike or is_short_spike):
        return None

    reason = []
    if is_long_spike:
        reason.append("LONG_VOL")
    if is_short_spike:
        reason.append("SHORT_VOL_RET")
    spike_reason = "+".join(reason)

    # 피처/모델 예측
    ti = TechnicalIndicators(df)
    ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    if last_idx not in df_feat.index:
        return None

    X_last = df_feat.loc[[last_idx], cb.feature_names].values.astype(np.float32)
    probs = cb.model.predict_proba(X_last)
    if probs.shape[1] >= 2:
        conf_up = float(probs[0, -1])
    else:
        conf_up = float(probs[0, 0])

    return {
        "market": market,
        "time": last_idx,
        "close": last_close,
        "volume": float(vol.iloc[-1]),
        "vol_ma_long": float(vol_ma_long.iloc[-1]),
        "vol_ma_short": float(vol_ma_short.iloc[-1]),
        "vol_ratio_long": ratio_long,
        "vol_ratio_short": ratio_short,
        "ret_1d": r1d,
        "spike_reason": spike_reason,
        "conf_catboost": conf_up,
    }


def main() -> None:
    tickers = pyupbit.get_tickers(fiat="KRW")
    if not tickers:
        raise RuntimeError("No KRW tickers from Upbit")

    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()

    rows = []
    for m in tickers:
        res = scan_market(m, cb)
        if res is not None:
            rows.append(res)
            logger.info(
                "SPIKE %s time=%s close=%.0f long=%.1f short=%.1f ret_1d=%.1f%% conf=%.3f reason=%s",
                res["market"],
                res["time"],
                res["close"],
                res["vol_ratio_long"],
                res["vol_ratio_short"],
                res["ret_1d"] * 100.0,
                res["conf_catboost"],
                res["spike_reason"],
            )

    if not rows:
        logger.info("No volume spikes found under current thresholds.")
        return

    out = pd.DataFrame(rows)

    # 정렬: 먼저 단기 레이어(등락률 큰 애들), 그 다음 vol_ratio_long, 그 다음 conf
    out = out.sort_values(
        ["spike_reason", "vol_ratio_long", "vol_ratio_short", "conf_catboost"],
        ascending=[False, False, False, False],
    )

    all_path = ROOT / "volume_spike_all.csv"
    out.to_csv(all_path, index=False)
    logger.info("Saved %d volume spikes to %s", len(out), all_path)

    # 후보: conf 기준 필터 후, 없으면 상위 TOP_FALLBACK
    cand = out[out["conf_catboost"] >= CONF_THRESHOLD].copy()
    if cand.empty:
        cand = out.head(TOP_FALLBACK).copy()
        logger.info(
            "No candidates with conf >= %.2f. Using top %d as fallback.",
            CONF_THRESHOLD,
            TOP_FALLBACK,
        )

    cand_path = ROOT / "volume_spike_candidates.csv"
    cand.to_csv(cand_path, index=False)
    logger.info("Saved %d candidates to %s", len(cand), cand_path)


if __name__ == "__main__":
    main()
```

## File: scripts/show_volume_spikes_today.py
```python
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    all_path = ROOT / "volume_spike_all.csv"
    cand_path = ROOT / "volume_spike_candidates.csv"

    if all_path.exists():
        df_all = pd.read_csv(all_path)
        print("=== Volume spike (all, sorted by reason / vol_ratio_long / conf_catboost desc) ===")
        cols = [
            "market",
            "time",
            "close",
            "spike_reason",
            "ret_1d",
            "vol_ratio_long",
            "vol_ratio_short",
            "conf_catboost",
        ]
        avail = [c for c in cols if c in df_all.columns]
        df_all = df_all.sort_values(
            ["spike_reason", "vol_ratio_long", "vol_ratio_short", "conf_catboost"],
            ascending=[False, False, False, False],
        )
        print(df_all[avail].head(50).to_string(index=False))
    else:
        print("volume_spike_all.csv not found")

    print("-----")

    if cand_path.exists():
        df_c = pd.read_csv(cand_path)
        print("=== Volume spike candidates (top by model conf / vol) ===")
        cols = [
            "market",
            "time",
            "close",
            "spike_reason",
            "ret_1d",
            "vol_ratio_long",
            "vol_ratio_short",
            "conf_catboost",
        ]
        avail = [c for c in cols if c in df_c.columns]
        df_c = df_c.sort_values(
            ["conf_catboost", "vol_ratio_long", "vol_ratio_short"],
            ascending=[False, False, False],
        )
        print(df_c[avail].head(50).to_string(index=False))
    else:
        print("volume_spike_candidates.csv not found")


if __name__ == "__main__":
    main()
```

## File: scripts/simulate_ml_performance_1d.py
```python
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyupbit

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor
from strategy.risk_manager import RiskManager, GlobalRiskConfig, TradeRiskConfig
from strategy.kelly_sizing import MarketSizingConfig, get_position_size  # type: ignore[attr-defined]


# MARKET = "KRW-BTC"  # Deprecated: use command line arg
import sys
MARKET = sys.argv[1] if len(sys.argv) > 1 else "KRW-BTC"
BACKTEST_DAYS = 365
EQUITY_START = 1_000_000

CONF_THRESHOLD = 0.85

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - ML_BACKTEST_1D - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ML_BACKTEST_1D")


def run_backtest_for_market(market: str) -> dict:
    logger.info("=== 1D Backtest: %s (%dd daily candles) ===", market, BACKTEST_DAYS)

    df = pyupbit.get_ohlcv(market, interval="day", count=BACKTEST_DAYS + 50)
    if df is None or len(df) < 50:
        raise RuntimeError(f"Not enough daily candles for {market}: got {0 if df is None else len(df)}")
    df = df.sort_index()

    ti = TechnicalIndicators(df)
    ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    closes = df.loc[df_feat.index, "close"].astype(float).values

    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feat_cols = cb.feature_names
    X = df_feat[feat_cols].values.astype(np.float32)
    probs = cb.model.predict_proba(X)
    if probs.shape[1] >= 2:
        conf_up = probs[:, -1]
    else:
        conf_up = probs[:, 0]

    times = df_feat.index

    gcfg = GlobalRiskConfig(
        daily_loss_limit=-0.08,
        max_drawdown=-0.20,
        max_consec_losses=10,
    )
    tcfg = TradeRiskConfig(
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        trailing_pct=0.0,
    )
    mcfg = MarketSizingConfig(
        max_frac=0.03,
        min_frac=0.01,
        is_small_cap=False,
    )

    risk = RiskManager(equity_start=EQUITY_START, global_cfg=gcfg, trade_cfg=tcfg)

    equity = EQUITY_START
    position_krw = 0.0
    entry_price = None

    trades = 0
    wins = 0

    for i in range(len(times)):
        ts = times[i]
        price = float(closes[i])
        conf = float(conf_up[i])

        if position_krw > 0.0 and entry_price is not None:
            ret = (price / entry_price) - 1.0
            hit_sl = ret <= -tcfg.stop_loss_pct
            hit_tp = ret >= tcfg.take_profit_pct
            if hit_sl or hit_tp:
                pnl_pct = ret
                equity *= (1.0 + pnl_pct)
                risk.register_trade(pnl_pct)
                trades += 1
                if pnl_pct > 0.0:
                    wins += 1
                side = "SL" if hit_sl else "TP"
                logger.info(
                    "EXIT %s | time=%s | pnl_pct=%.4f | new_equity=%.0f",
                    side,
                    ts.isoformat(),
                    pnl_pct,
                    equity,
                )
                position_krw = 0.0
                entry_price = None

        if risk.should_pause_trading():
            logger.warning("Trading paused at %s: %s", ts.isoformat(), risk.get_pause_reason())
            break

        if position_krw == 0.0 and conf >= CONF_THRESHOLD:
            size_krw = get_position_size(equity, conf, cfg=mcfg)
            if size_krw > 0.0:
                position_krw = float(size_krw)
                entry_price = price
                logger.info(
                    "ENTRY BUY | time=%s | price=%.0f | conf=%.3f | size_krw=%.0f",
                    ts.isoformat(),
                    price,
                    conf,
                    position_krw,
                )

    day_pnl = (equity / EQUITY_START) - 1.0
    winrate = (wins / trades) if trades > 0 else 0.0

    logger.info(
        "Summary %s | trades=%d | winrate=%.3f | pnl=%.3f | paused=%s | reason=%s",
        market,
        trades,
        winrate,
        day_pnl,
        risk.should_pause_trading(),
        risk.get_pause_reason(),
    )

    return {
        "market": market,
        "trades": trades,
        "winrate": winrate,
        "pnl": day_pnl,
        "paused": risk.should_pause_trading(),
        "pause_reason": risk.get_pause_reason(),
    }


def main() -> None:
    res = run_backtest_for_market(MARKET)
    header = "===== 1D ML Backtest Summary ====="
    print()
    print(header)
    print(
        f" {res['market']:7s}  trades={res['trades']:3d}  "
        f"winrate={res['winrate']:.3f}  pnl={res['pnl']:.3f}  "
        f"paused={res['paused']}  reason={res['pause_reason']}"
    )


if __name__ == "__main__":
    main()
```

## File: scripts/simulate_ml_performance.py
```python
"""
ML Backtest Replay (다중 코인, 1분봉 기준)

- UpbitCollector로 7일 1분봉 수집
- TechnicalIndicators로 feature 생성
- CatBoost 모델로 분류 점수(confidence) 산출
- Kelly Sizer + RiskManager로 포지션 사이즈 / 손절 / 익절 / Kill Switch 적용
- 결과: 7일 PnL, 승률, 평균 R:R, 트레이드 수 출력
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

import logging
import numpy as np
import pandas as pd

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.upbit_collector import UpbitCollector
from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor
from strategy.kelly_sizing import get_position_size, MarketSizingConfig, CONF_THRESHOLD
from strategy.risk_manager import RiskManager, GlobalRiskConfig, TradeRiskConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ML_BACKTEST")

# 7-코인 유니버스
MARKETS: List[str] = [
    "KRW-BTC",  # v1: ML Time Machine / 실거래 대상 BTC 단일 코인
]

EQUITY_START = 1_000_000  # 100만 KRW 가정
BACKTEST_DAYS = 30         # 7일 구간


def get_market_config(market: str) -> Dict[str, Any]:
    """
    코인별 Kelly 상한 등 설정.
    - BTC, ETH: max_frac=0.04
    - XRP, SOL, AVAX: max_frac=0.03
    - DOGE, ADA: max_frac=0.02 (더 보수적으로)
    """
    if market in ("KRW-BTC", "KRW-ETH"):
        max_frac = 0.04
        is_small_cap = False
    elif market in ("KRW-XRP", "KRW-SOL", "KRW-AVAX"):
        max_frac = 0.03
        is_small_cap = False
    else:
        max_frac = 0.02
        is_small_cap = True

    mcfg = MarketSizingConfig(
        max_frac=max_frac,
        min_frac=0.01,
        is_small_cap=is_small_cap,
    )

    gcfg = GlobalRiskConfig(
        daily_loss_limit=-0.08,
        max_drawdown=-0.20,
        max_consec_losses=10,
    )

    # 튜닝 결과 반영: SL 0.4%, TP 1.0%
    tcfg = TradeRiskConfig(
        stop_loss_pct=0.004,
        take_profit_pct=0.010,
        trailing_pct=0.0,
    )

    return {
        "mcfg": mcfg,
        "gcfg": gcfg,
        "tcfg": tcfg,
    }


def run_backtest_for_market(market: str) -> Dict[str, Any]:
    logger.info(f"=== Backtest: {market} ({BACKTEST_DAYS}d 1m candles) ===")
    cfg = get_market_config(market)
    mcfg: MarketSizingConfig = cfg["mcfg"]
    gcfg: GlobalRiskConfig = cfg["gcfg"]
    tcfg: TradeRiskConfig = cfg["tcfg"]

    collector = UpbitCollector()
    df = collector.collect_historical_1m(market, days=BACKTEST_DAYS)

    if df is None or len(df) < 1000:
        logger.error(f"Not enough data for {market}: len={len(df) if df is not None else 0}")
        return {"market": market, "trades": 0, "winrate": 0.0, "day_pnl": 0.0}

    df = df.sort_index()
    logger.info(f"{market}: got {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    # Feature 생성
    ti = TechnicalIndicators(df)
    df = ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    # CatBoost 로드
    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feat_cols = cb.feature_names
    X = df_feat[feat_cols].values
    probs = cb.model.predict_proba(X)

    # 이진 분류(하락/상승) 가정: 상승 확률을 confidence로 사용
    if probs.shape[1] >= 2:
        conf_up = probs[:, -1]
    else:
        conf_up = probs[:, 0]

    closes = df_feat["close"].values
    index = df_feat.index

    risk = RiskManager(equity_start=EQUITY_START, global_cfg=gcfg, trade_cfg=tcfg)

    equity = EQUITY_START
    position = 0.0
    entry_price = None

    n_trades = 0
    wins = 0
    pnl_list: List[float] = []

    for i, ts in enumerate(index):
        price = float(closes[i])
        confidence = float(conf_up[i])

        # 1) 기존 포지션 관리 (close-to-close 기준 SL/TP)
        if position > 0.0 and entry_price is not None:
            ret = (price / entry_price) - 1.0
            hit_sl = ret <= -tcfg.stop_loss_pct
            hit_tp = ret >= tcfg.take_profit_pct

            if hit_sl or hit_tp:
                pnl_pct = ret
                equity *= (1.0 + pnl_pct)
                risk.register_trade(pnl_pct)
                n_trades += 1
                if pnl_pct > 0:
                    wins += 1
                pnl_list.append(pnl_pct)

                position = 0.0
                entry_price = None

                if risk.should_pause_trading():
                    logger.info(f"Kill Switch triggered for {market}: {risk.get_pause_reason()}")
                    break

        # Kill Switch 발동 시 신규 진입 금지
        if risk.should_pause_trading():
            continue

        # 2) 신규 진입 (롱 온리, 튜닝된 최소 confidence + Kelly)
        if position == 0.0:
            if confidence < CONF_THRESHOLD:
                continue

            size_krw = get_position_size(equity, confidence, cfg=mcfg)
            if size_krw <= 0:
                continue

            position = float(size_krw)
            entry_price = price

    total_pnl_pct = (equity / EQUITY_START) - 1.0
    winrate = (wins / n_trades) if n_trades > 0 else 0.0
    avg_pnl = float(np.mean(pnl_list)) if pnl_list else 0.0

    logger.info(
        f"{market}: trades={n_trades}, winrate={winrate:.3f}, "
        f"pnl={total_pnl_pct:.3f}, avg_trade_pnl={avg_pnl:.4f}, "
        f"paused={risk.should_pause_trading()}, reason={risk.get_pause_reason()}"
    )

    return {
        "market": market,
        "trades": n_trades,
        "winrate": winrate,
        "day_pnl": total_pnl_pct,
        "avg_trade_pnl": avg_pnl,
        "paused": risk.should_pause_trading(),
        "pause_reason": risk.get_pause_reason(),
    }


def main():
    results = []
    for m in MARKETS:
        res = run_backtest_for_market(m)
        results.append(res)

    df_res = pd.DataFrame(results)
    print(f"\n===== ML Backtest Summary ({BACKTEST_DAYS}d window) =====")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
```

## File: scripts/start_trading_daemon.sh
```bash
#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "=== [DAEMON] live_trader_ml.py 24/7 데몬 실행 ==="
echo "DRYRUN=False 설정 여부를 scripts/live_trader_ml.py 에서 반드시 확인하세요."
echo "로그는 logs/live_trader_daemon.log 에 기록됩니다."
echo

nohup python -m scripts.live_trader_ml >> "$LOGDIR/live_trader_daemon.log" 2>&1 &

echo "[DAEMON] live_trader_ml.py 백그라운드 시작됨. PID 목록은 다음 명령으로 확인:"
echo "         ps aux | rg live_trader_ml.py"
```

## File: scripts/summarize_relative_down.py
```python
"""
전략 B (BTC 급락장 상대강세 롱) 요약:
- trades_relative_down.csv 읽어서
- 전체 트레이드 수, 승률, 총 수익률, 평균 상대 초과수익, 일평균 매매 횟수 출력
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    path = ROOT / "trades_relative_down.csv"
    if not path.exists():
        print("trades_relative_down.csv not found")
        return

    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])

    n_trades = len(df)
    start = df["entry_time"].min()
    end = df["exit_time"].max()
    days = (end - start).days or 1
    trades_per_day = n_trades / days

    # 절대 수익률 기준 성과
    total_alt = float(np.prod(1.0 + df["r_alt_fwd"]) - 1.0)
    winrate = float((df["r_alt_fwd"] > 0).mean())
    avg_alt = float(df["r_alt_fwd"].mean())

    # BTC 대비 상대 초과수익
    avg_rel = float(df["rel_outperf"].mean())

    print("=== Strategy B: Relative outperformance on BTC down days ===")
    print(f"period_days      : {days}")
    print(f"trades           : {n_trades}")
    print(f"trades_per_day   : {trades_per_day:.3f}")
    print(f"winrate          : {winrate:.3f}")
    print(f"total_alt_pnl    : {total_alt:.3f}")
    print(f"avg_alt_ret      : {avg_alt:.4f}")
    print(f"avg_rel_outperf  : {avg_rel:.4f}")

if __name__ == "__main__":
    main()
```

## File: scripts/train_all_models.py
```python
"""
모든 모델 일괄 학습 스크립트
- KRW-BTC 7일 1분봉 수집
- Feature 생성
- CatBoost (분류) 학습
- LSTM (회귀) 학습
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from collectors.upbit_collector import UpbitCollector
from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor
from models.lstm_model import LSTMPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    market = "KRW-BTC"
    days = 7
    horizon = 60

    logger.info(f"Collecting {days} days of 1-minute data for {market}...")
    collector = UpbitCollector()
    df = collector.collect_historical_1m(market, days=days)

    logger.info(f"✅ Collected {len(df)} candles")

    if df is None or len(df) < 500:
        logger.error(f"❌ Not enough data collected: {len(df) if df is not None else 0}")
        sys.exit(1)

    # Feature 생성
    logger.info("Generating technical indicators...")
    ti = TechnicalIndicators(df)
    df = ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    logger.info(f"✅ Feature dataframe: shape={df_feat.shape}")

    # ─────────────────────────────
    # 1) CatBoost 모델 학습
    # ─────────────────────────────
    logger.info("Training CatBoost model...")
    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    X_cb, y_cb, feat_cb = cb.prepare_data(df_feat, target_horizon=horizon, threshold=0.001)
    train_acc, test_acc = cb.train(X_cb, y_cb, iterations=500)
    cb.save()
    logger.info(f"✅ CatBoost: Train acc={train_acc:.4f}, Test acc={test_acc:.4f}")

    # ─────────────────────────────
    # 2) LSTM 모델 학습
    # ─────────────────────────────
    logger.info("Training LSTM model...")
    lstm = LSTMPredictor(seq_len=60, horizon=horizon, model_path="models/lstm_model.pt")
    X_l, y_l = lstm.prepare_data(df_feat)
    lstm.train(X_l, y_l, epochs=10)
    logger.info("✅ LSTM training done.")

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("✅ All models trained and saved")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == "__main__":
    main()
```

## File: scripts/train_relative_down_model.py
```python
"""
전략 B: BTC 하락장에서 BTC 대비 아웃퍼폼할 알트를 고르는 상대강세 CatBoost 모델 학습 스크립트.

- 유니버스: KRW-ETH, KRW-DOGE, KRW-AVAX (이미 일봉 전략에서 엣지 확인된 3코인).[file:3]
- 타깃:
  - BTC 일봉 수익률 r_btc_today <= -0.03 (BTC가 하루 -3% 이상 빠진 날)만 사용.
  - H=2일 기준으로
      r_btc_fwd = close_btc[t+2] / close_btc[t] - 1
      r_alt_fwd = close_alt[t+2] / close_alt[t] - 1
      label = 1 if (r_alt_fwd - r_btc_fwd) > 0 else 0  (BTC보다 2일 동안 더 잘 오르면 1)
- 피처: 기존 TechnicalIndicators 일봉 피처(alt 기준).[file:2]
- 출력: models/catboost_rel_down.cbm
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyupbit
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor  # feature 이름 재사용용.[file:2]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - TRAIN_REL - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TRAIN_REL")

ALTS = ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]
BTC = "KRW-BTC"
DAYS = 730
H = 2
BTC_DROP_THRESH = -0.03  # BTC가 하루 -3% 이상 빠진 날만 사용.[web:60][web:37]


def fetch_daily(market: str) -> pd.DataFrame:
    df = pyupbit.get_ohlcv(market, interval="day", count=DAYS + 50)
    if df is None or len(df) < 200:
        raise RuntimeError(f"Not enough data for {market}")
    return df.sort_index()


def build_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    logger.info("Fetching BTC daily data for relative labels...")
    df_btc = fetch_daily(BTC)
    btc_close = df_btc["close"].astype(float)

    # BTC 하루 수익률 (오늘이 하락장인지 판별용)
    btc_ret_1d = btc_close.pct_change()

    X_list = []
    y_list = []

    for alt in ALTS:
        logger.info("Processing alt: %s", alt)
        df_alt = fetch_daily(alt)

        # 날짜 교집합 맞추기
        common_idx = df_btc.index.intersection(df_alt.index)
        df_b = df_btc.loc[common_idx].copy()
        df_a = df_alt.loc[common_idx].copy()

        # 피처 생성 (알트 기준)
        ti = TechnicalIndicators(df_a)
        ti.add_all_indicators()
        ti.add_price_features()
        df_feat = ti.get_feature_dataframe()

        # 피처/종가/레이블용 시계열 동기화
        idx = df_feat.index
        btc_c = df_b.loc[idx, "close"].astype(float)
        alt_c = df_a.loc[idx, "close"].astype(float)

        # BTC 하루 수익률도 같은 인덱스로 맞추기
        btc_ret_today = btc_ret_1d.loc[idx]

        for t in range(len(idx) - H):
            ts = idx[t]
            ts_fwd = idx[t + H]

            # 오늘이 BTC 하락장(-3% 이하) 아니면 스킵
            if pd.isna(btc_ret_today.iloc[t]) or btc_ret_today.iloc[t] > BTC_DROP_THRESH:
                continue

            c_btc_now = btc_c.iloc[t]
            c_btc_fwd = btc_c.loc[ts_fwd]
            c_alt_now = alt_c.iloc[t]
            c_alt_fwd = alt_c.loc[ts_fwd]

            r_btc_fwd = c_btc_fwd / c_btc_now - 1.0
            r_alt_fwd = c_alt_fwd / c_alt_now - 1.0
            rel = r_alt_fwd - r_btc_fwd

            label = 1 if rel > 0.0 else 0

            X_list.append(df_feat.iloc[t].values)
            y_list.append(label)

    if not X_list:
        raise RuntimeError("No training samples built. Check BTC_DROP_THRESH or data range.")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=int)
    logger.info("Built dataset: X.shape=%s, positive ratio=%.3f", X.shape, y.mean())
    return df_feat.columns.to_list(), X, y


def train_and_save():
    feature_names, X, y = build_dataset()

    # 시간 순서 보존한 80/20 split
    n = len(y)
    split = int(n * 0.8)
    X_train, X_valid = X[:split], X[split:]
    y_train, y_valid = y[:split], y[split:]

    train_pool = Pool(X_train, y_train, feature_names=feature_names)
    valid_pool = Pool(X_valid, y_valid, feature_names=feature_names)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        depth=6,
        learning_rate=0.05,
        iterations=500,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=50,
    )

    logger.info("Training CatBoost relative down-market model...")
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    out_path = ROOT / "models" / "catboost_rel_down.cbm"
    model.save_model(out_path)
    logger.info("Saved model to %s", out_path)


if __name__ == "__main__":
    train_and_save()
```
