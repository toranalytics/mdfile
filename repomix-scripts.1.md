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
  00_integrity_check.py
  01_setup_database.py
  auto_daily.sh
  backtest_by_regime.backup.py
  backtest_by_regime.py
  backtest_relative_down_model.py
  backtest_volume_spike_strategy.py
  compare_strategy_vs_hodl.py
  download_multi_tf_history.py
  generate_signals_1d.py
  gridsearch_1d_signals.py
  label_regimes_from_ohlcv.py
  loop_auto_daily.sh
  multi_best_1d_trades.py
  order_utils.py
  param_sweep_ml_backtest.py
  param_sweep_range_bt.py
  param_sweep_range_multi.py
  portfolio_manager_example.py
  predict_latest.py
  run_realtime_trading.py
  runlive_minimal.py
  scan_volume_spikes_1d.py
  show_volume_spikes_today.py
  simulate_ml_performance_1d.py
  simulate_ml_performance.py
  summarize_relative_down.py
  train_all_models.py
  train_relative_down_model.py
```

# Files

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

## File: scripts/auto_daily.sh
```bash
#!/bin/zsh
cd /Users/junebeomseo/trading
source venv/bin/activate

python scripts/multi_best_1d_trades.py
python scripts/train_relative_down_model.py
python scripts/backtest_relative_down_model.py
python scripts/scan_volume_spikes_1d.py
python scripts/backtest_volume_spike_strategy.py || echo "no spike backtest yet"

python scripts/compare_strategy_vs_hodl.py
python scripts/summarize_relative_down.py
python scripts/show_volume_spikes_today.py

echo "----- head signals_1d_trades_KRW_ETH.csv -----"
head -n 10 signals_1d_trades_KRW_ETH.csv 2>/dev/null || echo "no ETH trades file"

echo "----- head trades_relative_down.csv -----"
head -n 10 trades_relative_down.csv 2>/dev/null || echo "no relative_down trades file"

echo "----- head volume_spike_all.csv -----"
head -n 10 volume_spike_all.csv 2>/dev/null || echo "no volume_spike_all"

echo "----- head volume_spike_candidates.csv -----"
head -n 10 volume_spike_candidates.csv 2>/dev/null || echo "no volume_spike_candidates"

echo "----- head backtest_volume_spike_trades.csv -----"
head -n 10 backtest_volume_spike_trades.csv 2>/dev/null || echo "no backtest_volume_spike_trades.csv"
```

## File: scripts/backtest_by_regime.backup.py
```python
"""
Backtest ML strategy performance by market regime (CRASH / BULL / RANGE).

- Uses:
  - 1m OHLCV: data/ohlcv/{MARKET}_minute1.parquet
  - Daily regimes: data/regimes/{MARKET}_day_regimes.parquet
- Runs the same 1m ML-based strategy separately on:
  - ALL days
  - CRASH-only days
  - BULL-only days
  - RANGE-only days
"""

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from collectors.upbit_collector import UpbitCollector  # not strictly needed if using local parquet only
from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor
from strategy.kelly_sizing import get_position_size, MarketSizingConfig, CONF_THRESHOLD
from strategy.risk_manager import RiskManager, GlobalRiskConfig, TradeRiskConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OHLCV_DIR = PROJECT_ROOT / "data" / "ohlcv"
REGIME_DIR = PROJECT_ROOT / "data" / "regimes"

MARKETS: List[str] = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-SOL",
    "KRW-DOGE",
    "KRW-ADA",
    "KRW-AVAX",
]

EQUITY_START = 1_000_000  # 100만 KRW 가정


def get_market_config(market: str) -> Dict[str, Any]:
    """
    Same logic as simulate_ml_performance.py:
    - BTC/ETH: max_frac=0.04
    - XRP/SOL/AVAX: max_frac=0.03
    - DOGE/ADA: max_frac=0.02 (small-cap style)
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

    tcfg = TradeRiskConfig(
        stop_loss_pct=0.004,   # 튜닝된 값: -0.4%
        take_profit_pct=0.010, # 튜닝된 값: +1.0%
        trailing_pct=0.0,
    )

    return {"mcfg": mcfg, "gcfg": gcfg, "tcfg": tcfg}


def load_1m_ohlcv(market: str) -> pd.DataFrame:
    path = OHLCV_DIR / f"{market.replace('-', '_')}_minute1.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    return df.sort_index()


def load_daily_regimes(market: str) -> pd.DataFrame:
    path = REGIME_DIR / f"{market.replace('-', '_')}_day_regimes.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    return df.sort_index()


def prepare_ml_inputs(df_1m: pd.DataFrame) -> Dict[str, Any]:
    """
    Apply TechnicalIndicators, then CatBoost to get confidence per 1m bar.
    """
    ti = TechnicalIndicators(df_1m)
    df = ti.add_all_indicators()
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

    return {"closes": closes, "conf_up": conf_up, "index": index}


def run_strategy_on_slice(
    closes: np.ndarray,
    conf_up: np.ndarray,
    index: pd.DatetimeIndex,
    date_mask: np.ndarray,  # bool mask on index (same length as index)
    mcfg: MarketSizingConfig,
    gcfg: GlobalRiskConfig,
    tcfg: TradeRiskConfig,
) -> Dict[str, Any]:
    """
    Run the 1m ML strategy only on bars where date_mask == True.
    """
    risk = RiskManager(equity_start=EQUITY_START, global_cfg=gcfg, trade_cfg=tcfg)

    equity = EQUITY_START
    position = 0.0
    entry_price = None

    n_trades = 0
    wins = 0
    pnl_list: List[float] = []

    for i, ts in enumerate(index):
        if not date_mask[i]:
            # 바깥 레짐 구간: 포지션이 열려있다면 강제 청산할지, 유지할지 선택
            # 여기서는 단순화를 위해: 구간 밖에서는 신규 진입만 금지, 기존 포지션은 일반 규칙으로만 처리.
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

        # 1) 기존 포지션 관리
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

        # 2) 신규 진입
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
        "trades": n_trades,
        "winrate": winrate,
        "pnl": total_pnl_pct,
        "avg_trade_pnl": avg_pnl,
        "paused": risk.should_pause_trading(),
        "pause_reason": risk.get_pause_reason(),
    }


def backtest_market_by_regime(market: str) -> pd.DataFrame:
    print(f"\n=== {market} ===")

    cfg = get_market_config(market)
    mcfg: MarketSizingConfig = cfg["mcfg"]
    gcfg: GlobalRiskConfig = cfg["gcfg"]
    tcfg: TradeRiskConfig = cfg["tcfg"]

    # 1m 데이터 & ML 입력 준비
    df_1m = load_1m_ohlcv(market)
    ml = prepare_ml_inputs(df_1m)
    closes = ml["closes"]
    conf_up = ml["conf_up"]
    index = ml["index"]

    # 일봉 레짐 불러와서 날짜별 regime 매핑
    df_reg = load_daily_regimes(market)
    # index.date -> regime 매핑 딕셔너리
    daily_reg = df_reg["regime"]
    reg_map = daily_reg.to_dict()  # key: Timestamp(09:00), value: CRASH/BULL/RANGE

    # 1m 인덱스의 날짜(YYYY-MM-DD)로 레짐을 붙인다
    dates = index.date
    regimes = []
    for ts in index:
        day_key = pd.Timestamp(ts.date()).replace(hour=9, minute=0, second=0, microsecond=0)
        regimes.append(reg_map.get(day_key, "UNKNOWN"))
    regimes = np.array(regimes, dtype=object)

    results = []

    # 1) 전체
    mask_all = regimes != "UNKNOWN"
    res_all = run_strategy_on_slice(closes, conf_up, index, mask_all, mcfg, gcfg, tcfg)
    res_all["market"] = market
    res_all["regime"] = "ALL"
    results.append(res_all)

    # 2) CRASH / BULL / RANGE
    for reg_name in ["CRASH", "BULL", "RANGE"]:
        mask = regimes == reg_name
        if not mask.any():
            results.append({
                "market": market,
                "regime": reg_name,
                "trades": 0,
                "winrate": 0.0,
                "pnl": 0.0,
                "avg_trade_pnl": 0.0,
                "paused": False,
                "pause_reason": None,
            })
            continue
        res = run_strategy_on_slice(closes, conf_up, index, mask, mcfg, gcfg, tcfg)
        res["market"] = market
        res["regime"] = reg_name
        results.append(res)

    return pd.DataFrame(results)


def main():
    all_results = []
    for m in MARKETS:
        df = backtest_market_by_regime(m)
        all_results.append(df)

    df_all = pd.concat(all_results, ignore_index=True)
    print("\n===== Regime-wise ML Backtest Summary =====")
    print(df_all.to_string(index=False))


if __name__ == "__main__":
    main()
```

## File: scripts/backtest_by_regime.py
```python
"""
Backtest ML strategy performance by market regime (CRASH / BULL / RANGE).

- Uses local parquet data:
  - 1m OHLCV:  data/ohlcv/{MARKET}_minute1.parquet
  - Daily regimes: data/regimes/{MARKET}_day_regimes.parquet
- Runs the same 1m ML-based strategy separately on:
  - ALL days      (regime != UNKNOWN)
  - CRASH-only
  - BULL-only
  - RANGE-only

Quant intent:
- Use same ML model / SL-TP / conf_threshold=0.70 everywhere.[file:1][file:3]
- Diagnose regime-dependent PnL & trade counts.
- Enforce "risk OFF in CRASH" policy candidate (no new entries during CRASH) at config level.
"""

from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from collectors.upbit_collector import UpbitCollector  # noqa: F401
from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor
from strategy.kelly_sizing import get_position_size, MarketSizingConfig, CONF_THRESHOLD
from strategy.risk_manager import RiskManager, GlobalRiskConfig, TradeRiskConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OHLCV_DIR = PROJECT_ROOT / "data" / "ohlcv"
REGIME_DIR = PROJECT_ROOT / "data" / "regimes"

MARKETS: List[str] = [
    "KRW-BTC",  # v1: ML 스캘퍼 실거래/백테스트 대상은 BTC 단일 코인
]

EQUITY_START = 1_000_000  # 100만 KRW


def get_market_config(market: str, regime: str | None = None) -> Dict[str, Any]:
    """
    Market-level risk config, with optional regime override.

    Base (from system codex / simulate_ml_performance):
    - BTC/ETH: max_frac = 0.04 (major)
    - XRP/SOL/AVAX: max_frac = 0.03
    - DOGE/ADA: max_frac = 0.02 (small-cap style)[file:1][file:3]

    Regime policy (10년차 퀀트 기준 제안):
    - CRASH: ML 스캘퍼 완전 risk-off → max_frac = 0.0 (엔트리 금지).
    - BULL/RANGE/ALL: 기본값 유지.
    """
    if market in ("KRW-BTC", "KRW-ETH"):
        base_max_frac = 0.04
        is_small_cap = False
    elif market in ("KRW-XRP", "KRW-SOL", "KRW-AVAX"):
        base_max_frac = 0.03
        is_small_cap = False
    else:
        base_max_frac = 0.02
        is_small_cap = True

    max_frac = base_max_frac
    # 레짐별 override
    if regime == "CRASH":
        # survival first: CRASH 레짐에서는 ML 스캘퍼 엔트리 금지
        max_frac = 0.0

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

    tcfg = TradeRiskConfig(
        stop_loss_pct=0.004,   # -0.4%
        take_profit_pct=0.010, # +1.0%
        trailing_pct=0.0,
    )

    return {"mcfg": mcfg, "gcfg": gcfg, "tcfg": tcfg}


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


def run_strategy_on_slice(
    closes: np.ndarray,
    conf_up: np.ndarray,
    index: pd.DatetimeIndex,
    date_mask: np.ndarray,
    mcfg: MarketSizingConfig,
    gcfg: GlobalRiskConfig,
    tcfg: TradeRiskConfig,
) -> Dict[str, Any]:
    """
    Run the 1m ML strategy only on bars where date_mask == True.

    - date_mask == False:
      * 신규 진입 금지
      * 기존 포지션은 SL/TP 기준으로만 정리 (현실적인 크로스-레짐 청산 허용)
    """
    risk = RiskManager(equity_start=EQUITY_START, global_cfg=gcfg, trade_cfg=tcfg)

    equity = EQUITY_START
    position = 0.0
    entry_price = None

    n_trades = 0
    wins = 0
    pnl_list: List[float] = []

    for i, ts in enumerate(index):
        if not date_mask[i]:
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


def backtest_market_by_regime(market: str) -> pd.DataFrame:
    print(f"\n=== {market} ===")

    # 1) 1m 데이터 & ML 입력
    df_1m = load_1m_ohlcv(market)
    ml = prepare_ml_inputs(df_1m)
    closes = ml["closes"]
    conf_up = ml["conf_up"]
    index = ml["index"]

    # 2) 일봉 레짐 → 날짜별 regime 매핑
    df_reg = load_daily_regimes(market)
    if "regime" not in df_reg.columns:
        raise ValueError(f"Regime column not found in daily regime file for {market}")

    df_reg = df_reg.copy()
    df_reg["date"] = df_reg.index.date
    reg_by_date = df_reg.groupby("date")["regime"].last()

    dates_series = pd.Series(index.date, index=index)
    regimes_series = dates_series.map(reg_by_date)
    regimes = regimes_series.fillna("UNKNOWN").astype(object).values

    # 디버그: 레짐별 바 개수 출력 (퀀트 시선에서 필수 체크)
    unique, counts = np.unique(regimes, return_counts=True)
    bar_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"Bar counts by regime: {bar_counts}")

    results: List[Dict[str, Any]] = []

    # 3) ALL (UNKNOWN 제외)
    cfg_all = get_market_config(market, regime=None)
    mask_all = regimes != "UNKNOWN"
    res_all = run_strategy_on_slice(
        closes, conf_up, index, mask_all,
        cfg_all["mcfg"], cfg_all["gcfg"], cfg_all["tcfg"],
    )
    res_all["market"] = market
    res_all["regime"] = "ALL"
    res_all["bars"] = int(mask_all.sum())
    results.append(res_all)

    # 4) CRASH / BULL / RANGE 개별
    for reg_name in ["CRASH", "BULL", "RANGE"]:
        mask = regimes == reg_name
        bars = int(mask.sum())

        cfg_reg = get_market_config(market, regime=reg_name)

        if bars == 0:
            results.append(
                {
                    "market": market,
                    "regime": reg_name,
                    "bars": 0,
                    "trades": 0,
                    "winrate": 0.0,
                    "pnl": 0.0,
                    "avg_trade_pnl": 0.0,
                    "paused": False,
                    "pause_reason": None,
                }
            )
            continue

        res = run_strategy_on_slice(
            closes, conf_up, index, mask,
            cfg_reg["mcfg"], cfg_reg["gcfg"], cfg_reg["tcfg"],
        )
        res["market"] = market
        res["regime"] = reg_name
        res["bars"] = bars
        results.append(res)

    return pd.DataFrame(results)


def main() -> None:
    all_results: List[pd.DataFrame] = []

    for market in MARKETS:
        try:
            df_res = backtest_market_by_regime(market)
            all_results.append(df_res)
        except FileNotFoundError as e:
            print(f"[WARN] Missing data for {market}: {e}")
        except Exception as e:
            print(f"[ERROR] Failed on {market}: {e}")

    if not all_results:
        print("No results to show.")
        return

    summary = pd.concat(all_results, ignore_index=True)

    print("\n===== Regime-wise ML Backtest Summary =====")
    cols = [
        "market",
        "regime",
        "bars",
        "trades",
        "winrate",
        "pnl",
        "avg_trade_pnl",
        "paused",
        "pause_reason",
    ]
    cols = [c for c in cols if c in summary.columns]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
```

## File: scripts/backtest_relative_down_model.py
```python
"""
전략 B 백테스트: BTC 하락장(-3% 이상) 날에, 상대강세 모델로 알트 롱 진입.

- 유니버스: KRW-ETH, KRW-DOGE, KRW-AVAX.
- 조건:
  - 오늘 BTC 수익률 r_btc_today <= -0.03 (하락장).[web:71]
  - 상대강세 모델 P(y=1 | features) >= 0.6 인 알트에만 진입.
- 진입/청산:
  - 진입: 오늘 알트 종가에 매수.
  - 청산: H=2일 뒤 알트 종가에 매도 (단순화).
- 성과:
  - 알트 절대 수익률 r_alt_fwd 기준 PnL.
  - 동시에 r_alt_fwd - r_btc_fwd 분포도 같이 본다.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyupbit
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.technical_indicators import TechnicalIndicators


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - BT_REL - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BT_REL")

ALTS = ["KRW-ETH", "KRW-DOGE", "KRW-AVAX"]
BTC = "KRW-BTC"
DAYS = 730
H = 2
BTC_DROP_THRESH = -0.03
CONF_THRESHOLD = 0.60


def fetch_daily(market: str) -> pd.DataFrame:
    df = pyupbit.get_ohlcv(market, interval="day", count=DAYS + 50)
    if df is None or len(df) < 200:
        raise RuntimeError(f"Not enough data for {market}")
    return df.sort_index()


def main():
    logger.info("Loading BTC daily data...")
    df_btc = fetch_daily(BTC)
    btc_close = df_btc["close"].astype(float)
    btc_ret_1d = btc_close.pct_change()

    logger.info("Loading relative down-market model...")
    model_path = ROOT / "models" / "catboost_rel_down.cbm"
    model = CatBoostClassifier()
    model.load_model(model_path)

    trades = []

    for alt in ALTS:
        logger.info("Backtesting alt: %s", alt)
        df_alt = fetch_daily(alt)

        common_idx = df_btc.index.intersection(df_alt.index)
        df_b = df_btc.loc[common_idx].copy()
        df_a = df_alt.loc[common_idx].copy()

        ti = TechnicalIndicators(df_a)
        ti.add_all_indicators()
        ti.add_price_features()
        df_feat = ti.get_feature_dataframe()

        idx = df_feat.index
        btc_c = df_b.loc[idx, "close"].astype(float)
        alt_c = df_a.loc[idx, "close"].astype(float)
        btc_ret_today = btc_ret_1d.loc[idx]

        X_all = df_feat.values.astype(np.float32)
        probs = model.predict_proba(X_all)
        if probs.shape[1] >= 2:
            conf_up = probs[:, -1]
        else:
            conf_up = probs[:, 0]

        for t in range(len(idx) - H):
            ts = idx[t]
            ts_fwd = idx[t + H]

            r_btc_today = btc_ret_today.iloc[t]
            if pd.isna(r_btc_today) or r_btc_today > BTC_DROP_THRESH:
                continue

            c_btc_now = btc_c.iloc[t]
            c_btc_fwd = btc_c.loc[ts_fwd]
            c_alt_now = alt_c.iloc[t]
            c_alt_fwd = alt_c.loc[ts_fwd]

            conf = float(conf_up[t])
            if conf < CONF_THRESHOLD:
                continue

            r_btc_fwd = c_btc_fwd / c_btc_now - 1.0
            r_alt_fwd = c_alt_fwd / c_alt_now - 1.0
            rel = r_alt_fwd - r_btc_fwd

            trades.append(
                {
                    "market": alt,
                    "entry_time": ts,
                    "exit_time": ts_fwd,
                    "entry_price": c_alt_now,
                    "exit_price": c_alt_fwd,
                    "btc_ret_today": r_btc_today,
                    "r_alt_fwd": r_alt_fwd,
                    "r_btc_fwd": r_btc_fwd,
                    "rel_outperf": rel,
                    "conf_rel": conf,
                }
            )

    if not trades:
        logger.info("No trades generated under current thresholds.")
        return

    df_trades = pd.DataFrame(trades)
    out_path = ROOT / "trades_relative_down.csv"
    df_trades.to_csv(out_path, index=False)
    logger.info("Saved %d trades to %s", len(df_trades), out_path)

    # 간단한 요약 출력
    total_alt = float(np.prod(1.0 + df_trades["r_alt_fwd"]) - 1.0)
    total_rel = float(df_trades["rel_outperf"].mean())
    winrate = float((df_trades["r_alt_fwd"] > 0).mean())
    n_trades = len(df_trades)

    logger.info(
        "RESULT: trades=%d winrate=%.3f total_alt_pnl=%.3f avg_rel_outperf=%.4f",
        n_trades,
        winrate,
        total_alt,
        total_rel,
    )
    print("trades:", n_trades)
    print("winrate:", round(winrate, 3))
    print("total_alt_pnl:", round(total_alt, 3))
    print("avg_rel_outperf:", round(total_rel, 4))


if __name__ == "__main__":
    main()
```

## File: scripts/backtest_volume_spike_strategy.py
```python
"""
이벤트 전략 C: 볼륨/가격 스파이크 후 2일 홀딩 전략 백테스트

전제:
- scan_volume_spikes_1d.py 가 과거 1년치에 대해 실행되어
  volume_spike_history.csv 를 만들어 놓았다고 가정하는 대신,
- 여기서 직접 각 코인 일봉 데이터를 받아서
  우리 스파이크 조건(20일/5일/절대 등락률)을 다시 적용해서 신호를 만들고,
- 신호 발생일(D) 종가 진입, D+2일 종가 청산으로 수익률 계산.[web:99][web:152]
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyupbit
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor

# 백테스트 파라미터
DAYS_BACK = 365

VOL_LOOKBACK_LONG = 20
VOL_LOOKBACK_SHORT = 5
MIN_VOL_RATIO_LONG = 3.0
MIN_VOL_RATIO_SHORT = 2.0
MIN_ABS_RET_1D = 0.10  # 너가 방금 설정한 값.[web:94]

MIN_PRICE_KRW = 50.0
CONF_ENTRY = 0.60  # 이 이상이면 진입 고려

def build_signals_for_market(market: str, cb: CatBoostPredictor):
    # 과거 DAYS_BACK + 버퍼 만큼 요청
    cnt = DAYS_BACK + VOL_LOOKBACK_LONG + 5
    try:
        df = pyupbit.get_ohlcv(market, interval="day", count=cnt)
    except Exception as e:
        print(f"[BT_VOL] fetch error {market}: {e}")
        return None
    if df is None or len(df) < VOL_LOOKBACK_LONG + 10:
        return None

    df = df.sort_index()
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    ret_1d = close.pct_change()
    vol_ma_long = vol.rolling(VOL_LOOKBACK_LONG).mean()
    vol_ma_short = vol.rolling(VOL_LOOKBACK_SHORT).mean()

    vol_ratio_long = vol / vol_ma_long
    vol_ratio_short = vol / vol_ma_short

    # 피처 + 모델 확률
    ti = TechnicalIndicators(df)
    ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    # latest DAYS_BACK 일만 사용
    cutoff = df.index.max() - timedelta(days=DAYS_BACK)
    mask_period = df.index >= cutoff

    rows = []
    for ts in df.index[mask_period]:
        if ts not in df_feat.index:
            continue
        if pd.isna(vol_ratio_long.loc[ts]) or pd.isna(vol_ratio_short.loc[ts]) or pd.isna(ret_1d.loc[ts]):
            continue

        c = float(close.loc[ts])
        if c < MIN_PRICE_KRW:
            continue

        r1d = float(ret_1d.loc[ts])
        vr_long = float(vol_ratio_long.loc[ts])
        vr_short = float(vol_ratio_short.loc[ts])

        is_long_spike = vr_long >= MIN_VOL_RATIO_LONG
        is_short_spike = (vr_short >= MIN_VOL_RATIO_SHORT) and (abs(r1d) >= MIN_ABS_RET_1D)

        if not (is_long_spike or is_short_spike):
            continue

        reason = []
        if is_long_spike:
            reason.append("LONG_VOL")
        if is_short_spike:
            reason.append("SHORT_VOL_RET")
        spike_reason = "+".join(reason)

        X = df_feat.loc[[ts], cb.feature_names].values.astype(np.float32)
        probs = cb.model.predict_proba(X)
        if probs.shape[1] >= 2:
            conf_up = float(probs[0, -1])
        else:
            conf_up = float(probs[0, 0])

        if conf_up < CONF_ENTRY:
            continue

        rows.append(
            {
                "market": market,
                "time": ts,
                "close": c,
                "ret_1d": r1d,
                "vol_ratio_long": vr_long,
                "vol_ratio_short": vr_short,
                "spike_reason": spike_reason,
                "conf_catboost": conf_up,
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows)


def backtest_spike_strategy():
    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()

    tickers = pyupbit.get_tickers(fiat="KRW")
    all_signals = []

    for m in tickers:
        sig = build_signals_for_market(m, cb)
        if sig is None:
            continue
        sig["market"] = m
        all_signals.append(sig)

    if not all_signals:
        print("No spike signals with conf >= %.2f in last %d days." % (CONF_ENTRY, DAYS_BACK))
        return

    sig_all = pd.concat(all_signals, ignore_index=True)
    sig_all = sig_all.sort_values(["time", "market"])
    sig_all.to_csv(ROOT / "volume_spike_signals_backtest.csv", index=False)

    # 각 신호별: D 종가 진입, D+2일 종가 청산 수익률 계산
    trades = []
    for _, row in sig_all.iterrows():
        mkt = row["market"]
        ts = pd.to_datetime(row["time"])
        entry_price = float(row["close"])

        # D+2일 종가 가져오기
        df2 = pyupbit.get_ohlcv(mkt, interval="day", count=5, to=ts + timedelta(days=5))
        if df2 is None or len(df2) == 0:
            continue
        df2 = df2.sort_index()
        # ts 기준으로 위치 찾고, 그로부터 +2번째 인덱스
        if ts not in df2.index:
            continue
        idx = df2.index.get_loc(ts)
        exit_idx = idx + 2
        if exit_idx >= len(df2):
            continue
        exit_ts = df2.index[exit_idx]
        exit_price = float(df2["close"].iloc[exit_idx])

        ret = exit_price / entry_price - 1.0

        trades.append(
            {
                "market": mkt,
                "entry_time": ts,
                "exit_time": exit_ts,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "ret": ret,
                "spike_reason": row["spike_reason"],
                "conf_catboost": row["conf_catboost"],
            }
        )

    if not trades:
        print("No trades generated for spike strategy (maybe CONF_ENTRY too high or conditions too strict).")
        return

    df_tr = pd.DataFrame(trades)
    df_tr = df_tr.sort_values("entry_time")
    df_tr.to_csv(ROOT / "trades_volume_spike_strategy.csv", index=False)

    # 성과 요약
    n_trades = len(df_tr)
    winrate = float((df_tr["ret"] > 0).mean())
    total_pnl = float(np.prod(1.0 + df_tr["ret"]) - 1.0)
    avg_ret = float(df_tr["ret"].mean())

    start = df_tr["entry_time"].min()
    end = df_tr["exit_time"].max()
    days = (end - start).days or 1
    trades_per_day = n_trades / days

    print("=== Strategy C: Volume/Price Spike 2-day Hold ===")
    print(f"period_days    : {days}")
    print(f"trades         : {n_trades}")
    print(f"trades_per_day : {trades_per_day:.3f}")
    print(f"winrate        : {winrate:.3f}")
    print(f"total_pnl      : {total_pnl:.3f}")
    print(f"avg_ret        : {avg_ret:.4f}")

    by_reason = df_tr.groupby("spike_reason")["ret"].agg(
        n="count",
        winrate=lambda x: float((x > 0).mean()),
        total=lambda x: float(np.prod(1.0 + x) - 1.0),
        avg=lambda x: float(x.mean()),
    ).reset_index()
    print("\n=== Breakdown by spike_reason ===")
    print(by_reason.to_string(index=False))


def main():
    backtest_spike_strategy()


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

## File: scripts/download_multi_tf_history.py
```python
"""
Download multi-timeframe OHLCV history from Upbit using pyupbit.

- Markets: 7코인 (BTC/ETH/XRP/SOL/DOGE/ADA/AVAX)
- Timeframes: 1분, 15분, 60분, 일봉
- Output: data/ohlcv/{MARKET}_{INTERVAL}.parquet
"""

import os
from pathlib import Path
from datetime import datetime

import pyupbit
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "ohlcv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKETS = [
    "KRW-BTC",
    "KRW-ETH",
    "KRW-XRP",
    "KRW-SOL",
    "KRW-DOGE",
    "KRW-ADA",
    "KRW-AVAX",
]

# Upbit / pyupbit에서 지원하는 interval 문자열[web:65][web:66]
INTERVALS = {
    "minute1": "1m",
    "minute15": "15m",
    "minute60": "1h",
    "day": "1d",
}

# 타임프레임별 최대 캔들 수 (필요시 이후 늘리면 됨)
MAX_COUNT = {
    "minute1": 20000,   # 약 13~14일치
    "minute15": 20000,  # 약 수개월
    "minute60": 20000,  # 2년 이상
    "day": 2000,        # 수년치
}


def download_ohlcv(market: str, interval: str, count: int) -> pd.DataFrame:
    """
    pyupbit.get_ohlcv를 사용해 OHLCV를 가져온다.
    interval 예: 'minute1', 'minute15', 'minute60', 'day'.[web:65][web:66]
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Downloading {market} {interval} (count={count})")

    df = pyupbit.get_ohlcv(
        ticker=market,
        interval=interval,
        count=count,
    )

    if df is None or df.empty:
        print(f"  -> WARNING: no data for {market} {interval}")
        return pd.DataFrame()

    # 인덱스는 datetime, 컬럼은 open/high/low/close/volume 등으로 들어옴[web:65][web:66]
    print(f"  -> got {len(df)} rows, from {df.index[0]} to {df.index[-1]}")
    return df


def main():
    for market in MARKETS:
        for interval, label in INTERVALS.items():
            count = MAX_COUNT[interval]
            df = download_ohlcv(market, interval, count)
            if df.empty:
                continue

            out_path = OUT_DIR / f"{market.replace('-', '_')}_{interval}.parquet"
            df.to_parquet(out_path)
            print(f"  -> saved to {out_path}")

    print("\n[download_multi_tf_history] DONE.")


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

## File: scripts/loop_auto_daily.sh
```bash
#!/bin/zsh
cd /Users/junebeomseo/trading
source venv/bin/activate

while true; do
  echo "==== AUTO DAILY RUN $(date) ===="
  ./scripts/auto_daily.sh >> logs/auto_daily_loop.log 2>&1
  echo "==== SLEEP 86400s ===="
  sleep 86400
done
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

## File: scripts/param_sweep_ml_backtest.py
```python
"""
BTC 전용 ML 백테스트 파라미터 스윕 스크립트.

- 입력 데이터/피처/ML 예측(confidence)은 한 번만 계산
- 아래 4개 파라미터 조합을 바꿔가면서 7일치 BTC를 반복 시뮬레이션:
  1) stop_loss_pct
  2) take_profit_pct
  3) conf_threshold (진입 최소 confidence)
  4) max_frac (Kelly 상한, 1트레이드당 계정 몇 %까지 쓰는지)

결과:
- 조합별 trades, winrate, day_pnl(7일 누적), avg_trade_pnl, paused, pause_reason 출력
- day_pnl 기준 내림차순으로 정렬해서 TOP N을 보여줌
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
from strategy.kelly_sizing import get_position_size, MarketSizingConfig
from strategy.risk_manager import RiskManager, GlobalRiskConfig, TradeRiskConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ML_PARAM_SWEEP")

MARKET = "KRW-BTC"
EQUITY_START = 1_000_000  # 100만 KRW

# 그리드 서치 범위 (필요하면 나중에 넓혀도 됨)
STOP_LIST = [0.004, 0.006]          # -0.4%, -0.6%
TP_LIST   = [0.010, 0.015, 0.020]   # +1.0%, +1.5%, +2.0%
CONF_LIST = [0.65, 0.70]            # 진입 최소 confidence
MAX_FRAC_LIST = [0.04, 0.08]       # 1트레이드당 계정 4%, 8% 상한


def prepare_data() -> Dict[str, Any]:
    """
    1) 7일치 1분봉 수집
    2) 피처 생성
    3) CatBoost로 상승 확률(conf_up) 계산
    """
    logger.info(f"=== Fetching data for {MARKET} (7d 1m candles) ===")
    collector = UpbitCollector()
    df = collector.collect_historical_1m(MARKET, days=7)

    if df is None or len(df) < 1000:
        raise RuntimeError(f"Not enough data for {MARKET}: len={len(df) if df is not None else 0}")

    df = df.sort_index()
    logger.info(f"{MARKET}: got {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    ti = TechnicalIndicators(df)
    df = ti.add_all_indicators()
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

    return {
        "closes": closes,
        "conf_up": conf_up,
        "index": index,
    }


def run_simulation(
    closes: np.ndarray,
    conf_up: np.ndarray,
    conf_threshold: float,
    max_frac: float,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> Dict[str, Any]:
    """
    하나의 파라미터 조합에 대해 7일치 BTC를 시뮬레이션.
    """
    # Kelly / Risk 설정
    mcfg = MarketSizingConfig(
        max_frac=max_frac,
        min_frac=0.02,
        is_small_cap=False,
    )
    gcfg = GlobalRiskConfig(
        daily_loss_limit=-0.08,
        max_drawdown=-0.20,
        max_consec_losses=10,
    )
    tcfg = TradeRiskConfig(
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        trailing_pct=0.0,   # 여기서는 trailing-stop 미사용
    )
    risk = RiskManager(equity_start=EQUITY_START, global_cfg=gcfg, trade_cfg=tcfg)

    equity = EQUITY_START
    position = 0.0
    entry_price = None

    n_trades = 0
    wins = 0
    pnl_list: List[float] = []

    for i in range(len(closes)):
        price = float(closes[i])
        confidence = float(conf_up[i])

        # 1) 기존 포지션 관리 (close-to-close 기준 SL/TP)
        if position > 0.0 and entry_price is not None:
            ret = (price / entry_price) - 1.0
            hit_sl = ret <= -stop_loss_pct
            hit_tp = ret >= take_profit_pct

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

        # Kill Switch 발동 시 신규 진입 금지
        if risk.should_pause_trading():
            continue

        # 2) 신규 진입 (롱 온리, confidence 필터 + Kelly)
        if position == 0.0:
            if confidence < conf_threshold:
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
        "market": MARKET,
        "trades": n_trades,
        "winrate": winrate,
        "day_pnl": total_pnl_pct,
        "avg_trade_pnl": avg_pnl,
        "paused": risk.should_pause_trading(),
        "pause_reason": risk.get_pause_reason(),
        "conf_threshold": conf_threshold,
        "max_frac": max_frac,
        "stop_loss_pct": stop_loss_pct,
        "take_profit_pct": take_profit_pct,
    }


def main():
    data = prepare_data()
    closes = data["closes"]
    conf_up = data["conf_up"]

    results: List[Dict[str, Any]] = []

    for sl in STOP_LIST:
        for tp in TP_LIST:
            for conf_th in CONF_LIST:
                for mf in MAX_FRAC_LIST:
                    logger.info(
                        f"=== TEST sl={sl:.4f}, tp={tp:.4f}, "
                        f"conf_th={conf_th:.2f}, max_frac={mf:.2f} ==="
                    )
                    res = run_simulation(
                        closes=closes,
                        conf_up=conf_up,
                        conf_threshold=conf_th,
                        max_frac=mf,
                        stop_loss_pct=sl,
                        take_profit_pct=tp,
                    )
                    logger.info(
                        f"Result: trades={res['trades']}, "
                        f"winrate={res['winrate']:.3f}, "
                        f"day_pnl={res['day_pnl']:.3f}, "
                        f"avg_trade_pnl={res['avg_trade_pnl']:.4f}, "
                        f"paused={res['paused']}, "
                        f"pause_reason={res['pause_reason']}"
                    )
                    results.append(res)

    df = pd.DataFrame(results)
    # day_pnl 기준 내림차순 정렬
    df_sorted = df.sort_values(by="day_pnl", ascending=False).reset_index(drop=True)

    print("\n===== BTC Param Sweep Results (7d window) =====")
    print(df_sorted.to_string(index=False))

    # 상위 10개만 따로 보여주기
    top_n = min(10, len(df_sorted))
    if top_n > 0:
        print(f"\n===== TOP {top_n} Combos by day_pnl =====")
        print(
            df_sorted.head(top_n)[
                [
                    "stop_loss_pct",
                    "take_profit_pct",
                    "conf_threshold",
                    "max_frac",
                    "trades",
                    "winrate",
                    "day_pnl",
                    "avg_trade_pnl",
                    "paused",
                    "pause_reason",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
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

## File: scripts/predict_latest.py
```python
"""
최근 데이터로 CatBoost / LSTM 예측 실행
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
from collectors.upbit_collector import UpbitCollector
from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor
from models.lstm_model import LSTMPredictor
import models.lstm_model_load_patch  # <= 이 줄이 패치 import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    market = "KRW-BTC"
    horizon = 60
    lookback_minutes = 60 * 24

    logger.info(f"Collecting latest {lookback_minutes} minutes for {market}...")
    collector = UpbitCollector()
    df = collector.collect_historical_1m(market, days=1)

    if df is None or len(df) < lookback_minutes:
        logger.error(f"Not enough recent data: {len(df) if df is not None else 0}")
        sys.exit(1)

    df = df.iloc[-lookback_minutes:]
    logger.info(f"Using recent window: {df.index[0]} -> {df.index[-1]} (len={len(df)})")

    ti = TechnicalIndicators(df)
    df = ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    logger.info(f"Feature df shape: {df_feat.shape}")

    latest_row = df_feat.iloc[[-1]]
    logger.info(f"Latest feature timestamp: {latest_row.index[0]}")

    logger.info("Loading CatBoost model...")
    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feature_cols = cb.feature_names
    X_latest = latest_row[feature_cols]
    proba = cb.model.predict_proba(X_latest)[0]
    pred_class = cb.model.predict(X_latest)[0]
    logger.info(f"CatBoost prediction class: {pred_class}")
    logger.info(f"Probabilities [down, flat, up] (예시): {proba}")

    logger.info("Loading LSTM model...")
    lstm = LSTMPredictor(seq_len=60, horizon=horizon, model_path="models/lstm_model.pt")
    lstm.load()

    pred_series = lstm.predict_future(df_feat)

    print("\n========== Prediction Result ==========")
    print(f"Base time: {latest_row.index[0]}")
    print(f"CatBoost class: {pred_class}")
    print(f"CatBoost proba: {proba}")
    print("\nLSTM next 60 minutes (head):")
    print(pred_series.head())
    print("... (tail):")
    print(pred_series.tail())
    print("=======================================")

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

## File: scripts/runlive_minimal.py
```python
"""
Minimal real-time loop for KRW-BTC using REST polling (no WebSocket, no actual orders).

- Every 60s:
  - Fetch latest 1m candles (last 200) via pyupbit.
  - Build features with TechnicalIndicators (same as backtest).
  - Run CatBoost model -> confidence.
  - Run Kelly sizing + RiskManager -> virtual position management.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyupbit

from features.technical_indicators import TechnicalIndicators
from models.catboost_model import CatBoostPredictor
from strategy.kelly_sizing import get_position_size, MarketSizingConfig, CONF_THRESHOLD
from strategy.risk_manager import RiskManager, GlobalRiskConfig, TradeRiskConfig

MARKET = "KRW-BTC"
EQUITY_START = 1_000_000
POLL_INTERVAL_SEC = 60
HISTORY_LEN = 200

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/minimal_paper.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("RUNLIVE_MINIMAL")


def fetch_recent_1m_ohlcv_sync(market: str, count: int) -> pd.DataFrame:
    df = pyupbit.get_ohlcv(market, interval="minute1", count=count)
    if df is None or len(df) < count // 2:
        raise RuntimeError(f"Not enough candles for {market}: got {0 if df is None else len(df)}")
    return df.sort_index()


def prepare_features(df_1m: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    ti = TechnicalIndicators(df_1m)
    ti.add_all_indicators()
    ti.add_price_features()
    df_feat = ti.get_feature_dataframe()

    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    feat_cols = cb.feature_names

    X = df_feat[feat_cols].values.astype(np.float32)
    index = df_feat.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Feature dataframe index is not DatetimeIndex")
    return X, index


def get_latest_confidence(X: np.ndarray) -> float:
    cb = CatBoostPredictor(model_path="models/catboost_model.cbm")
    cb.load()
    probs = cb.model.predict_proba(X)
    conf_up = probs[:, -1] if probs.shape[1] >= 2 else probs[:, 0]
    return float(conf_up[-1])


async def main() -> None:
    logger.info("🚀 Starting minimal live loop (paper only) for %s", MARKET)

    gcfg = GlobalRiskConfig(
        daily_loss_limit=-0.08,
        max_drawdown=-0.20,
        max_consec_losses=10,
    )
    tcfg = TradeRiskConfig(
        stop_loss_pct=0.006,
        take_profit_pct=0.010,
        trailing_pct=0.0,
    )
    mcfg = MarketSizingConfig(
        max_frac=0.04,
        min_frac=0.01,
        is_small_cap=False,
    )
    risk = RiskManager(equity_start=EQUITY_START, global_cfg=gcfg, trade_cfg=tcfg)

    equity = EQUITY_START
    position_krw = 0.0
    entry_price = None

    last_day = datetime.now(timezone.utc).date()

    loop = asyncio.get_event_loop()

    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.date() != last_day:
                risk.reset_daily()
                last_day = now.date()
                logger.info("🕛 New day -> daily risk reset")

            # 1) Get recent candles (blocking call in executor)
            df = await loop.run_in_executor(
                None, fetch_recent_1m_ohlcv_sync, MARKET, HISTORY_LEN
            )
            last_candle = df.iloc[-1]
            last_price = float(last_candle["close"])

            # 2) Features & confidence
            X, index = prepare_features(df)
            confidence = get_latest_confidence(X)

            logger.info(
                "Tick %s | price=%.0f | conf=%.3f | equity=%.0f | pos=%.0f",
                index[-1].isoformat(),
                last_price,
                confidence,
                equity,
                position_krw,
            )

            # 3) Manage existing position
            if position_krw > 0.0 and entry_price is not None:
                ret = (last_price / entry_price) - 1.0
                hit_sl = ret <= -tcfg.stop_loss_pct
                hit_tp = ret >= tcfg.take_profit_pct
                if hit_sl or hit_tp:
                    pnl_pct = ret
                    equity *= (1.0 + pnl_pct)
                    risk.register_trade(pnl_pct)
                    side = "SL" if hit_sl else "TP"
                    logger.info("EXIT %s | pnl_pct=%.4f | new_equity=%.0f", side, pnl_pct, equity)
                    position_krw = 0.0
                    entry_price = None

            if risk.should_pause_trading():
                logger.warning("⚠️ Trading paused: %s", risk.get_pause_reason())
                await asyncio.sleep(POLL_INTERVAL_SEC)
                continue

            # 4) Entry
            if position_krw == 0.0:
                if confidence >= CONF_THRESHOLD:
                    size_krw = get_position_size(equity, confidence, cfg=mcfg)
                    if size_krw > 0:
                        position_krw = float(size_krw)
                        entry_price = last_price
                        logger.info(
                            "ENTRY BUY | size_krw=%.0f | entry_price=%.0f | conf=%.3f",
                            position_krw,
                            entry_price,
                            confidence,
                        )

        except Exception as e:
            logger.exception("Loop error: %s", e)

        await asyncio.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
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


MARKET = "KRW-BTC"
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
