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
- Only files matching these patterns are included: config/**/*
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
config/
  config.yaml
  live_config.yaml
```

# Files

## File: config/config.yaml
```yaml
trading:
  market: KRW-XRP # Primary active
  universe: ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL"] # Candidate pool
  interval: minute
  fee: 0.0005

# 포트폴리오 비중 설정
portfolio:
  weights:
    KRW-BTC: 0.35    # 35% 비중
    KRW-ETH: 0.25    # 25% 비중 
    KRW-XRP: 0.20    # 20% 비중
    KRW-SOL: 0.10    # 10% 비중
    CASH: 0.10       # 10% 현금
  rebalance_threshold: 0.05  # 5% 이상 벗어나면 리밸런싱
  kelly_fraction: 0.1        # 켈리 비율 10%

risk:
  max_drawdown: 0.05 # 5% Daily MDD Kill Switch
  max_position_size: 1000000 # 1M KRW Max per trade (Nano)
  min_position_size: 6000 # 6000 KRW Min (업비트 최소 주문 금액)
  stop_loss_pct: 0.005 # 0.5% base fixed SL (Dynamic TSL takes priority)

features:
  window_size: 100 # For rolling features
  lags: [1, 5, 10]

models:
  catboost_path: backend/models/catboost_sota.pkl
  mamba_path: backend/models/mamba_trend.pth
  scaler_path: backend/models/mamba_scaler.pkl
  features_path: backend/models/model_features.txt
```

## File: config/live_config.yaml
```yaml
trading:
  market: "KRW-BTC"
  confidence_threshold: 0.65 # Minimum entry confidence (Spec 2.3)
  trend_requirement: 0.02 # Required confidence trend
  momentum_threshold: 0.005 # Price momentum requirement
  volume_threshold: 1.1 # Volume increase requirement
  min_profit_target: 0.002 # Base take profit (0.2%)

risk:
  kelly_safety_factor: 0.5 # Half-Kelly implementation
  max_position_fraction: 0.20 # Hard cap per trade
  min_confidence_edge: 0.55 # Minimum for any position
  stop_loss_pct: 0.015 # Hard stop loss (1.5%)

execution:
  dry_run: true # FORCE DRY RUN
  portfolio_rebalance_threshold: 0.001
  position_timeout_minutes: 60 # Max hold time
  retry_attempts: 3

models:
  catboost_path: "backend/models/catboost_model.cbm"
  features_path: "backend/models/features.json"
  ensemble_weight_catboost: 0.7
  ensemble_weight_mamba: 0.3
```
