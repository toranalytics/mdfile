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
- Only files matching these patterns are included: features/**/*, execution/**/*, risk/**/*, strategy/**/*, collectors/**/*, models/**/*, backtest/**/*
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
backtest/
  __init__.py
collectors/
  __init__.py
  upbit_collector.py
  websocket_collector.py
execution/
  __init__.py
features/
  __init__.py
  regime_detection.py
  technical_indicators.py
models/
  __init__.py
  catboost_model.cbm
  catboost_model.meta
  catboost_model.py
  catboost_rel_down.cbm
  ensemble_predictor.py
  lstm_model_load_patch.py
  lstm_model.pt
  lstm_model.py
  transformer_1d.pt
  transformer_model.py
risk/
  __init__.py
strategy/
  __init__.py
  kelly_sizing.backup.py
  kelly_sizing.py
  risk_manager.py
```

# Files

## File: backtest/__init__.py
```python

```

## File: collectors/__init__.py
```python

```

## File: collectors/upbit_collector.py
```python
"""
Upbit REST API Collector
- 과거 데이터 수집 (분봉, 일봉)
- 현재가 조회
"""
import pyupbit
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

class UpbitCollector:
    def __init__(self):
        self.markets = self._get_krw_markets()
        logger.info(f"Initialized with {len(self.markets)} KRW markets")
    
    def _get_krw_markets(self) -> List[str]:
        """KRW 마켓 목록 조회"""
        try:
            tickers = pyupbit.get_tickers(fiat="KRW")
            return [t for t in tickers if t.startswith("KRW-")]
        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            return ["KRW-BTC", "KRW-ETH"]  # Fallback
    
    def get_ohlcv(
        self,
        market: str,
        interval: str = "minute1",
        count: int = 200
    ) -> Optional[pd.DataFrame]:
        """단일 호출 OHLCV"""
        try:
            df = pyupbit.get_ohlcv(market, interval=interval, count=count)
            if df is None or df.empty:
                logger.warning(f"No data for {market}")
                return None
            return df
        except Exception as e:
            logger.error(f"Failed to get OHLCV for {market}: {e}")
            return None

    def get_current_price(self, market: str) -> Optional[float]:
        """현재가 조회"""
        try:
            price = pyupbit.get_current_price(market)
            return float(price) if price else None
        except Exception as e:
            logger.error(f"Failed to get current price for {market}: {e}")
            return None
    
    def get_orderbook(self, market: str) -> Optional[dict]:
        """호가 정보 조회"""
        try:
            orderbook = pyupbit.get_orderbook(market)
            return orderbook[0] if orderbook else None
        except Exception as e:
            logger.error(f"Failed to get orderbook for {market}: {e}")
            return None
    
    def collect_historical_1m(
        self,
        market: str,
        days: int = 7
    ) -> pd.DataFrame:
        """
        과거 1분봉 데이터 수집
        - pyupbit.get_ohlcv에 count만 크게 주면 내부에서 200개씩 나눠 호출함[web:69][web:77]
        """
        target_count = days * 24 * 60  # 1분봉 개수

        try:
            df = pyupbit.get_ohlcv(
                market,
                interval="minute1",
                count=target_count
            )
            if df is None or df.empty:
                logger.error(f"No data returned for {market}")
                return pd.DataFrame()

            # 안전용 정렬 + 중복 제거[web:63]
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="first")]

            logger.info(
                f"✅ Final: {len(df)} candles "
                f"({df.index[0]} to {df.index[-1]})"
            )
            return df

        except Exception as e:
            logger.error(f"Error collecting historical data: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    collector = UpbitCollector()
    print("\nCollecting 1 day historical data...")
    historical = collector.collect_historical_1m("KRW-BTC", days=1)
    print(f"\n✅ Collected {len(historical)} candles")
    if len(historical) > 0:
        print(f"Range: {historical.index[0]} to {historical.index[-1]}")
```

## File: collectors/websocket_collector.py
```python
"""
Upbit WebSocket Collector
- 실시간 틱 데이터 수집
- 실시간 호가 데이터 수집
"""
import websockets
import json
import asyncio
from datetime import datetime
import logging
from typing import Callable, List
import uuid

logger = logging.getLogger(__name__)

class UpbitWebSocketCollector:
    def __init__(self, markets: List[str]):
        self.markets = markets
        self.ws_url = "wss://api.upbit.com/websocket/v1"
        self.is_running = False
        
    async def subscribe_trade(self, callback: Callable):
        """
        실시간 체결 데이터 구독
        
        Args:
            callback: 데이터 수신 시 호출할 함수
                     callback(data: dict) 형태
        """
        subscribe_data = [
            {"ticket": str(uuid.uuid4())},
            {
                "type": "trade",
                "codes": self.markets,
                "isOnlyRealtime": True
            }
        ]
        
        try:
            async with websockets.connect(self.ws_url, ping_interval=60) as ws:
                await ws.send(json.dumps(subscribe_data))
                logger.info(f"Subscribed to trade for {len(self.markets)} markets")
                
                self.is_running = True
                
                while self.is_running:
                    try:
                        data = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        
                        # Parse binary data
                        parsed = json.loads(data.decode("utf-8"))
                        
                        # 콜백 호출
                        await callback(parsed)
                        
                    except asyncio.TimeoutError:
                        logger.warning("WebSocket recv timeout")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_running = False
    
    async def subscribe_orderbook(self, callback: Callable):
        """
        실시간 호가 데이터 구독
        
        Args:
            callback: 데이터 수신 시 호출할 함수
        """
        subscribe_data = [
            {"ticket": str(uuid.uuid4())},
            {
                "type": "orderbook",
                "codes": self.markets,
                "isOnlyRealtime": True
            }
        ]
        
        try:
            async with websockets.connect(self.ws_url, ping_interval=60) as ws:
                await ws.send(json.dumps(subscribe_data))
                logger.info(f"Subscribed to orderbook for {len(self.markets)} markets")
                
                self.is_running = True
                
                while self.is_running:
                    try:
                        data = await asyncio.wait_for(ws.recv(), timeout=30.0)
                        parsed = json.loads(data.decode("utf-8"))
                        await callback(parsed)
                        
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing orderbook: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_running = False
    
    def stop(self):
        """WebSocket 연결 종료"""
        self.is_running = False
        logger.info("WebSocket collector stopped")

# Test callback function
async def test_trade_callback(data: dict):
    """체결 데이터 출력"""
    print(f"[{data['code']}] {data['trade_price']:>12,.0f} KRW | Vol: {data['trade_volume']:>10.6f}")

async def test_orderbook_callback(data: dict):
    """호가 데이터 출력"""
    print(f"[{data['code']}] Bid: {data['orderbook_units'][0]['bid_price']:>12,.0f} | Ask: {data['orderbook_units'][0]['ask_price']:>12,.0f}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Test: 실시간 체결 데이터
    collector = UpbitWebSocketCollector(["KRW-BTC", "KRW-ETH"])
    
    print("Starting WebSocket test (Ctrl+C to stop)...")
    print("Listening for BTC and ETH trades...")
    
    try:
        asyncio.run(collector.subscribe_trade(test_trade_callback))
    except KeyboardInterrupt:
        print("\nStopped")
        collector.stop()
```

## File: execution/__init__.py
```python

```

## File: features/__init__.py
```python

```

## File: features/regime_detection.py
```python
"""
시장 레짐 탐지
- Trending vs Ranging
- Volatility regime
- Volume regime
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """시장 레짐 탐지"""
    
    def __init__(self, n_regimes: int = 3):
        """
        Args:
            n_regimes: 레짐 개수 (기본 3: Low/Medium/High volatility)
        """
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    
    def detect_trend_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        추세 레짐 탐지
        
        Returns:
            Series: 'trending', 'ranging'
        """
        # ADX-like calculation
        close = df['close']
        
        # Directional movement
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - close.shift())
        tr3 = abs(df['low'] - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed indicators
        window = 14
        atr = tr.rolling(window).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window).mean() / atr
        
        # DX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX
        adx = dx.rolling(window).mean()
        
        # Regime classification
        regime = pd.Series('ranging', index=df.index)
        regime[adx > 25] = 'trending'
        
        return regime
    
    def detect_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        변동성 레짐 탐지 (K-Means clustering)
        
        Returns:
            Series: 0 (low), 1 (medium), 2 (high)
        """
        # Calculate volatility features
        close = df['close']
        returns = close.pct_change()
        
        features = pd.DataFrame({
            'volatility': returns.rolling(20).std(),
            'atr_ratio': df['atr_14'] / close if 'atr_14' in df.columns else returns.rolling(14).std(),
            'hl_ratio': (df['high'] - df['low']) / close
        })
        
        # Remove NaN
        features = features.dropna()
        
        # Standardize
        features_scaled = self.scaler.fit_transform(features)
        
        # Cluster
        labels = self.kmeans.fit_predict(features_scaled)
        
        # Map to original index
        regime = pd.Series(np.nan, index=df.index)
        regime.loc[features.index] = labels
        
        # Forward fill
        regime = regime.fillna(method='ffill')
        
        return regime.astype(int)
    
    def detect_volume_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        거래량 레짐 탐지
        
        Returns:
            Series: 'low', 'normal', 'high'
        """
        volume = df['volume']
        volume_sma = volume.rolling(20).mean()
        volume_std = volume.rolling(20).std()
        
        z_score = (volume - volume_sma) / volume_std
        
        regime = pd.Series('normal', index=df.index)
        regime[z_score < -1] = 'low'
        regime[z_score > 1] = 'high'
        
        return regime
    
    def add_all_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 레짐 추가"""
        df = df.copy()
        
        df['trend_regime'] = self.detect_trend_regime(df)
        df['volatility_regime'] = self.detect_volatility_regime(df)
        df['volume_regime'] = self.detect_volume_regime(df)
        
        logger.info("Added all regime indicators")
        
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from collectors.upbit_collector import UpbitCollector
    from features.technical_indicators import TechnicalIndicators
    
    # Collect data
    collector = UpbitCollector()
    df = collector.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    
    if df is not None:
        # Add technical indicators first
        ti = TechnicalIndicators(df)
        df = ti.add_all_indicators()
        
        # Add regimes
        detector = RegimeDetector()
        df = detector.add_all_regimes(df)
        
        print("Data with regimes:")
        print(df[['close', 'trend_regime', 'volatility_regime', 'volume_regime']].tail(20))
        
        # Statistics
        print(f"\nTrend regime distribution:")
        print(df['trend_regime'].value_counts())
        
        print(f"\nVolatility regime distribution:")
        print(df['volatility_regime'].value_counts())
        
        print(f"\nVolume regime distribution:")
        print(df['volume_regime'].value_counts())
```

## File: features/technical_indicators.py
```python
"""
기술적 지표 계산
- RSI, MACD, Bollinger Bands
- Volume indicators
- Momentum indicators
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """기술적 지표 계산 클래스"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: OHLCV 데이터프레임 (columns: open, high, low, close, volume)
        """
        self.df = df.copy()
        
    def add_all_indicators(self) -> pd.DataFrame:
        """모든 지표 추가"""
        self.add_trend_indicators()
        self.add_momentum_indicators()
        self.add_volatility_indicators()
        self.add_volume_indicators()
        return self.df
    
    def add_trend_indicators(self):
        """추세 지표"""
        close = self.df['close']
        
        # Moving Averages
        self.df['sma_5'] = SMAIndicator(close, window=5).sma_indicator()
        self.df['sma_10'] = SMAIndicator(close, window=10).sma_indicator()
        self.df['sma_20'] = SMAIndicator(close, window=20).sma_indicator()
        self.df['sma_60'] = SMAIndicator(close, window=60).sma_indicator()
        
        self.df['ema_5'] = EMAIndicator(close, window=5).ema_indicator()
        self.df['ema_10'] = EMAIndicator(close, window=10).ema_indicator()
        self.df['ema_20'] = EMAIndicator(close, window=20).ema_indicator()
        self.df['ema_60'] = EMAIndicator(close, window=60).ema_indicator()
        
        # MACD
        macd = MACD(close)
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()
        
        logger.info("Added trend indicators")
    
    def add_momentum_indicators(self):
        """모멘텀 지표"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        
        # RSI
        self.df['rsi_14'] = RSIIndicator(close, window=14).rsi()
        self.df['rsi_7'] = RSIIndicator(close, window=7).rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high, low, close)
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()
        
        # Rate of Change
        self.df['roc_10'] = close.pct_change(periods=10) * 100
        self.df['roc_20'] = close.pct_change(periods=20) * 100
        
        logger.info("Added momentum indicators")
    
    def add_volatility_indicators(self):
        """변동성 지표"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        
        # Bollinger Bands
        bb = BollingerBands(close)
        self.df['bb_upper'] = bb.bollinger_hband()
        self.df['bb_middle'] = bb.bollinger_mavg()
        self.df['bb_lower'] = bb.bollinger_lband()
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
        self.df['bb_position'] = (close - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        # ATR
        self.df['atr_14'] = AverageTrueRange(high, low, close, window=14).average_true_range()
        
        # Historical Volatility
        returns = close.pct_change()
        self.df['volatility_10'] = returns.rolling(window=10).std() * np.sqrt(10)
        self.df['volatility_20'] = returns.rolling(window=20).std() * np.sqrt(20)
        
        logger.info("Added volatility indicators")
    
    def add_volume_indicators(self):
        """거래량 지표"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        volume = self.df['volume']
        
        # Volume Moving Averages
        self.df['volume_sma_10'] = volume.rolling(window=10).mean()
        self.df['volume_sma_20'] = volume.rolling(window=20).mean()
        self.df['volume_ratio'] = volume / self.df['volume_sma_20']
        
        # On Balance Volume
        self.df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(high, low, close, volume)
        self.df['vwap'] = vwap.volume_weighted_average_price()
        
        logger.info("Added volume indicators")
    
    def add_price_features(self):
        """가격 관련 파생 변수"""
        close = self.df['close']
        high = self.df['high']
        low = self.df['low']
        open_ = self.df['open']
        
        # Returns
        self.df['return_1'] = close.pct_change(1)
        self.df['return_5'] = close.pct_change(5)
        self.df['return_10'] = close.pct_change(10)
        
        # High-Low range
        self.df['hl_ratio'] = (high - low) / close
        
        # Close position in range
        self.df['close_position'] = (close - low) / (high - low)
        
        # Body size (candle)
        self.df['body_size'] = abs(close - open_) / close
        
        logger.info("Added price features")
    
    def get_feature_dataframe(self) -> pd.DataFrame:
        """최종 feature 데이터프레임 반환"""
        # NaN 제거 (초기 지표 계산 불가 구간)
        return self.df.dropna()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    from collectors.upbit_collector import UpbitCollector
    
    collector = UpbitCollector()
    df = collector.get_ohlcv("KRW-BTC", interval="minute1", count=200)
    
    if df is not None:
        print("Original data:")
        print(df.head())
        print(f"\nShape: {df.shape}")
        
        # Add indicators
        ti = TechnicalIndicators(df)
        df_features = ti.add_all_indicators()
        ti.add_price_features()
        df_features = ti.get_feature_dataframe()
        
        print(f"\nWith features:")
        print(df_features.head())
        print(f"\nShape: {df_features.shape}")
        print(f"\nFeature columns ({len(df_features.columns)}):")
        print(df_features.columns.tolist())
        
        # Statistics
        print(f"\nRSI range: {df_features['rsi_14'].min():.2f} - {df_features['rsi_14'].max():.2f}")
        print(f"MACD range: {df_features['macd'].min():.2f} - {df_features['macd'].max():.2f}")
        print(f"BB Width mean: {df_features['bb_width'].mean():.4f}")
```

## File: models/__init__.py
```python

```

## File: models/catboost_model.py
```python
"""
CatBoost 가격 예측 모델
- 60분 후 가격 방향 예측 (상승/하락)
- Feature importance 분석
"""
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class CatBoostPredictor:
    """CatBoost 기반 가격 방향 예측"""
    
    def __init__(self, model_path: str = 'models/catboost_model.cbm'):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = None
        self.feature_importance = None
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_horizon: int = 60,
        threshold: float = 0.001
    ) -> tuple:
        """
        데이터 준비
        
        Args:
            df: Feature 데이터프레임
            target_horizon: 예측 시점 (분)
            threshold: 상승/하락 임계값 (0.1% = 0.001)
        
        Returns:
            (X, y, feature_names)
        """
        # Target: 60분 후 가격 변화율
        df = df.copy()
        df['future_return'] = df['close'].pct_change(target_horizon).shift(-target_horizon)
        
        # Binary classification
        df['target'] = 0  # 하락/중립
        df.loc[df['future_return'] > threshold, 'target'] = 1  # 상승
        
        # Remove NaN
        df = df.dropna()
        
        # Features (exclude target and price columns)
        exclude_cols = ['target', 'future_return', 'open', 'high', 'low', 'close', 'volume', 'value']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['target']
        
        self.feature_names = feature_cols
        
        logger.info(f"Data prepared: {len(X)} samples, {len(feature_cols)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        iterations: int = 1000,
        early_stopping_rounds: int = 50
    ):
        """
        모델 학습
        
        Args:
            X: Features
            y: Target (0: 하락, 1: 상승)
            test_size: 테스트 세트 비율
            iterations: 학습 반복 횟수
            early_stopping_rounds: 조기 종료 라운드
        """
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Create CatBoost datasets
        train_pool = Pool(X_train, y_train)
        test_pool = Pool(X_test, y_test)
        
        # Initialize model
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=0.05,
            depth=6,
            loss_function='Logloss',
            eval_metric='Accuracy',
            random_seed=42,
            early_stopping_rounds=early_stopping_rounds,
            verbose=100
        )
        
        # Train
        logger.info("Training CatBoost model...")
        self.model.fit(
            train_pool,
            eval_set=test_pool,
            plot=False
        )
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        logger.info(f"Train Accuracy: {train_acc:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        
        logger.info("\nTest Set Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred_test)}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Important Features:")
        logger.info(f"\n{self.feature_importance.head(10)}")
        
        return train_acc, test_acc
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        예측
        
        Args:
            X: Feature 데이터프레임
        
        Returns:
            예측 확률 [P(하락), P(상승)]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict_proba(X)
    
    def predict_signal(self, X: pd.DataFrame, threshold: float = 0.6) -> int:
        """
        트레이딩 시그널 생성
        
        Args:
            X: Feature 데이터프레임 (single row)
            threshold: 상승 확률 임계값
        
        Returns:
            1 (BUY), 0 (HOLD), -1 (SELL)
        """
        proba = self.predict(X)[0]
        
        if proba[1] > threshold:
            return 1  # BUY
        elif proba[0] > threshold:
            return -1  # SELL
        else:
            return 0  # HOLD
    
    def save(self):
        """모델 저장"""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(self.model_path))
        
        # Feature names 저장
        metadata = {
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance.to_dict()
        }
        joblib.dump(metadata, self.model_path.with_suffix('.meta'))
        
        logger.info(f"Model saved to {self.model_path}")
    
    def load(self):
        """모델 로드"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = CatBoostClassifier()
        self.model.load_model(str(self.model_path))
        
        # Metadata 로드
        metadata = joblib.load(self.model_path.with_suffix('.meta'))
        self.feature_names = metadata['feature_names']
        self.feature_importance = pd.DataFrame(metadata['feature_importance'])
        
        logger.info(f"Model loaded from {self.model_path}")

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    logging.basicConfig(level=logging.INFO)
    
    from collectors.upbit_collector import UpbitCollector
    from features.technical_indicators import TechnicalIndicators
    
    # Collect data (7 days)
    collector = UpbitCollector()
    print("Collecting 7 days of 1-minute data...")
    df = collector.collect_historical_1m("KRW-BTC", days=7)
    
    if len(df) > 0:
        print(f"Collected {len(df)} candles")
        
        # Add features
        ti = TechnicalIndicators(df)
        df = ti.add_all_indicators()
        ti.add_price_features()
        df = ti.get_feature_dataframe()
        
        print(f"Features added: {df.shape}")
        
        # Train model
        predictor = CatBoostPredictor()
        X, y, feature_names = predictor.prepare_data(df, target_horizon=60, threshold=0.001)
        
        train_acc, test_acc = predictor.train(X, y, iterations=500)
        
        # Save model
        predictor.save()
        
        # Test prediction
        print("\n=== Test Prediction ===")
        latest_features = X.iloc[[-1]]
        proba = predictor.predict(latest_features)
        signal = predictor.predict_signal(latest_features, threshold=0.6)
        
        print(f"Latest prediction:")
        print(f"  P(하락): {proba[0][0]:.4f}")
        print(f"  P(상승): {proba[0][1]:.4f}")
        print(f"  Signal: {['SELL', 'HOLD', 'BUY'][signal + 1]}")
```

## File: models/ensemble_predictor.py
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from models.catboost_model import CatBoostPredictor
from models.lstm_model import LSTMPredictor
from features.technical_indicators import TechnicalIndicators
from collectors.upbit_collector import UpbitCollector

class EnsemblePredictor:
    """
    CatBoost (분류) + LSTM (수익률 회귀) 앙상블
    - CatBoost: 상승 확률 P_up
    - LSTM: 예상 수익률 r_hat
    - 최종 점수: score = w1 * P_up + w2 * sigmoid(k * r_hat)
    """
    def __init__(self,
                 catboost_model_path: str = "models/catboost_model.cbm",
                 lstm_model_path: str = "models/lstm_model.pt",
                 w_prob: float = 0.6,
                 w_ret: float = 0.4,
                 k_ret: float = 50.0):
        self.cb = CatBoostPredictor(model_path=catboost_model_path)
        self.lstm = LSTMPredictor(model_path=lstm_model_path)
        self.w_prob = w_prob
        self.w_ret = w_ret
        self.k_ret = k_ret

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-x))

    def prepare_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        df_raw: OHLCV 최소 200개 이상
        """
        ti = TechnicalIndicators(df_raw)
        df_feat = ti.add_all_indicators()
        ti.add_price_features()
        df_feat = ti.get_feature_dataframe()
        return df_feat

    def predict(self, df_feat: pd.DataFrame) -> dict:
        """
        df_feat: feature 포함된 전체 시계열
        returns:
            {
                "p_up": float,
                "future_return": float,
                "ensemble_score": float,
                "signal": int (-1,0,1)
            }
        """
        # CatBoost part
        X_cb, y_cb, feat_cb = self.cb.prepare_data(df_feat)
        self.cb.load()
        latest_cb = X_cb.iloc[[-1]]
        proba = self.cb.predict(latest_cb)[0]
        p_up = float(proba[1])

        # LSTM part
        self.lstm.load()
        latest_seq = df_feat.iloc[-self.lstm.seq_len:]
        fut_ret = float(self.lstm.predict_future_return(latest_seq))

        # Ensemble
        score_ret = self._sigmoid(self.k_ret * fut_ret)
        ensemble_score = self.w_prob * p_up + self.w_ret * score_ret

        # Signal
        if ensemble_score > 0.6:
            signal = 1   # BUY
        elif ensemble_score < 0.4:
            signal = -1  # SELL
        else:
            signal = 0   # HOLD

        return {
            "p_up": p_up,
            "future_return": fut_ret,
            "ensemble_score": ensemble_score,
            "signal": signal,
        }

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    collector = UpbitCollector()
    print("Collecting 2 days of 1-minute data for ensemble test...")
    df = collector.collect_historical_1m("KRW-BTC", days=2)

    if len(df) < 500:
        print(f"Not enough data: {len(df)}")
        sys.exit(0)

    ens = EnsemblePredictor()
    df_feat = ens.prepare_features(df)
    result = ens.predict(df_feat)

    print("\n=== Ensemble Prediction ===")
    print(f"P(Up):          {result['p_up']:.4f}")
    print(f"Future return:  {result['future_return']:.4%}")
    print(f"Ensemble score: {result['ensemble_score']:.4f}")
    print(f"Signal:         {['SELL','HOLD','BUY'][result['signal']+1]}")
```

## File: models/lstm_model_load_patch.py
```python
import logging
from pathlib import Path
import torch

from models.lstm_model import LSTMNet, LSTMPredictor

logger = logging.getLogger(__name__)

def patched_load(self: LSTMPredictor):
    """
    1) 기존 체크포인트가 state_dict만 있는 경우도 지원.
    2) meta 정보(input_size 등)가 없으면, feature_cols 길이로 input_size를 추론.
    """
    ckpt_path = Path(self.model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    logger.info(f"Loading LSTM model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

    # -------------------------
    # case 1: 단순 state_dict만 있는 경우
    # -------------------------
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        state_dict = ckpt if isinstance(ckpt, dict) else ckpt
        # feature_cols는 예측 시점(df_feat에서) 자동으로 맞출 거라,
        # 여기서는 input_size만 seq의 feature 차원 길이로 나중에 재구성할 거라고 가정한다.
        if self.feature_cols is None:
            # 일단 빈 리스트로 두고, predict_future에서 df_feat.columns로 채운다.
            self.feature_cols = []
        # input_size는 실제 forward에서 사용되므로, 일단 dummy로 1로 두었다가
        # predict_future 호출 시점에 재생성하는 전략을 쓸 수도 있지만,
        # 여기서는 간단히 "한 번 더 저장해서 새 포맷으로 쓰자" 쪽이 안전하다.
        input_size = 1
        self.model = LSTMNet(input_size=input_size)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info("LSTM model (state_dict-only) loaded with dummy input_size=1.")
        return

    # -------------------------
    # case 2: 우리가 새로 정의한 포맷(dict 안에 meta 포함)
    # -------------------------
    state_dict = ckpt["state_dict"]
    input_size = ckpt.get("input_size")
    feature_cols = ckpt.get("feature_cols")

    if input_size is None and feature_cols is not None:
        input_size = len(feature_cols)

    if input_size is None:
        raise RuntimeError(
            "Checkpoint missing 'input_size'. "
            "Retrain LSTM once with the new saver to embed meta info."
        )

    self.feature_cols = feature_cols or self.feature_cols
    if "scaler" in ckpt:
        self.scaler = ckpt["scaler"]
    self.seq_len = ckpt.get("seq_len", self.seq_len)
    self.horizon = ckpt.get("horizon", self.horizon)

    self.model = LSTMNet(input_size=input_size)
    self.model.load_state_dict(state_dict)
    self.model.to(self.device)
    self.model.eval()
    logger.info("LSTM model loaded successfully (meta checkpoint).")

# 실제로 메서드 덮어쓰기
LSTMPredictor.load = patched_load
```

## File: models/lstm_model.py
```python
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class LSTMPredictor:
    def __init__(self, seq_len=60, horizon=60, model_path="models/lstm_model.pt", device=None):
        self.seq_len = seq_len
        self.horizon = horizon
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None

    # -----------------------------
    # 학습용 데이터 준비
    # -----------------------------
    def prepare_data(self, df_feat):
        """
        df_feat: feature dataframe (index: datetime)
        마지막 close 기준 미래 horizon 수익률 같은 걸 예측한다고 가정.
        여기서는 예시로 'close' 컬럼을 타겟으로 사용.
        """
        if "close" not in df_feat.columns:
            raise ValueError("df_feat must contain 'close' column for target.")

        self.feature_cols = [c for c in df_feat.columns if c != "close"]

        X = df_feat[self.feature_cols].values
        y = df_feat["close"].values

        # 스케일링
        X_scaled = self.scaler.fit_transform(X)

        seq_len = self.seq_len
        X_list, y_list = [], []
        for i in range(len(X_scaled) - seq_len - self.horizon + 1):
            X_list.append(X_scaled[i : i + seq_len])
            # horizon 뒤 close 값
            y_list.append(y[i + seq_len + self.horizon - 1])

        X_arr = np.array(X_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        return X_arr, y_arr

    # -----------------------------
    # 학습
    # -----------------------------
    def train(self, X, y, epochs=10, lr=1e-3, batch_size=64):
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_size = X.shape[-1]
        self.model = LSTMNet(input_size=input_size).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

            avg_loss = total_loss / len(dataset)
            logger.info(f"[LSTM] Epoch {epoch}/{epochs} - loss={avg_loss:.6f}")

        self.save()

    # -----------------------------
    # 저장 / 로드
    # -----------------------------
    def save(self):
        if self.model is None:
            raise RuntimeError("Model is not trained, cannot save.")

        ckpt = {
            "state_dict": self.model.state_dict(),
            "input_size": self.model.lstm.input_size,
            "feature_cols": self.feature_cols,
            "scaler": self.scaler,
            "seq_len": self.seq_len,
            "horizon": self.horizon,
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, self.model_path)
        logger.info(f"LSTM model saved to {self.model_path}")

    def load(self):
        """
        PyTorch 2.6부터 torch.load 기본값 weights_only=True라서
        전체 checkpoint를 읽으려면 weights_only=False를 명시해야 함.[web:105][web:108][web:112]
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        logger.info(f"Loading LSTM model from {self.model_path}")
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)

        input_size = ckpt["input_size"]
        self.feature_cols = ckpt["feature_cols"]
        self.scaler = ckpt["scaler"]
        self.seq_len = ckpt.get("seq_len", self.seq_len)
        self.horizon = ckpt.get("horizon", self.horizon)

        self.model = LSTMNet(input_size=input_size)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        logger.info("LSTM model loaded successfully.")

    # -----------------------------
    # 예측
    # -----------------------------
    def predict_future(self, df_feat):
        """
        df_feat 전체를 받아서 마지막 seq_len 구간 기준으로
        horizon만큼 미래 값을 예측한 시계열 Series를 반환.[web:96][web:112]
        """
        import pandas as pd

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self.feature_cols is None:
            raise RuntimeError("feature_cols is not set in LSTMPredictor.")

        X = df_feat[self.feature_cols].values

        if len(X) < self.seq_len:
            raise ValueError(f"Not enough data for seq_len={self.seq_len}: len={len(X)}")

        seq = X[-self.seq_len :]

        X_scaled = self.scaler.transform(seq)
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            y_pred = self.model(x_tensor).cpu().numpy().flatten()

        index = pd.date_range(
            start=df_feat.index[-1] + pd.Timedelta(minutes=1),
            periods=self.horizon,
            freq="T",
        )
        series = pd.Series(y_pred[0], index=index)
        return series
```

## File: models/transformer_model.py
```python
import torch
import torch.nn as nn

class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        h = self.input_proj(x)
        h = self.encoder(h)
        h_last = h[:, -1, :]
        logit = self.cls_head(h_last).squeeze(-1)
        return logit
```

## File: risk/__init__.py
```python

```

## File: strategy/__init__.py
```python

```

## File: strategy/kelly_sizing.backup.py
```python
"""
Kelly-style position sizing for SOTA Upbit bot (2026).
- confidence: 0.0 ~ 1.0 (CatBoost / Ensemble score)
- balance: current KRW balance
- max_frac: per-trade max fraction of equity (e.g. 0.04 major)
- SMALL_BALANCE 영역(<= 20만)에서는 계정 부트스트랩을 위해 더 공격적으로 사용.
"""

from dataclasses import dataclass

MIN_ORDER_KRW = 5000        # 업비트 최소 주문 단위 + 여유
SMALL_BALANCE_KRW = 200000  # 소액 계정 구간 기준

# 튜닝 결과 반영: conf 0.70 미만은 아예 트레이드 안 함
CONF_THRESHOLD = 0.70
HIGH_CONF = 0.85
VERY_HIGH_CONF = 0.92


@dataclass
class MarketSizingConfig:
    max_frac: float = 0.04   # 메이저 코인: 4% 상한 (BTC 튜닝 결과 기반)
    min_frac: float = 0.01   # 한 번에 최소 1% 이상은 써야 의미 있음
    is_small_cap: bool = False


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def get_position_size(balance_krw: float,
                      confidence: float,
                      cfg: MarketSizingConfig | None = None) -> int:
    """
    Return position size in KRW.
    - 잃을 때: Kelly fraction + Kill switch가 작게 베팅하도록
    - 딸 때: confidence 높을수록, balance 클수록 더 크게 베팅
    """
    if cfg is None:
        cfg = MarketSizingConfig()

    # 1) 비정상 입력 방어
    if balance_krw <= 0:
        return 0
    if confidence is None or not (0.0 <= confidence <= 1.0):
        return 0

    # 2) confidence threshold 이하 → 무조건 스킵
    if confidence < CONF_THRESHOLD:
        return 0

    # 3) 잔고가 너무 작으면 트레이드 자체를 막음
    if balance_krw < MIN_ORDER_KRW:
        return 0

    # 4) 기본 Kelly-style 비율 (confidence 선형 매핑)
    #    - conf=0.70 → 약 0.02
    #    - conf=0.85 → 약 0.03
    #    - conf=1.00 → 약 0.04 (단, cfg.max_frac로 캡)
    base_frac = (confidence - CONF_THRESHOLD) / (1.0 - CONF_THRESHOLD)  # 0~1
    kelly_frac = 0.02 + base_frac * 0.02  # 2% ~ 4%
    kelly_frac = clamp(kelly_frac, cfg.min_frac, cfg.max_frac)

    # 5) 소액 계정 부트스트랩 로직
    if balance_krw <= SMALL_BALANCE_KRW:
        # 소액 계정에서는 고신뢰 구간에서 더 과감하게,
        # 중간 신뢰 구간에서는 "최소 주문 + 약간의 Kelly" 정도만.
        if confidence >= VERY_HIGH_CONF:
            frac = min(0.20, cfg.max_frac * 2.5)  # 최대 20%까지
        elif confidence >= HIGH_CONF:
            frac = min(0.12, cfg.max_frac * 2.0)  # 최대 12%까지
        else:
            frac = max(0.03, kelly_frac * 1.5)
    else:
        # 일반 계정: 튜닝된 Kelly 기반 + 고신뢰에서 살짝 상향
        frac = kelly_frac
        if confidence >= VERY_HIGH_CONF:
            frac *= 1.3
        elif confidence >= HIGH_CONF:
            frac *= 1.1

    # 6) 대형 알트 / 잡코인은 상한을 더 낮게 (예시: 3% / 2%)
    if cfg.is_small_cap:
        frac = min(frac, 0.03)

    frac = clamp(frac, cfg.min_frac, cfg.max_frac)
    size = int(balance_krw * frac)

    # 7) 최소 주문 단위 이하라면 아예 트레이드 안 함
    if size < MIN_ORDER_KRW:
        return 0

    # float → int KRW (안전하게 한 틱 아래로 내림)
    return int(size)
```

## File: strategy/kelly_sizing.py
```python
"""
Kelly-style position sizing for SOTA Upbit bot (2026).

- confidence: 0.0 ~ 1.0 (CatBoost / Ensemble score)
- balance_krw: current KRW balance

Design (v2026-02-05):
- CONF_THRESHOLD = 0.70: 이 미만은 무조건 사이즈 0.
- Base Kelly fraction: 2% ~ 4% of equity (major 기준).[file:1][file:3]
- SMALL_BALANCE 계정(<= 200k KRW)에서는 high/very-high conf에서 좀 더 공격적.
- Regime-level risk-off 지원:
    * cfg.max_frac <= 0 이면 무조건 0 KRW 사이즈 (예: CRASH 레짐).
"""

from dataclasses import dataclass

MIN_ORDER_KRW = 5000        # 업비트 최소 주문 단위 + 여유
SMALL_BALANCE_KRW = 200000  # 소액 계정 기준

# 튜닝 반영: conf 0.70 미만은 아예 트레이드 안 함.[file:1][file:3]
CONF_THRESHOLD = 0.70
HIGH_CONF = 0.85
VERY_HIGH_CONF = 0.92


@dataclass
class MarketSizingConfig:
    # 메이저 기본: 4% 상한, 최소 1%
    max_frac: float = 0.04
    min_frac: float = 0.01
    is_small_cap: bool = False


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def get_position_size(
    balance_krw: float,
    confidence: float,
    cfg: MarketSizingConfig,
) -> float:
    """
    Return position size in KRW.

    - Regime-level 하드 킬:
        cfg.max_frac <= 0 -> 엔트리 완전 금지.
    - 소액 계정:
        VERY_HIGH_CONF: 최대 20% (단, cfg.max_frac 이내),
        HIGH_CONF: 최대 12% (단, cfg.max_frac 이내).[file:1][file:3]
    """
    # 0) Regime-level hard kill (예: CRASH 레짐)
    if cfg.max_frac <= 0:
        return 0.0

    # 1) Confidence gate
    if confidence < CONF_THRESHOLD:
        return 0.0

    # 2) Base Kelly fraction: 2% ~ 4%
    #    - conf=0.70 -> 0.02
    #    - conf=1.00 -> 0.04
    base_frac = (confidence - CONF_THRESHOLD) / (1.0 - CONF_THRESHOLD)
    base_frac = max(0.0, min(1.0, base_frac))
    kelly_frac = 0.02 + base_frac * 0.02  # 2% ~ 4%
    kelly_frac = clamp(kelly_frac, cfg.min_frac, cfg.max_frac)

    # 3) Small-balance bootstrap logic
    if balance_krw <= SMALL_BALANCE_KRW:
        if confidence >= VERY_HIGH_CONF:
            # 극고신뢰: 최대 20%까지 허용 (단, cfg.max_frac 이내)
            kelly_frac = max(kelly_frac, min(0.20, cfg.max_frac))
        elif confidence >= HIGH_CONF:
            # 고신뢰: 최대 12%까지 허용 (단, cfg.max_frac 이내)
            kelly_frac = max(kelly_frac, min(0.12, cfg.max_frac))

    # 4) Small-cap 경우 재캡
    if cfg.is_small_cap:
        kelly_frac = min(kelly_frac, cfg.max_frac)

    size_krw = balance_krw * kelly_frac

    # 5) 최소 주문 금액 체크
    if size_krw < MIN_ORDER_KRW:
        return 0.0

    return float(size_krw)
```

## File: strategy/risk_manager.py
```python
"""
RiskManager:
- per-trade stop / take-profit / trailing-stop 규칙
- 일일 손익 / 연속 손실 / 최대 낙폭 기반 Kill Switch
"""

from dataclasses import dataclass, field


@dataclass
class GlobalRiskConfig:
    daily_loss_limit: float = -0.08   # 하루 -8% 이하면 거래 중단
    max_drawdown: float = -0.20       # 전체 자본 기준 -20% MDD
    max_consec_losses: int = 10       # 연속 손실 횟수 제한


@dataclass
class TradeRiskConfig:
    """Per-trade risk config (SL/TP/trailing). Updated 2026-02-05.

    - stop_loss_pct: 0.006 (-0.6%)  — BTC RANGE regime sweep 최적값.[file:1]
    - take_profit_pct: 0.010 (+1.0%)
    - trailing_pct: 0.0            — trailing-stop 비활성
    """
    stop_loss_pct: float = 0.006   # -0.6% (BTC RANGE sweep 최적)
    take_profit_pct: float = 0.010 # +1.0%
    trailing_pct: float = 0.0      # trailing-stop 비활성


@dataclass
class RiskState:
    equity_start: float
    equity: float
    peak_equity: float
    daily_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    consec_losses: int = 0
    trading_paused: bool = False
    pause_reason: str | None = None


class RiskManager:
    def __init__(self,
                 equity_start: float,
                 global_cfg: GlobalRiskConfig | None = None,
                 trade_cfg: TradeRiskConfig | None = None):
        self.global_cfg = global_cfg or GlobalRiskConfig()
        self.trade_cfg = trade_cfg or TradeRiskConfig()
        self.state = RiskState(
            equity_start=equity_start,
            equity=equity_start,
            peak_equity=equity_start,
        )

    # -------- per-trade helpers --------
    def get_stop_take_levels(self, entry_price: float) -> tuple[float, float]:
        sl = entry_price * (1.0 - self.trade_cfg.stop_loss_pct)
        tp = entry_price * (1.0 + self.trade_cfg.take_profit_pct)
        return sl, tp

    # -------- global risk accounting --------
    def register_trade(self, pnl_pct: float):
        """
        pnl_pct: trade-level PnL in percent (e.g. 0.01 = +1%)
        """
        if self.state.trading_paused:
            return

        # equity 업데이트
        self.state.equity *= (1.0 + pnl_pct)
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

        # 일일 손익 (단순 누적)
        self.state.daily_pnl_pct += pnl_pct

        # 최대 낙폭 갱신
        dd = (self.state.equity / self.state.peak_equity) - 1.0
        self.state.max_drawdown_pct = min(self.state.max_drawdown_pct, dd)

        # 연속 손실 카운트
        if pnl_pct < 0:
            self.state.consec_losses += 1
        else:
            self.state.consec_losses = 0

        # Kill Switch 조건 체크
        self._check_kill_switch()

    def _check_kill_switch(self):
        if self.state.trading_paused:
            return

        if self.state.daily_pnl_pct <= self.global_cfg.daily_loss_limit:
            self.state.trading_paused = True
            self.state.pause_reason = f"DAILY_LOSS {self.state.daily_pnl_pct:.3f}"
            return

        if self.state.max_drawdown_pct <= self.global_cfg.max_drawdown:
            self.state.trading_paused = True
            self.state.pause_reason = f"MAX_DRAWDOWN {self.state.max_drawdown_pct:.3f}"
            return

        if self.state.consec_losses >= self.global_cfg.max_consec_losses:
            self.state.trading_paused = True
            self.state.pause_reason = f"CONSEC_LOSSES {self.state.consec_losses}"

    def should_pause_trading(self) -> bool:
        return self.state.trading_paused

    def get_pause_reason(self) -> str | None:
        return self.state.pause_reason
```
