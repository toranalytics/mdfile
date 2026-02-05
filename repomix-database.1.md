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
- Only files matching these patterns are included: database/**/*
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
database/
  __init__.py
  connection.py
  features.db
  schema.sql
```

# Files

## File: database/__init__.py
```python

```

## File: database/connection.py
```python
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

def get_db_config():
    import subprocess
    actual_user = subprocess.check_output(['whoami']).decode().strip()
    
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'upbit_trading'),
        'user': os.getenv('DB_USER', actual_user),
        'password': os.getenv('DB_PASSWORD', '')
    }

def get_connection():
    try:
        config = get_db_config()
        conn = psycopg2.connect(**config)
        return conn
    except Exception as e:
        logger.error(f"DB connection failed: {e}")
        raise

def test_connection():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ PostgreSQL: {version[0][:60]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing database connection...")
    test_connection()
```

## File: database/features.db
```

```

## File: database/schema.sql
```sql
CREATE TABLE IF NOT EXISTS tick_data_raw (
    time TIMESTAMPTZ NOT NULL,
    market TEXT NOT NULL,
    trade_price DOUBLE PRECISION NOT NULL,
    trade_volume DOUBLE PRECISION NOT NULL,
    ask_bid TEXT,
    bid1_price DOUBLE PRECISION,
    bid1_size DOUBLE PRECISION,
    ask1_price DOUBLE PRECISION,
    ask1_size DOUBLE PRECISION,
    sequential_id BIGINT,
    prev_closing_price DOUBLE PRECISION,
    change_price DOUBLE PRECISION,
    PRIMARY KEY (time, market)
);

CREATE INDEX IF NOT EXISTS idx_tick_market_time ON tick_data_raw (market, time DESC);

CREATE TABLE IF NOT EXISTS candles_1m (
    time TIMESTAMPTZ NOT NULL,
    market TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    buy_volume DOUBLE PRECISION,
    sell_volume DOUBLE PRECISION,
    tick_count INTEGER,
    vwap DOUBLE PRECISION,
    bid_ask_spread DOUBLE PRECISION,
    PRIMARY KEY (time, market)
);

CREATE INDEX IF NOT EXISTS idx_candles_1m_market_time ON candles_1m (market, time DESC);

CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    size DOUBLE PRECISION NOT NULL,
    exit_time TIMESTAMPTZ,
    exit_price DOUBLE PRECISION,
    exit_reason TEXT,
    pnl_amt DOUBLE PRECISION,
    pnl_pct DOUBLE PRECISION,
    expected_return DOUBLE PRECISION,
    p_profit DOUBLE PRECISION,
    kelly_fraction DOUBLE PRECISION,
    confidence_score DOUBLE PRECISION,
    regime TEXT,
    status TEXT DEFAULT 'OPEN',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions (status, entry_time DESC);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    position_id TEXT REFERENCES positions(position_id),
    time TIMESTAMPTZ NOT NULL,
    market TEXT NOT NULL,
    side TEXT NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    size DOUBLE PRECISION NOT NULL,
    fee DOUBLE PRECISION,
    order_id TEXT,
    execution_type TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_time ON trades (time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_position ON trades (position_id);
```
