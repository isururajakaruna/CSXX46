# 📊 Simple MA Strategy - Config Parameter Analysis

Analysis of which parameters in your config are **actually used** by the strategy.

---

## ✅ **USED Parameters** (Keep These)

### **Core Strategy Parameters**

| Parameter | Value in Your Config | Purpose | Default |
|-----------|---------------------|---------|---------|
| `base_symbol` | `"BTC"` | ✅ **REQUIRED** - Base trading asset | N/A |
| `quote_symbol` | `"USDT"` | ✅ **REQUIRED** - Quote trading asset | N/A |
| `log_trade_activity` | `false` | ✅ Enable/disable trade logging | N/A |
| `order_value_pct` | `10` | ✅ % of equity per trade | N/A |

### **Moving Average Parameters**

| Parameter | Value | Purpose | Default |
|-----------|-------|---------|---------|
| `short_ma_window_length` | `30` | ✅ Fast MA period | N/A |
| `long_ma_window_length` | `150` | ✅ Slow MA period | N/A |
| `is_short_ma_weighted` | `false` | ✅ Use WMA vs SMA for short MA | N/A |
| `is_long_ma_weighted` | `false` | ✅ Use WMA vs SMA for long MA | N/A |
| `ma_candle_length` | `60` | ✅ Candle aggregation for MA | `1` |

### **Indicator Parameters**

| Parameter | Value | Purpose | Default |
|-----------|-------|---------|---------|
| `std_window_length` | `300` | ✅ Volatility window for TP/SL | N/A |
| `rsi_d` | `30` | ✅ RSI threshold (buy>70, sell<30) | `20` |
| `rsi_candle_length` | `60` | ✅ Candle aggregation for RSI | `1` |
| `use_obv` | `true` | ✅ Use OBV instead of RSI | `false` |
| `obv_window_length` | `5` | ✅ OBV lookback window | N/A |

### **Position Management**

| Parameter | Value | Purpose | Default |
|-----------|-------|---------|---------|
| `go_long` | `true` | ✅ Allow long positions | `true` |
| `go_short` | `true` | ✅ Allow short positions | `false` |
| `tp_multiple` | `4` | ✅ Take-profit = 4 × volatility | `4` |
| `sl_multiple` | `8` | ✅ Stop-loss = 8 × volatility | `8` |

### **Liquidation Parameters** (from AdvancedBaseStrategy)

| Parameter | Value | Purpose | Default |
|-----------|-------|---------|---------|
| `enable_auto_liquidation` | `true` | ✅ Auto-rebalance when imbalanced | `false` |

### **Pending Order Manager Parameters** (from AdvancedBaseStrategy)

| Parameter | Value | Purpose | Default |
|-----------|-------|---------|---------|
| `use_pom` | `false` | ✅ Use Pending Order Manager | `false` |
| `almost_market_order_perc` | `0.01` | ✅ POM: almost-market threshold | `0.01` |
| `price_gap_perc_cancel` | `5` | ✅ POM: cancel order gap % | `5` |
| `price_gap_perc_reenter` | `0.05` | ✅ POM: re-enter gap % | `0.05` |

---

## ❌ **UNUSED Parameter** (Remove This)

| Parameter | Value | Why Unused |
|-----------|-------|------------|
| `trade_interval` | `900` | ❌ **NOT READ by strategy at all!** No code references it. |

---

## 🎯 **Cleaned Config (Recommended)**

```json
{
  "exchange": {
    "config": {
      "extra": {
        "data_source": "ats/data/BTC_USDT_very_short.csv",
        "fees": {
          "config": {
            "limit": {
              "buy": {"base": 0, "quote": 0},
              "sell": {"base": 0, "quote": 0}
            },
            "market": {
              "buy": {"base": 0, "quote": 0},
              "sell": {"base": 0, "quote": 0}
            }
          },
          "namespace": "fees:generic"
        },
        "min_trading_size": 0.0001,
        "wallet": {
          "assets": {
            "BTC": 0,
            "USDT": 10000
          }
        }
      },
      "plot_data_max_len": -1,
      "symbol": {
        "base": "BTC",
        "quote": "USDT"
      }
    },
    "namespace": "exchanges:back_trading"
  },
  "strategy": {
    "config": {
      "base_symbol": "BTC",
      "quote_symbol": "USDT",
      
      "_core": "=== CORE SETTINGS ===",
      "log_trade_activity": false,
      "order_value_pct": 10,
      
      "_ma": "=== MOVING AVERAGE SETTINGS ===",
      "short_ma_window_length": 30,
      "long_ma_window_length": 150,
      "is_short_ma_weighted": false,
      "is_long_ma_weighted": false,
      "ma_candle_length": 60,
      
      "_indicators": "=== INDICATOR SETTINGS ===",
      "std_window_length": 300,
      "rsi_d": 30,
      "rsi_candle_length": 60,
      "use_obv": true,
      "obv_window_length": 5,
      
      "_position": "=== POSITION MANAGEMENT ===",
      "go_long": true,
      "go_short": true,
      "tp_multiple": 4,
      "sl_multiple": 8,
      
      "_liquidation": "=== AUTO-LIQUIDATION ===",
      "enable_auto_liquidation": true,
      
      "_pom": "=== PENDING ORDER MANAGER (POM) ===",
      "use_pom": false,
      "almost_market_order_perc": 0.01,
      "price_gap_perc_cancel": 5,
      "price_gap_perc_reenter": 0.05
    },
    "namespace": "strategies:simple_ma_strategy"
  }
}
```

---

## 📝 **Summary**

### **Your Config Status:**
- ✅ **21 parameters used** correctly
- ❌ **1 parameter unused** (`trade_interval`)
- 📊 **95% efficiency** (only 1 wasted param)

### **Recommendations:**

1. ✅ **Remove `trade_interval: 900`** - it does nothing
2. ✅ **Keep everything else** - all other params are read by the strategy
3. 💡 **Consider:**
   - Your `use_pom: false` means POM params are loaded but not used
   - Your `use_obv: true` means RSI is calculated but not used for signals

---

## 🔍 **How Strategy Uses Your Config**

### **Signal Generation:**

1. **Primary Signal:** MA Crossover
   - Fast MA (30) crosses above Slow MA (150) → BUY signal
   - Fast MA (30) crosses below Slow MA (150) → SELL signal

2. **Confirmation Signal:** OBV (since `use_obv: true`)
   - OBV at max in last 5 periods → BUY confirmation
   - OBV at min in last 5 periods → SELL confirmation
   - *(RSI is ignored because `use_obv: true`)*

3. **Exit Conditions:**
   - Take-Profit: 4 × volatility (std)
   - Stop-Loss: 8 × volatility (std)

---

## 💡 **Optimization Tips**

### **Currently:**
- MA: 30/150 with 60-candle aggregation (very slow signals)
- OBV: 5-period lookback (very responsive)
- **Conflict:** Slow MAs + Fast OBV = mixed signal speed

### **Suggestions:**

**For Faster Trading:**
```json
{
  "short_ma_window_length": 10,
  "long_ma_window_length": 50,
  "ma_candle_length": 1,
  "obv_window_length": 3
}
```

**For Slower/Safer Trading:**
```json
{
  "short_ma_window_length": 50,
  "long_ma_window_length": 200,
  "ma_candle_length": 60,
  "obv_window_length": 10
}
```

---

## ⚠️ **Important Notes**

1. **Candle Length Consistency:**
   - Your `ma_candle_length: 60` and `rsi_candle_length: 60` mean indicators look at 60-min candles
   - This makes signals VERY slow (updates only every hour if your base candles are 1-min)

2. **POM (Pending Order Manager):**
   - You have `use_pom: false`, so these params do nothing:
     - `almost_market_order_perc`
     - `price_gap_perc_cancel`
     - `price_gap_perc_reenter`
   - **But they're harmless** - just loaded and ignored

3. **RSI vs OBV:**
   - Your `use_obv: true` means RSI is calculated but NOT used for signals
   - If you want RSI signals: set `use_obv: false`

---

**Version**: 1.0  
**Analyzed**: November 12, 2025  
**Strategy**: `strategies:simple_ma_strategy`

