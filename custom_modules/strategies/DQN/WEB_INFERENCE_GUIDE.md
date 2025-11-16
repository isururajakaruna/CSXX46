# 🌐 DQN Web Interface - Inference Guide

Complete guide to running DQN inference through the ATS web portal.

---

## 📋 **Quick Start**

### **Step 1: Train a Model**

```bash
cd custom_modules/custom_scripts
./train_interactive_DQN.sh
```

Your model will be saved to:
```
custom_modules/strategies/DQN/saved_models/run_YYYYMMDD_HHMMSS/dqn_final.pth
```

---

### **Step 2: Start ATS Server**

```bash
cd /path/to/ats
./start.sh
```

Wait for:
- Backend: `http://localhost:5010` 
- Frontend: `http://localhost:3000`

---

### **Step 3: Create Inference Config**

Copy and edit `inference_config_web.json`:

```json
{
  "exchange": {
    "config": {
      "extra": {
        "data_source": "ats/data/BTC_USDT_short.csv",
        "fees": {
          "config": {
            "limit": {
              "buy": {"base": 0, "quote": 0},
              "sell": {"base": 0, "quote": 0}
            },
            "market": {
              "buy": {"base": 0.1, "quote": 0},
              "sell": {"base": 0, "quote": 0.1}
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
      
      "training_mode": false,
      "model_path": "custom_modules/strategies/DQN/saved_models/run_20251111_232359/dqn_final.pth",
      "epsilon": 0.0,
      "decision_interval": 5,
      
      "order_value_pct": 10,
      "max_position_size": 1.0,
      "log_trade_activity": true,
      
      "use_indicators": true,
      "ma_window": 20,
      "rsi_window": 14
    },
    "namespace": "strategies:DQN"
  }
}
```

**Key Parameters to Update:**
- `model_path`: Your trained model (relative path)
- `data_source`: Your data file (relative path)
- `wallet.assets`: Initial balance
- `decision_interval`: How often to make decisions (5 = every 5 candles)

---

### **Step 4: Submit via Web UI**

1. Open `http://localhost:3000`
2. Navigate to **Jobs** → **Create Job**
3. Paste your JSON config
4. Click **Submit**
5. Monitor job status and results

---

## 🔧 **Configuration Details**

### **Exchange Section**

```json
"exchange": {
  "config": {
    "extra": {
      "data_source": "ats/data/BTC_USDT_short.csv",  // ← Change this
      "wallet": {
        "assets": {
          "BTC": 0,      // ← Starting BTC
          "USDT": 10000  // ← Starting USDT
        }
      }
    },
    "symbol": {
      "base": "BTC",   // ← Base asset
      "quote": "USDT"  // ← Quote asset
    }
  },
  "namespace": "exchanges:back_trading"  // ← Backtesting exchange
}
```

### **Strategy Section - DQN Inference**

```json
"strategy": {
  "config": {
    // Required for inference
    "training_mode": false,  // ← MUST be false
    "model_path": "custom_modules/strategies/DQN/saved_models/run_20251111_232359/dqn_final.pth",  // ← Your model
    "epsilon": 0.0,  // ← 0.0 = greedy (no exploration)
    "decision_interval": 5,  // ← Make decision every 5 candles (1 = every candle)
    
    // Trading parameters
    "order_value_pct": 10,      // ← % of equity per trade
    "max_position_size": 1.0,   // ← Max position size
    "log_trade_activity": true, // ← Log each trade
    
    // Indicators (must match training)
    "use_indicators": true,
    "ma_window": 20,
    "rsi_window": 14
  },
  "namespace": "strategies:DQN"  // ← Points to custom_modules/strategies/DQN
}
```

---

## 📁 **File Paths**

### **Model Path (Relative)**

✅ **Correct:**
```json
"model_path": "custom_modules/strategies/DQN/saved_models/run_20251111_232359/dqn_final.pth"
```

❌ **Wrong:**
```json
"model_path": "/Users/username/ats/custom_modules/..."  // Absolute paths won't work
"model_path": "saved_models/run_20251111_232359/dqn_final.pth"  // Missing custom_modules/strategies/DQN
```

### **Data Source (Relative)**

✅ **Correct:**
```json
"data_source": "ats/data/BTC_USDT_short.csv"
"data_source": "custom_modules/data/my_data.csv"
```

❌ **Wrong:**
```json
"data_source": "/Users/username/ats/data/..."  // Absolute paths might not work across environments
```

---

## ⏱️ **Decision Interval - Controlling Trading Frequency**

### **What is `decision_interval`?**

The `decision_interval` parameter controls **how often** the DQN agent makes trading decisions:

- `decision_interval: 1` → Make decision **every candle** (most frequent)
- `decision_interval: 5` → Make decision **every 5 candles**
- `decision_interval: 10` → Make decision **every 10 candles** (less frequent)

### **Why use it?**

**Problem**: Without `decision_interval`, the DQN runs on every candle, which can lead to:
- ❌ Over-trading (too many trades)
- ❌ High transaction costs  
- ❌ Noisy signals
- ❌ Rapid position changes

**Solution**: Set `decision_interval > 1` to:
- ✅ Reduce trading frequency
- ✅ Lower transaction costs
- ✅ Smoother trading patterns
- ✅ Focus on meaningful market movements

### **Recommended Values**

| Candle Period | Recommended `decision_interval` | Trading Frequency |
|---------------|--------------------------------|-------------------|
| 1-minute candles | 5-10 | Every 5-10 minutes |
| 5-minute candles | 3-6 | Every 15-30 minutes |
| 15-minute candles | 2-4 | Every 30-60 minutes |
| 1-hour candles | 1-3 | Every 1-3 hours |

### **Example**

If your data has **1-minute candles** and `decision_interval: 5`:
- ✅ Candle 1: Observe (HOLD)
- ✅ Candle 2: Observe (HOLD)
- ✅ Candle 3: Observe (HOLD)
- ✅ Candle 4: Observe (HOLD)
- 🎯 **Candle 5: Make decision (BUY/SELL/HOLD)**
- ✅ Candle 6: Observe (HOLD)
- ...

---

## 🎯 **Configuration Examples**

### **Example 1: Conservative Trading**

```json
{
  "exchange": { ... },
  "strategy": {
    "config": {
      "training_mode": false,
      "model_path": "custom_modules/strategies/DQN/saved_models/run_20251111_232359/dqn_final.pth",
      "epsilon": 0.0,
      "decision_interval": 10,     // ← Less frequent decisions (every 10 candles)
      
      "order_value_pct": 5,        // ← Small positions (5%)
      "max_position_size": 0.5,    // ← Max 50% in market
      "log_trade_activity": true
    },
    "namespace": "strategies:DQN"
  }
}
```

### **Example 2: Aggressive Trading**

```json
{
  "exchange": { ... },
  "strategy": {
    "config": {
      "training_mode": false,
      "model_path": "custom_modules/strategies/DQN/saved_models/run_20251111_232359/dqn_final.pth",
      "epsilon": 0.0,
      "decision_interval": 3,      // ← Frequent decisions (every 3 candles)
      
      "order_value_pct": 20,       // ← Large positions (20%)
      "max_position_size": 1.0,    // ← Max 100% in market
      "log_trade_activity": true
    },
    "namespace": "strategies:DQN"
  }
}
```

### **Example 3: With Exploration**

```json
{
  "exchange": { ... },
  "strategy": {
    "config": {
      "training_mode": false,
      "model_path": "custom_modules/strategies/DQN/saved_models/run_20251111_232359/dqn_final.pth",
      "epsilon": 0.1,              // ← 10% random actions
      "decision_interval": 5,
      
      "order_value_pct": 10,
      "max_position_size": 1.0,
      "log_trade_activity": true
    },
    "namespace": "strategies:DQN"
  }
}
```

---

## 🐛 **Troubleshooting**

### **"Module 'DQN' not found"**

**Problem**: Namespace is incorrect

**Solution**: Ensure namespace is exactly `"strategies:DQN"` (case-sensitive)

### **"Model file not found"**

**Problem**: Model path is incorrect or absolute

**Solutions:**
1. List available models:
   ```bash
   ls -lh custom_modules/strategies/DQN/saved_models/*/dqn_final.pth
   ```
2. Use **relative path** from project root
3. Check model actually exists

### **"Invalid strategy configuration"**

**Problem**: Missing required parameters

**Solution**: Ensure these are present:
- `training_mode: false`
- `model_path: "..."`
- `base_symbol` and `quote_symbol`

### **Poor Performance / Random Actions**

**Possible Causes:**
1. Model undertrained (train more episodes)
2. Wrong model checkpoint (use `dqn_final.pth`)
3. High epsilon value (use `0.0` for pure inference)
4. Indicator mismatch (training vs inference configs)
5. Different data distribution (train/test data mismatch)

---

## 📊 **Viewing Results**

### **Via Web UI**

1. Go to **Jobs** → **Job List**
2. Find your job by name/ID
3. Click **View Details**
4. Check:
   - **Status**: Running/Completed/Failed
   - **Metrics**: P&L, Win Rate, etc.
   - **Trades**: Buy/Sell history
   - **Charts**: Price action and trades

### **Via API**

```bash
# Get job status
curl http://localhost:5010/v1/jobs/{job_id}

# Get results
curl http://localhost:5010/v1/jobs/{job_id}/results

# Get trade history
curl http://localhost:5010/v1/jobs/{job_id}/trades
```

---

## 🔄 **Comparing Multiple Models**

### **Test Different Training Runs**

Create separate configs for each model:

**Config 1**: `inference_5ep.json`
```json
{
  ...
  "strategy": {
    "config": {
      "model_path": "custom_modules/strategies/DQN/saved_models/run_1/dqn_final.pth",
      ...
    }
  }
}
```

**Config 2**: `inference_20ep.json`
```json
{
  ...
  "strategy": {
    "config": {
      "model_path": "custom_modules/strategies/DQN/saved_models/run_2/dqn_final.pth",
      ...
    }
  }
}
```

Submit both, then compare results in the web UI.

---

## 📈 **Performance Metrics**

After job completes, check:

| Metric | Description |
|--------|-------------|
| **Total Return %** | Overall profit/loss |
| **Win Rate** | % of profitable trades |
| **Sharpe Ratio** | Risk-adjusted returns |
| **Max Drawdown** | Largest loss from peak |
| **Number of Trades** | Trading frequency |
| **Avg Trade Duration** | How long positions are held |
| **Best/Worst Trade** | Highest gain/loss |

---

## 🎓 **Best Practices**

### **✅ DO:**
- Set `training_mode: false` for inference
- Set `epsilon: 0.0` for pure exploitation
- Use **relative paths** (not absolute)
- Use `dqn_final.pth` (fully trained)
- Test on **unseen data**
- Match indicator configs with training
- Start with small `order_value_pct` (5-10%)

### **❌ DON'T:**
- Use `training_mode: true` (will fail)
- Use high epsilon in production
- Use absolute paths with usernames
- Use early checkpoints for production
- Test on same data used for training
- Change indicator windows from training

---

## ✅ **Pre-Flight Checklist**

Before submitting job:

- [ ] ATS server running (`./start.sh`)
- [ ] Model trained and path correct
- [ ] `training_mode` set to `false`
- [ ] `epsilon` set to `0.0`
- [ ] `namespace` is `"strategies:DQN"`
- [ ] Data source exists
- [ ] Wallet balance configured
- [ ] Paths are **relative** (not absolute)

---

## 🔗 **Related Files**

- `inference_config_web.json` - Template config for web UI
- `INFERENCE_GUIDE.md` - General inference guide
- `DQN_README.md` - DQN strategy overview
- `HOW_TO_USE.md` - Complete user guide

---

**Version**: 1.0  
**Last Updated**: November 11, 2025  
**Status**: ✅ Production Ready

