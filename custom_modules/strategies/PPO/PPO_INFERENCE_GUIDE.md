# 🎯 PPO Inference Guide - Web Interface

Complete guide to running trained PPO models for trading inference via ATS web portal.

---

## 📋 **Quick Start**

### **Step 1: Train a PPO Model**

```bash
cd custom_modules/custom_scripts
./train_interactive_PPO.sh
```

Your model will be saved to:
```
custom_modules/strategies/PPO/saved_models/run_YYYYMMDD_HHMMSS/ppo_final.pth
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
        "wallet": {
          "assets": {
            "BTC": 0,
            "USDT": 10000
          }
        }
      },
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
      "model_path": "custom_modules/strategies/PPO/saved_models/run_20251112_012434/ppo_final.pth",
      "decision_interval": 5,
      
      "order_value_pct": 10,
      "max_position_size": 1.0,
      "log_trade_activity": true,
      
      "use_indicators": true,
      "ma_window": 20,
      "rsi_window": 14
    },
    "namespace": "strategies:PPO"
  }
}
```

**Key Parameters to Update:**
- `model_path`: Your trained model (relative path to `.pth` file)
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

### **Strategy Section - PPO Inference**

```json
"strategy": {
  "config": {
    // Required for inference
    "training_mode": false,                    // ← MUST be false
    "model_path": "custom_modules/strategies/PPO/saved_models/run_20251112_012434/ppo_final.pth",  // ← Your model
    "decision_interval": 5,                    // ← Make decision every 5 candles
    
    // Trading parameters
    "order_value_pct": 10,                     // ← % of equity per trade
    "max_position_size": 1.0,                  // ← Max position size
    "log_trade_activity": true,                // ← Log each trade
    
    // Indicators (must match training)
    "use_indicators": true,
    "ma_window": 20,
    "rsi_window": 14
  },
  "namespace": "strategies:PPO"                 // ← Points to custom_modules/strategies/PPO
}
```

---

## ⏱️ **Decision Interval**

The `decision_interval` parameter controls **how often** the PPO agent makes trading decisions:

- `decision_interval: 1` → Make decision **every candle** (most frequent)
- `decision_interval: 5` → Make decision **every 5 candles**
- `decision_interval: 10` → Make decision **every 10 candles** (less frequent)

### **Why use it?**

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

---

## 📁 **File Paths**

### **Model Path (Relative)**

✅ **Correct:**
```json
"model_path": "custom_modules/strategies/PPO/saved_models/run_20251112_012434/ppo_final.pth"
```

❌ **Wrong:**
```json
"model_path": "/Users/username/ats/custom_modules/..."    // Absolute paths won't work
"model_path": "saved_models/run_20251112_012434/ppo_final.pth"  // Missing custom_modules/strategies/PPO
```

### **Data Source (Relative)**

✅ **Correct:**
```json
"data_source": "ats/data/BTC_USDT_short.csv"
"data_source": "custom_modules/data/my_data.csv"
```

---

## 🎯 **Configuration Examples**

### **Example 1: Conservative Trading**

```json
{
  "strategy": {
    "config": {
      "training_mode": false,
      "model_path": "custom_modules/strategies/PPO/saved_models/run_20251112_012434/ppo_final.pth",
      "decision_interval": 10,     // ← Less frequent decisions
      
      "order_value_pct": 5,        // ← Small positions (5%)
      "max_position_size": 0.5,    // ← Max 50% in market
      "log_trade_activity": true
    },
    "namespace": "strategies:PPO"
  }
}
```

### **Example 2: Aggressive Trading**

```json
{
  "strategy": {
    "config": {
      "training_mode": false,
      "model_path": "custom_modules/strategies/PPO/saved_models/run_20251112_012434/ppo_final.pth",
      "decision_interval": 3,      // ← More frequent decisions
      
      "order_value_pct": 20,       // ← Large positions (20%)
      "max_position_size": 1.0,    // ← Max 100% in market
      "log_trade_activity": true
    },
    "namespace": "strategies:PPO"
  }
}
```

---

## 🐛 **Troubleshooting**

### **"Module 'PPO' not found"**

**Problem**: Namespace is incorrect

**Solution**: Ensure namespace is exactly `"strategies:PPO"` (case-sensitive)

### **"Model file not found"**

**Problem**: Model path is incorrect or absolute

**Solutions:**
1. List available models:
   ```bash
   ls -lh custom_modules/strategies/PPO/saved_models/*/ppo_final.pth
   ```
2. Use **relative path** from project root
3. Check model actually exists

### **"Invalid strategy configuration"**

**Problem**: Missing required parameters

**Solution**: Ensure these are present:
- `training_mode: false`
- `model_path: "..."`
- `base_symbol` and `quote_symbol`

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
```

---

## 🎓 **Best Practices**

### **✅ DO:**
- Set `training_mode: false` for inference
- Use **relative paths** (not absolute)
- Use `ppo_final.pth` (fully trained)
- Test on **unseen data**
- Match indicator configs with training
- Start with `decision_interval: 5`

### **❌ DON'T:**
- Use `training_mode: true` (will try to collect transitions)
- Use absolute paths with usernames
- Use early checkpoints for production
- Test on same data used for training
- Change indicator windows from training
- Use `decision_interval: 1` in production (too frequent)

---

## ✅ **Pre-Flight Checklist**

Before submitting job:

- [ ] ATS server running (`./start.sh`)
- [ ] Model trained and path correct
- [ ] `training_mode` set to `false`
- [ ] `namespace` is `"strategies:PPO"`
- [ ] Data source exists
- [ ] Wallet balance configured
- [ ] Paths are **relative** (not absolute)
- [ ] `decision_interval` configured appropriately

---

## 🔗 **Related Files**

- `inference_config_web.json` - Template config for web UI
- `../DQN/WEB_INFERENCE_GUIDE.md` - Similar guide for DQN
- `../DQN/REWARD_STRATEGIES_GUIDE.md` - Reward shaping (applicable to both)

---

## 📈 **PPO vs DQN**

| Feature | PPO | DQN |
|---------|-----|-----|
| **Algorithm** | Policy Gradient | Value-based |
| **Training** | Iterations (rollouts) | Episodes |
| **Stability** | More stable | Can be unstable |
| **Sample Efficiency** | Less efficient | More efficient |
| **Continuous Actions** | Yes | No (discrete only) |
| **Our Actions** | BUY/SELL/HOLD | BUY/SELL/HOLD |

**Both support:**
- `decision_interval` for controlled trading frequency
- `model_path` for inference
- Same state/action space
- Same indicator configuration

---

**Version**: 1.0  
**Last Updated**: November 12, 2025  
**Status**: ✅ Production Ready

