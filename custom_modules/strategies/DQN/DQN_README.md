# DQN Trading Agent - Complete Guide

This directory contains a complete Deep Q-Network (DQN) implementation for automated trading using reinforcement learning.

## 📁 Files Overview

### Core DQN Components

1. **`dqn_model.py`** - Neural network architectures
   - `DQN`: Standard Deep Q-Network
   - `DuelingDQN`: Advanced dueling architecture (optional)
   
2. **`replay_buffer.py`** - Experience replay buffers
   - `ReplayBuffer`: Standard uniform sampling
   - `PrioritizedReplayBuffer`: Priority-based sampling (advanced)
   
3. **`dqn_agent.py`** - Main DQN agent class
   - Action selection (ε-greedy)
   - Training loop
   - Model saving/loading
   
4. **`strategy.py`** - Trading strategy with DQN integration
   - State extraction from market data
   - Action execution (BUY/SELL/HOLD)
   - Reward calculation
   - Transition storage

5. **`train_dqn.py`** - Training script
   - Episode management via API
   - Training loop
   - Model checkpointing
   - Progress tracking

6. **`training_config.yaml`** - Configuration file
   - All hyperparameters
   - Trading parameters
   - Paths and settings

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Activate conda environment
conda activate ats

# PyTorch should already be installed. If not:
pip install torch torchvision torchaudio
```

### 2. Run a Test

Test individual components:

```bash
cd custom_modules/strategies/DQN

# Test DQN model
python dqn_model.py

# Test replay buffer
python replay_buffer.py

# Test DQN agent
python dqn_agent.py
```

### 3. Start Training

```bash
# From the ats root directory
cd /path/to/ats  # Your ATS installation directory

# Make sure ATS server is running
# In another terminal:
# conda activate ats
# python ats/main.py

# Run training
python custom_modules/strategies/DQN/train_dqn.py --episodes 100 --save_dir models/dqn

# With custom data
python custom_modules/strategies/DQN/train_dqn.py \
  --episodes 100 \
  --data /path/to/your/data.csv \
  --save_dir models/dqn \
  --save_freq 10
```

### 4. Use Trained Model

After training, use the model for trading:

```python
# In your job configuration
{
  "strategy": {
    "namespace": "strategies:DQN",
    "config": {
      "base_symbol": "BTC",
      "quote_symbol": "USDT",
      "training_mode": false,              # Evaluation mode
      "model_path": "models/dqn/dqn_final.pth",  # Path to trained model
      "epsilon": 0.0,                      # Greedy policy (no exploration)
      "use_indicators": true,
      "ma_window": 20,
      "rsi_window": 14,
      "order_value_pct": 10
    }
  }
}
```

---

## 📊 State Space

The DQN agent observes a **7-dimensional state vector**:

| Index | Feature | Description | Normalization |
|-------|---------|-------------|---------------|
| 0 | Price | Current BTC price | Normalized by initial price |
| 1 | Volume | Trading volume | Scaled (/1000) |
| 2 | Moving Average | 20-period MA | Normalized by initial price |
| 3 | RSI | Relative Strength Index | Scaled to 0-1 (original: 0-100) |
| 4 | Wallet Ratio | BTC value / Total equity | Already 0-1 |
| 5 | Position Value | Value of BTC holdings | Normalized by initial equity |
| 6 | Cash | Available USDT | Normalized by initial equity |

**State Vector**: `[price_norm, volume_norm, ma_norm, rsi_norm, wallet_ratio, position_value_norm, cash_norm]`

---

## 🎮 Action Space

The agent can choose from **3 discrete actions**:

| Action | Value | Description |
|--------|-------|-------------|
| BUY | 0 | Buy BTC using a percentage of available cash |
| SELL | 1 | Sell a percentage of BTC holdings |
| HOLD | 2 | Do nothing |

**Order Size**: Configured by `order_value_pct` (default: 10% of equity)

---

## 💰 Reward Function

**Default**: Equity change per step

```python
reward = current_equity - previous_equity
```

**Positive reward**: Equity increased → good action  
**Negative reward**: Equity decreased → bad action

### Customization

You can modify the reward function in `strategy.py`:

```python
def _calculate_reward(self) -> float:
    current_equity = self._calculate_equity()
    reward = current_equity - self.previous_equity
    
    # Add custom reward shaping here
    # Example: Penalize for too many trades
    # if action != HOLD:
    #     reward -= 0.01  # Small penalty for trading
    
    self.previous_equity = current_equity
    return reward
```

---

## 🧠 DQN Architecture

### Standard DQN

```
Input (7) → FC(128) → ReLU → Dropout(0.2) → 
FC(64) → ReLU → Dropout(0.2) → 
FC(3) → Q-values
```

### Dueling DQN (Optional)

```
Input (7) → Shared Features(128) → 
    ├─ Value Stream → V(s)
    └─ Advantage Stream → A(s,a)
    
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

To use Dueling DQN, set `use_dueling: true` in config.

---

## 🔧 Key Hyperparameters

### Learning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor (future reward weight) |
| `batch_size` | 64 | Training batch size |

### Exploration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epsilon_start` | 1.0 | Initial exploration rate (100% random) |
| `epsilon_end` | 0.05 | Final exploration rate (5% random) |
| `epsilon_decay` | 0.995 | Decay rate per episode |

**Epsilon Schedule**: ε = max(ε_end, ε × decay) after each episode

### Buffer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_capacity` | 100,000 | Maximum transitions to store |
| `min_buffer_size` | 1,000 | Min size before training starts |

---

## 📈 Training Process

### How Training Works

1. **Episode Initialization**
   - Create trading job via API with current ε
   - Job runs backtest on historical data
   
2. **Trading Loop** (per candle)
   - Extract state from market data
   - Select action using ε-greedy policy
   - Execute trade (if BUY/SELL)
   - Calculate reward (equity change)
   - Store transition (s, a, r, s', done)
   
3. **Episode Completion**
   - Collect all transitions
   - Train DQN on random batches
   - Update target network (every N episodes)
   - Decay ε
   - Save checkpoint
   
4. **Repeat** for N episodes

### Training Tips

**1. Start with High Exploration**
- Begin with ε=1.0 to explore state space
- Gradually decay to exploit learned policy

**2. Monitor Loss**
- Loss should decrease over episodes
- If loss plateaus, consider:
  - Adjusting learning rate
  - Changing reward function
  - Adding more features to state

**3. Check Reward Trend**
- Average reward should increase
- If stuck, try:
  - Different hyperparameters
  - Reward shaping
  - Longer training

**4. Use Evaluation**
- Periodically test with ε=0 (greedy)
- Compare to baseline (random, buy-and-hold)

---

## 📝 Configuration File

Edit `training_config.yaml` to customize training:

```yaml
# Example: Fast training for testing
training:
  num_episodes: 20
  save_freq: 5

agent:
  learning_rate: 0.01  # Higher LR for faster learning
  epsilon_decay: 0.95  # Faster decay
  batch_size: 32       # Smaller batches

# Example: Production training
training:
  num_episodes: 500
  save_freq: 25

agent:
  learning_rate: 0.0001  # Slower, more stable
  epsilon_decay: 0.999   # Very gradual
  batch_size: 128        # Larger batches
  use_dueling: true      # Advanced architecture
  use_prioritized_replay: true
```

---

## 📂 Output Files

After training, you'll find:

```
models/dqn/
├── dqn_episode_10.pth           # Checkpoint at episode 10
├── dqn_episode_20.pth           # Checkpoint at episode 20
├── ...
├── dqn_final.pth                # Final trained model
├── training_stats_ep10.json     # Training statistics
├── training_stats_ep20.json
└── ...

logs/dqn/
└── training.log                 # Detailed training logs

results/dqn/
├── rewards.csv                  # Episode rewards
├── losses.csv                   # Training losses
└── metrics.csv                  # Other metrics
```

---

## 🔬 Advanced Features

### 1. Prioritized Experience Replay

Prioritizes transitions with higher TD error (more "surprising"):

```python
# In dqn_agent.py
agent = DQNAgent(
    ...
    use_prioritized_replay=True
)
```

### 2. Dueling DQN

Separates state value from action advantage:

```python
agent = DQNAgent(
    ...
    use_dueling=True
)
```

### 3. Custom Indicators

Add more indicators to state:

```python
# In strategy.py __init__
self.bollinger = get_indicator("bollinger_bands")({'N': 20, 'K': 2})

# In _get_state_observation
state = np.array([
    price_norm,
    volume_norm,
    ma_norm,
    rsi_norm,
    self.bollinger.upper_band,  # New
    self.bollinger.lower_band,  # New
    wallet_ratio,
    position_value_norm,
    cash_norm
])
```

Remember to update `state_dim` accordingly!

### 4. Multi-Asset Trading

Extend to trade multiple assets:

```python
# State: [asset1_price, asset1_volume, ..., asset2_price, asset2_volume, ...]
# Actions: [asset1_action, asset2_action, ...]
```

---

## 🐛 Troubleshooting

### Issue: Loss is NaN or Inf

**Cause**: Gradient explosion or numerical instability

**Fix**:
- Reduce learning rate
- Check state normalization
- Add gradient clipping (already implemented)
- Check for division by zero in reward

### Issue: Agent only HOLDs

**Cause**: 
- Learned that trading is penalized (fees too high)
- Insufficient exploration

**Fix**:
- Increase `epsilon_start` or slow down `epsilon_decay`
- Reduce trading fees in config
- Add reward shaping for trade diversity

### Issue: Training very slow

**Cause**: API overhead, large dataset

**Fix**:
- Use smaller dataset for initial testing
- Reduce `num_episodes`
- Increase `batch_size` (if memory allows)
- Train on multiple shorter episodes

### Issue: Model doesn't improve

**Cause**: Poor reward signal, insufficient data

**Fix**:
- Check reward function (print rewards)
- Ensure sufficient buffer size
- Try different hyperparameters
- Visualize Q-values over time

---

## 📚 Further Reading

### DQN Papers

1. **Original DQN**: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
2. **Double DQN**: "Deep Reinforcement Learning with Double Q-learning" (van Hasselt et al., 2015)
3. **Dueling DQN**: "Dueling Network Architectures for Deep Reinforcement Learning" (Wang et al., 2016)
4. **Prioritized Experience Replay**: "Prioritized Experience Replay" (Schaul et al., 2015)

### RL for Trading

- "Deep Reinforcement Learning for Trading" (Zhang et al., 2019)
- "Practical Deep Reinforcement Learning Approach for Stock Trading" (Xiong et al., 2018)

---

## 🤝 Contributing

To improve the DQN implementation:

1. Test new reward functions
2. Add more state features (order book, market sentiment)
3. Implement advanced algorithms (Rainbow DQN, C51, IQN)
4. Add multi-asset support
5. Improve training stability

---

## ⚠️ Disclaimer

**This is for educational and research purposes only.**

- Trading with real money involves significant risk
- Past performance doesn't guarantee future results
- Always test thoroughly before live trading
- Consider transaction costs and slippage
- RL agents can behave unpredictably

---

## 📞 Support

For issues or questions:

1. Check this README
2. Review code comments in source files
3. Check ATS main documentation
4. Test components individually
5. Print debug information (states, rewards, Q-values)

---

**Happy Training! 🚀**

