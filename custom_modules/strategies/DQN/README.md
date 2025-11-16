# RL Trading Strategy

This is a Reinforcement Learning-based trading strategy template for the ATS system.

## Overview

This strategy provides a foundation for implementing RL agents (DQN, PPO, A3C, etc.) for cryptocurrency trading.

## Features

- **State Observation**: Provides structured state observations including:
  - Current price and price changes
  - Wallet balances (base and quote assets)
  - Total equity and position ratios
  - Technical indicators (MA, RSI)
  - Volume changes

- **Action Space**: Three discrete actions:
  - `BUY`: Purchase base asset
  - `SELL`: Sell base asset
  - `HOLD`: Do nothing

- **Reward Calculation**: Simple equity-based reward (customizable)

- **Order Management**: Automatic tracking of orders and trades

- **Plotting**: Real-time visualization of equity and rewards

## Configuration Parameters

```yaml
strategy:
  namespace: 'strategies:DQN'
  config:
    base_symbol: 'BTC'              # Base asset symbol
    quote_symbol: 'USDT'            # Quote asset symbol
    order_value_pct: 10             # Percentage of equity per trade (0-100)
    max_position_size: 1.0          # Maximum position size
    log_trade_activity: true        # Enable trade logging
    use_indicators: true            # Include technical indicators
    ma_window: 20                   # Moving average window size
    rsi_window: 14                  # RSI window size
```

## Implementation Guide

### 1. Initialize Your RL Agent

In the `__init__` method, replace the placeholder:

```python
# Import your RL agent
from your_rl_library import RLAgent

self.rl_agent = RLAgent(
    state_size=len(self._get_state_observation()),
    action_size=3,  # BUY, SELL, HOLD
    config=your_config
)
```

### 2. Implement Action Selection

In the `_get_rl_action` method:

```python
def _get_rl_action(self, state: dict) -> str:
    # Convert state dict to numpy array or tensor
    state_array = self._state_to_array(state)
    
    # Get action from agent
    action_idx = self.rl_agent.predict(state_array)
    
    # Map action index to action string
    actions = ['BUY', 'SELL', 'HOLD']
    return actions[action_idx]
```

### 3. Customize Reward Function

In the `_calculate_reward` method:

```python
def _calculate_reward(self) -> float:
    current_equity = self._calculate_equity()
    
    # Example: Sharpe ratio based reward
    returns = (current_equity - self.previous_equity) / self.previous_equity
    
    # Add penalty for excessive trading
    trading_penalty = -0.001 * len(self.open_orders)
    
    reward = returns + trading_penalty
    
    self.previous_equity = current_equity
    return reward
```

### 4. Save/Load Model

In the `on_stop` method:

```python
def on_stop(self) -> None:
    # Save the trained model
    self.rl_agent.save('models/rl_agent_checkpoint.pkl')
    
    # Save training metrics
    import json
    with open('metrics/training_metrics.json', 'w') as f:
        json.dump({
            'total_return': total_return,
            'total_trades': len(self.trade_history),
            'episode_rewards': self.episode_rewards
        }, f)
```

## State Observation Structure

The default state observation includes:

```python
{
    'price': float,                 # Current close price
    'base_balance': float,          # Available base asset
    'quote_balance': float,         # Available quote asset
    'equity': float,                # Total equity in quote currency
    'position_ratio': float,        # Base asset ratio (0-1)
    'open_orders_count': int,       # Number of open orders
    'ma': float,                    # Moving average (if enabled)
    'rsi': float,                   # RSI indicator (if enabled)
    'price_to_ma_ratio': float,     # Price relative to MA
    'price_change': float,          # Price change from previous candle
    'volume_change': float,         # Volume change from previous candle
}
```

## Example: Integrating with Stable Baselines3

```python
from stable_baselines3 import PPO
import numpy as np

class Strategy(BaseStrategy):
    def __init__(self, config, state: SimpleState = None):
        super().__init__(config, state)
        
        # Load or create PPO agent
        self.rl_agent = PPO.load("ppo_trading_agent") if os.path.exists("ppo_trading_agent.zip") else None
        self.state_history = []
    
    def _get_rl_action(self, state: dict) -> str:
        if self.rl_agent is None:
            return 'HOLD'
        
        # Convert state to numpy array
        state_array = np.array([
            state['price'],
            state['base_balance'],
            state['quote_balance'],
            state['equity'],
            state['position_ratio'],
            state.get('ma', 0),
            state.get('rsi', 50),
            state.get('price_change', 0)
        ])
        
        # Get action from agent
        action_idx, _ = self.rl_agent.predict(state_array, deterministic=True)
        
        actions = ['BUY', 'SELL', 'HOLD']
        return actions[action_idx]
```

## Testing

To test the strategy with backtesting:

```python
import requests

# Create trading job
config = {
    "exchange": {
        "namespace": "exchanges:back_trading",
        "config": {
            "symbol": {"base": "BTC", "quote": "USDT"},
            "plot_data_max_len": -1,
            "extra": {
                "data_source": "ats/data/BTC_USDT_short.csv",
                "wallet": {"assets": {"USDT": 10000, "BTC": 0}},
                "min_trading_size": 0.0001,
                "fees": {
                    "namespace": "fees:generic",
                    "config": {
                        "limit": {"buy": {"base": 0.001, "quote": 0}, "sell": {"base": 0, "quote": 0.001}},
                        "market": {"buy": {"base": 0.001, "quote": 0}, "sell": {"base": 0, "quote": 0.001}}
                    }
                }
            }
        }
    },
    "strategy": {
        "namespace": "strategies:DQN",
        "config": {
            "base_symbol": "BTC",
            "quote_symbol": "USDT",
            "order_value_pct": 10,
            "log_trade_activity": True,
            "use_indicators": True,
            "ma_window": 20,
            "rsi_window": 14
        }
    }
}

# Submit to ATS
response = requests.post("http://localhost:5010/trading_job/create", json=config)
job_id = response.json()['job_id']

# Run the job
requests.get(f"http://localhost:5010/trading_job/run/{job_id}")
```

## Next Steps

1. **Implement your RL agent** in the `_get_rl_action` method
2. **Customize the reward function** to match your trading objectives
3. **Add training logic** if you want to train during backtesting
4. **Experiment with state features** by modifying `_get_state_observation`
5. **Add risk management** (stop-loss, position limits, etc.)

## Notes

- The template uses MARKET orders by default. Modify `_execute_action` for LIMIT orders.
- State normalization is important for RL - consider adding it to `_get_state_observation`.
- For training, you may want to run multiple episodes and collect experience replays.
- Consider adding epsilon-greedy exploration during training.

