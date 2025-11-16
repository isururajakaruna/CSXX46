"""
RL-based Trading Strategy with DQN
This strategy uses Deep Q-Learning for making trading decisions.
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

from ats.state.simple_state import SimpleState
from ats.exchanges.data_classes.candle import Candle
from ats.exchanges.data_classes.order import Order, OrderStatus
from ats.strategies.base_strategy import BaseStrategy
from ats.utils.general import helpers as general_helpers
from ats.exceptions.general_exceptions import ConfigValidationException
from ats.utils.logging.logger import logger

# Try to import DQN agent (optional for non-training runs)
try:
    from .dqn_agent import DQNAgent

    DQN_AVAILABLE = True
except ImportError:
    logger.warning("DQN agent not available - RL features disabled")
    DQN_AVAILABLE = False


class Strategy(BaseStrategy):
    """
    RL-based Trading Strategy

    This strategy template is designed for Reinforcement Learning agents.

    Key Features:
    - State observation: price, volume, indicators, wallet balance
    - Action space: BUY (0), SELL (1), HOLD (2)
    - Reward calculation based on profit/loss
    - DQN agent for decision making
    - Episode management for training

    Configuration Parameters:
    - base_symbol: Base asset symbol (e.g., 'BTC')
    - quote_symbol: Quote asset symbol (e.g., 'USDT')
    - order_value_pct: Percentage of equity to use per order (default: 10%)
    - max_position_size: Maximum position size to hold
    - log_trade_activity: Whether to log trades (default: False)
    - use_indicators: Whether to include technical indicators in state (default: True)
    - ma_window: Moving average window size (default: 20)
    - rsi_window: RSI window size (default: 14)
    - training_mode: Whether to run in training mode (default: False)
    - model_path: Path to load trained model (optional)
    - epsilon: Exploration rate for action selection (default: 0.0)
    """

    # Action constants
    ACTION_BUY = 0
    ACTION_SELL = 1
    ACTION_HOLD = 2

    def __init__(self, config, state: SimpleState = None):
        super().__init__(config, state)

        # Debug: log received config keys
        logger.info(
            f"RL Strategy initializing with config keys: {list(self.config.keys())}"
        )
        logger.info(f"  training_mode: {self.config.get('training_mode', 'NOT SET')}")
        logger.info(f"  episode_id: {self.config.get('episode_id', 'NOT SET')}")

        # Configuration
        self.log_trade_activity = self.config.get("log_trade_activity", False)
        self.order_value_pct = self.config.get("order_value_pct", 10) / 100
        self.max_position_size = self.config.get("max_position_size", 1.0)
        self.use_indicators = self.config.get("use_indicators", True)

        # Indicator windows
        self.ma_window = self.config.get("ma_window", 20)
        self.rsi_window = self.config.get("rsi_window", 14)

        # Training configuration
        self.training_mode = self.config.get("training_mode", False)
        self.epsilon = self.config.get("epsilon", 0.0)  # Exploration rate
        self.model_path = self.config.get("model_path", None)
        
        # Decision interval: how often to make trading decisions (in candles)
        # Default: 1 (every candle), can be set to 5, 10, etc. for less frequent trading
        self.decision_interval = self.config.get("decision_interval", 1)
        self.last_decision_step = -1  # Track when last decision was made
        
        # Reward configuration - encourage HOLD behavior
        reward_config = self.config.get("reward_config", {})
        self.reward_type = reward_config.get("reward_type", "simple")  # simple, shaped, or sharpe
        self.penalty_trade = reward_config.get("penalty_trade", 0.0)  # Penalty for ANY trade
        self.penalty_hold = reward_config.get("penalty_hold", 0.0)  # Penalty for holding (usually 0 or negative to encourage)
        self.penalty_flip = reward_config.get("penalty_flip", 0.0)  # Penalty for BUY->SELL or SELL->BUY
        self.reward_scale = reward_config.get("reward_scale", 1.0)  # Scale factor for rewards
        
        logger.info(f"Reward config: type={self.reward_type}, penalty_trade={self.penalty_trade}, penalty_hold={self.penalty_hold}, penalty_flip={self.penalty_flip}")

        # Auto-enable training mode if episode_id is provided
        if "episode_id" in self.config:
            self.training_mode = True
            logger.info(f"Auto-enabled training_mode because episode_id is present")

        # Initialize indicators if enabled
        if self.use_indicators:
            self.ma_indicator = general_helpers.get_class_from_namespace(
                "indicators:mean_indicator"
            )({"N": self.ma_window, "is_weighted": False, "candle_length": 1})
            self.rsi_indicator = general_helpers.get_class_from_namespace(
                "indicators:rsi_indicator"
            )({"N": self.rsi_window, "candle_length": 1})

        # Strategy state variables
        self.current_candle = None
        self.previous_candle = None
        self.initial_price = None
        self.base_asset_balance = 0
        self.quote_asset_balance = 0
        self.current_price = 0

        # Indicators values
        self.ma = None
        self.rsi = None

        # Tracking variables
        self.open_orders = {}  # Track open orders by order_id
        self.trade_history = []  # Track completed trades
        self.start_time = None
        self.step_count = 0

        # RL specific variables
        self.rl_agent = None  # DQN agent
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.transitions = []  # Store transitions for training
        self.previous_state = None
        self.previous_action = None

        # State dimension: UPDATED from 7 to 13 (adding 6 Tier 1 features)
        # [0-6]: Original features (price, volume, ma, rsi, wallet_ratio, position_value, cash)
        # [7]: volume_imbalance (buy vs sell pressure)
        # [8-10]: return_1, return_5, return_20 (multi-scale momentum)
        # [11]: buy_ratio_recent (action history for trading control)
        # [12]: unrealized_pnl (position quality)
        self.state_dim = 13

        # NEW: Price history for calculating returns
        self.price_history = deque(maxlen=100)  # Keep last 100 prices for momentum

        # NEW: Action history for tracking recent behavior
        self.action_history = deque(maxlen=10)  # Keep last 10 actions

        # NEW: Track entry price for unrealized P&L calculation
        self.avg_entry_price = 0.0
        self.entry_count = 0

        # Transition export configuration
        transitions_dir = self.config.get("transitions_dir")
        if transitions_dir and str(transitions_dir).strip():
            # Use provided transitions directory
            self.export_dir = Path(str(transitions_dir).strip())
        else:
            # Fallback to local transitions directory
            self.export_dir = Path(__file__).parent / "transitions"

        # Ensure export directory exists with proper permissions
        try:
            self.export_dir.mkdir(parents=True, exist_ok=True)
            # Test if directory is writable
            test_file = self.export_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            logger.warning(
                f"Cannot write to {self.export_dir}: {e}. Using temp directory."
            )
            import tempfile

            self.export_dir = Path(tempfile.gettempdir()) / "ats_rl_transitions"
            self.export_dir.mkdir(parents=True, exist_ok=True)

        self.episode_id = self.config.get(
            "episode_id", datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )

        # Flag to track if transitions have been exported
        self._transitions_exported = False

        # Initialize DQN agent if available
        if DQN_AVAILABLE and (self.training_mode or self.model_path):
            self._initialize_dqn_agent()
        else:
            logger.info("Running without DQN agent (random actions)")

        # Initial equity for reward calculation
        self.initial_equity = None
        self.previous_equity = None

        logger.info(
            f"RL Strategy initialized (training_mode={self.training_mode}, epsilon={self.epsilon})"
        )

    def validate_config(self) -> None:
        """
        Validates the strategy-specific configuration.
        Raise an exception if any required config is missing or invalid.
        """
        required_properties = ["base_symbol", "quote_symbol"]

        for prop in required_properties:
            if prop not in self.config:
                raise ConfigValidationException(
                    "RL Strategy Config",
                    f"Missing required property '{prop}' in config.",
                )

        # Validate optional parameters
        if "order_value_pct" in self.config:
            if not 0 < self.config["order_value_pct"] <= 100:
                raise ConfigValidationException(
                    "RL Strategy Config", "order_value_pct must be between 0 and 100"
                )

        logger.info("RL Strategy config validated successfully")

    def _initialize_dqn_agent(self):
        """Initialize DQN agent."""
        try:
            self.rl_agent = DQNAgent(
                state_dim=self.state_dim,
                action_dim=3,  # Buy, Sell, Hold
                hidden_dims=[128, 64],
                learning_rate=0.001,
                gamma=0.99,
                epsilon_start=self.epsilon,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                buffer_capacity=100000,
                batch_size=64,
                target_update_freq=10,
                use_dueling=False,
                use_prioritized_replay=False,
                device="cpu",
            )

            # Load model if path provided
            if self.model_path and os.path.exists(self.model_path):
                self.rl_agent.load(self.model_path)
                logger.info(f"DQN model loaded from {self.model_path}")
            else:
                logger.info("DQN agent initialized with random weights")

            # Set epsilon for this episode
            self.rl_agent.epsilon = self.epsilon

        except Exception as e:
            logger.error(f"Failed to initialize DQN agent: {e}")
            self.rl_agent = None

    def on_candle(self, candle: Candle) -> None:
        """
        Main strategy logic - called for each new candle.
        This is where the RL agent observes the state and takes actions.

        Args:
            candle: The received Candle object
        """
        # Update candle tracking
        self.previous_candle = self.current_candle
        self.current_candle = candle
        self.current_price = candle.close
        self.step_count += 1

        # NEW: Update price history for momentum calculations
        self.price_history.append(candle.close)

        # Get wallet balances
        try:
            wallet_balance = self.exchange.get_wallet_balance()
            self.base_asset_balance = wallet_balance[self.config["base_symbol"]].free
            self.quote_asset_balance = wallet_balance[self.config["quote_symbol"]].free
        except Exception as e:
            logger.error(f"Error getting wallet balances: {e}")
            # Don't return - continue with initialization
            self.base_asset_balance = 0
            self.quote_asset_balance = 0

        # Initialize on first candle
        try:
            if self.start_time is None:
                self.start_time = candle.time
                self.initial_price = candle.close
                self.initial_equity = self._calculate_equity()
                self.previous_equity = self.initial_equity
                self._init_plots()
                # Debug: log total_steps configuration
                total_steps = self.config.get("total_steps")
                logger.info(
                    f"Strategy started at {self.start_time}, Initial equity: {self.initial_equity}, total_steps: {total_steps}, training_mode: {self.training_mode}"
                )

            # Update indicators
            if self.use_indicators:
                self._update_indicators(candle)

            # Get current state observation
            state = self._get_state_observation()

            # Store transition from previous step if available
            if (
                self.previous_state is not None
                and self.previous_action is not None
                and self.training_mode
            ):
                # Calculate reward for previous action
                reward = self._calculate_reward()
                self.current_episode_reward += reward

                # Check if episode is done (you can add custom termination conditions)
                done = False  # Set to True if episode should end

                # Store transition for training
                transition = (
                    self.previous_state,
                    self.previous_action,
                    reward,
                    state,
                    done,
                )
                self.transitions.append(transition)

                # Log every 50 transitions
                if len(self.transitions) % 50 == 1:
                    logger.info(
                        f"Collected {len(self.transitions)} transitions so far..."
                    )

                # Log first few transitions for debugging
                if len(self.transitions) <= 5:
                    logger.info(
                        f"Transition {len(self.transitions)}: action={self.previous_action}, reward={reward:.4f}, done={done}"
                    )

                # Optionally store in agent's replay buffer
                if self.rl_agent is not None:
                    self.rl_agent.store_transition(*transition)
            else:
                # Debug why no transition was stored
                if self.training_mode:
                    reasons = []
                    if self.previous_state is None:
                        reasons.append("no previous_state")
                    if self.previous_action is None:
                        reasons.append("no previous_action")
                    if reasons:
                        logger.debug(
                            f"No transition stored (step {self.step_count}): {', '.join(reasons)}"
                        )

            # RL Agent decision making - only at decision intervals
            should_make_decision = (self.step_count - self.last_decision_step) >= self.decision_interval
            
            if should_make_decision:
                # Make new trading decision
                action = self._get_rl_action(state)
                self.last_decision_step = self.step_count
                
                if self.log_trade_activity and action != self.ACTION_HOLD:
                    logger.info(f"Step {self.step_count}: DQN Decision -> {'BUY' if action == self.ACTION_BUY else 'SELL'} (interval: {self.decision_interval} candles)")
            else:
                # Between decision intervals, always HOLD
                action = self.ACTION_HOLD

            # Execute action
            self._execute_action(action, candle)

            # NEW: Update action history for trading frequency control
            self.action_history.append(action)

            # Store current state and action for next transition
            self.previous_state = state
            self.previous_action = action

            # Update order status
            self._update_open_orders()

            # Log metrics
            if self.log_trade_activity and self.step_count % 100 == 0:
                self._log_metrics()

            # Update plots
            self._update_plots(candle.time)

            # Debug step progress
            total_steps = self.config.get("total_steps")
            if total_steps is not None:
                logger.debug(
                    f"Step {self.step_count}/{total_steps} (training_mode={self.training_mode}, transitions={len(self.transitions)}, exported={self._transitions_exported})"
                )
            else:
                logger.debug(
                    f"Step {self.step_count} (no total_steps, training_mode={self.training_mode}, transitions={len(self.transitions)}, exported={self._transitions_exported})"
                )

            # Export transitions if this is the last step or near the end
            total_steps = self.config.get("total_steps")
            should_export = False

            # Check if we've reached or exceeded total steps
            if total_steps is not None and self.step_count >= total_steps:
                should_export = True
                logger.info(f"Reached total_steps ({total_steps}), triggering export")
            # Also export if we're very close to the end (handles off-by-one errors)
            elif total_steps is not None and self.step_count >= total_steps - 1:
                should_export = True
                logger.info(
                    f"Near end of episode (step {self.step_count}/{total_steps}), triggering export"
                )

            if (
                should_export
                and self.training_mode
                and len(self.transitions) > 0
                and not self._transitions_exported
            ):
                logger.info(
                    f"Exporting transitions from on_candle (step {self.step_count})"
                )
                final_equity = self._calculate_equity()
                total_return = (
                    (final_equity - self.initial_equity) / self.initial_equity * 100
                    if self.initial_equity and self.initial_equity > 0
                    else 0.0
                )
                self._export_transitions(final_equity, total_return)
                self._transitions_exported = True

        except Exception as e:
            logger.error(f"Error in on_candle processing: {e}")
            import traceback

            traceback.print_exc()

    def _init_plots(self):
        """Initialize plot topics"""
        try:
            if self.use_indicators:
                self.exchange.get_plot_data().set_topic(
                    topic="Moving Average",
                    color="blue",
                    lines_or_markers="lines",
                    pattern="solid",
                )
                self.exchange.get_plot_data().set_topic(
                    topic="RSI",
                    color="purple",
                    lines_or_markers="lines",
                    pattern="solid",
                )

            self.exchange.get_plot_data().set_topic(
                topic="Total Equity",
                color="green",
                lines_or_markers="lines",
                pattern="solid",
            )
            
            # Only add training-specific plots in training mode
            if self.training_mode:
                self.exchange.get_plot_data().set_topic(
                    topic="Episode Reward",
                    color="orange",
                    lines_or_markers="lines",
                    pattern="solid",
                )
        except Exception as e:
            logger.error(f"Error initializing plots: {e}")

    def _update_indicators(self, candle: Candle):
        """Update technical indicators"""
        self.ma_indicator.add(candle.close)
        self.rsi_indicator.add(candle.close)

        if self.ma_indicator.is_ready():
            self.ma = self.ma_indicator.result
        else:
            self.ma = candle.close

        if self.rsi_indicator.is_ready():
            self.rsi = self.rsi_indicator.result
        else:
            self.rsi = 50

    def _get_state_observation(self) -> np.ndarray:
        """
        Get the current state observation for the DQN agent.
        
        Enhanced State Vector (13 features):
        [0] price_norm: Normalized current price
        [1] volume_norm: Log-transformed volume
        [2] ma_norm: Moving average
        [3] rsi_norm: RSI indicator
        [4] wallet_ratio: Position ratio
        [5] position_value_norm: Position value
        [6] cash_norm: Available cash
        [7] volume_imbalance: Buy vs sell volume (NEW! Tier 1)
        [8] return_1: 1-step return (NEW! Tier 1)
        [9] return_5: 5-step return (NEW! Tier 1)
        [10] return_20: 20-step return (NEW! Tier 1)
        [11] buy_ratio_recent: Recent buy action ratio (NEW! Tier 1)
        [12] unrealized_pnl: Position profit/loss (NEW! Tier 1)
        
        Returns:
            np.ndarray: State vector of shape (13,)
        """
        # ============================================
        # SAFEGUARDS
        # ============================================
        if self.initial_price is None or self.initial_price == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        if self.initial_equity is None or self.initial_equity == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # ============================================
        # EXISTING FEATURES (0-6)
        # ============================================
        
        # Get current candle data
        volume = (
            (self.current_candle.buy_vol + self.current_candle.sell_vol)
            if self.current_candle
            else 0
        )
        
        # Get indicator values
        ma_value = self.ma if self.ma is not None else self.current_price
        rsi_value = self.rsi if self.rsi is not None else 50
        
        # [0] Price: Normalized by initial price
        price_norm = self.current_price / self.initial_price
        
        # [1] Volume: Log-transform
        volume_norm = np.log1p(volume)
        
        # [2] MA: Normalized by initial price
        ma_norm = ma_value / self.initial_price
        
        # [3] RSI: Scaled to [0, 1]
        rsi_norm = rsi_value / 100.0
        
        # Portfolio metrics
        current_equity = self._calculate_equity()
        position_value = self.base_asset_balance * self.current_price
        
        # [4] Wallet Ratio: [0, 1]
        wallet_ratio = position_value / current_equity if current_equity > 0 else 0
        
        # [5] Position Value: Normalized by initial equity
        position_value_norm = position_value / self.initial_equity
        
        # [6] Cash: Normalized by initial equity
        cash_norm = self.quote_asset_balance / self.initial_equity
        
        # ============================================
        # NEW TIER 1 FEATURES (7-12)
        # ============================================
        
        # [7] Volume Imbalance (BUY vs SELL) ⭐⭐⭐⭐⭐
        if self.current_candle:
            buy_vol = self.current_candle.buy_vol
            sell_vol = self.current_candle.sell_vol
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                volume_imbalance = (buy_vol - sell_vol) / total_vol  # Range: [-1, 1]
            else:
                volume_imbalance = 0.0
        else:
            volume_imbalance = 0.0
        
        # [8-10] Short-term Returns (Momentum) ⭐⭐⭐⭐⭐
        def safe_return(steps_back):
            """Calculate return over N steps back safely."""
            if len(self.price_history) > steps_back:
                old_price = list(self.price_history)[-steps_back - 1]
                if old_price > 0:
                    return (self.current_price - old_price) / old_price
            return 0.0
        
        return_1 = safe_return(1)    # 1-step (3 seconds)
        return_5 = safe_return(5)    # 5-step (15 seconds)
        return_20 = safe_return(20)  # 20-step (1 minute)
        
        # [11] Recent Action History (Trading Frequency) ⭐⭐⭐⭐
        if len(self.action_history) > 0:
            buy_ratio_recent = sum(1 for a in self.action_history if a == self.ACTION_BUY) / len(self.action_history)
        else:
            buy_ratio_recent = 0.0
        
        # [12] Unrealized P&L (Position Quality) ⭐⭐⭐⭐
        if self.base_asset_balance > 0 and self.avg_entry_price > 0:
            unrealized_pnl = (self.current_price - self.avg_entry_price) / self.avg_entry_price
        else:
            unrealized_pnl = 0.0
        
        # ============================================
        # ASSEMBLE STATE VECTOR
        # ============================================
        state = np.array(
            [
                # Existing features (0-6)
                price_norm,
                volume_norm,
                ma_norm,
                rsi_norm,
                wallet_ratio,
                position_value_norm,
                cash_norm,
                # New Tier 1 features (7-12)
                volume_imbalance,
                return_1,
                return_5,
                return_20,
                buy_ratio_recent,
                unrealized_pnl,
            ],
            dtype=np.float32,
        )
        
        # Safety: Replace any NaN or Inf values
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state

    def _get_rl_action(self, state: np.ndarray) -> int:
        """
        Get action from DQN agent.

        Args:
            state: Current state observation (numpy array)

        Returns:
            int: Action to take (0: BUY, 1: SELL, 2: HOLD)
        """
        if self.rl_agent is not None:
            # Use DQN agent to select action
            action = self.rl_agent.select_action(state, training=self.training_mode)
            
            # Log action distribution periodically
            if self.step_count % 100 == 0:
                action_names = {0: 'BUY', 1: 'SELL', 2: 'HOLD'}
                logger.debug(f"DQN selected: {action_names.get(action, 'UNKNOWN')} (epsilon={self.rl_agent.epsilon:.3f})")
        else:
            # Fallback: random action if no agent available
            action = np.random.randint(3)
            logger.warning("No DQN agent available, using random action")

        return action

    def _execute_action(self, action: int, candle: Candle):
        """
        Execute the action determined by the DQN agent.

        Args:
            action: Action to execute (0: BUY, 1: SELL, 2: HOLD)
            candle: Current candle data
        """
        if action == self.ACTION_HOLD:
            return

        current_equity = self._calculate_equity()
        order_value = current_equity * self.order_value_pct

        if action == self.ACTION_BUY:
            # Calculate order size
            order_size = order_value / self.current_price

            # Get minimum trading size from exchange config (default 0.0001)
            min_size = self.exchange.config.get("extra", {}).get(
                "min_trading_size", 0.0001
            )

            # Check if order size meets minimum and we have enough balance
            if order_size < min_size:
                # Order too small, skip
                return

            # Check if we have enough quote balance
            if self.quote_asset_balance >= order_value:
                # NEW: Update average entry price
                total_value = self.base_asset_balance * self.avg_entry_price + order_value
                total_size = self.base_asset_balance + order_size
                self.avg_entry_price = total_value / total_size if total_size > 0 else self.current_price
                self.entry_count += 1
                
                order = Order(
                    quote_symbol=self.config["quote_symbol"],
                    base_symbol=self.config["base_symbol"],
                    order_type="MARKET",
                    order_side="BUY",
                    size=order_size,
                    time=candle.time,
                )

                # Register callback to track order status
                order.on_status_change(self._on_order_status_change)

                # Submit order
                self.exchange.submit_order(order)
                self.open_orders[order.order_id] = order

                if self.log_trade_activity:
                    logger.info(
                        f"BUY order submitted: {order_size:.6f} {self.config['base_symbol']} "
                        f"at {self.current_price:.2f} (avg entry: {self.avg_entry_price:.2f})"
                    )

        elif action == self.ACTION_SELL:
            # Calculate order size
            order_size = min(order_value / self.current_price, self.base_asset_balance)

            # Get minimum trading size from exchange config (default 0.0001)
            min_size = self.exchange.config.get("extra", {}).get(
                "min_trading_size", 0.0001
            )

            # Check if order size meets minimum
            if order_size < min_size:
                # Order too small, skip
                return

            # Check if we have enough base balance
            if self.base_asset_balance >= order_size and order_size > 0:
                # NEW: Reset entry price if selling all
                if order_size >= self.base_asset_balance * 0.99:  # Selling 99%+
                    self.avg_entry_price = 0.0
                    self.entry_count = 0
                
                order = Order(
                    quote_symbol=self.config["quote_symbol"],
                    base_symbol=self.config["base_symbol"],
                    order_type="MARKET",
                    order_side="SELL",
                    size=order_size,
                    time=candle.time,
                )

                # Register callback to track order status
                order.on_status_change(self._on_order_status_change)

                # Submit order
                self.exchange.submit_order(order)
                self.open_orders[order.order_id] = order

                if self.log_trade_activity:
                    realized_pnl = (
                        (self.current_price - self.avg_entry_price) / self.avg_entry_price * 100
                        if self.avg_entry_price > 0
                        else 0.0
                    )
                    logger.info(
                        f"SELL order submitted: {order_size:.6f} {self.config['base_symbol']} "
                        f"at {self.current_price:.2f} (realized P&L: {realized_pnl:+.2f}%)"
                    )

    def _on_order_status_change(self, order: Order, modified_time):
        """
        Callback for order status changes.

        Args:
            order: The order that changed status
            modified_time: Time of the status change
        """
        if order.order_status == OrderStatus.FILLED:
            # Track completed trade
            self.trade_history.append(
                {
                    "order_id": order.order_id,
                    "side": order.order_side,
                    "size": order.size,
                    "price": order.price,
                    "time": modified_time,
                    "fees": order.fees,
                }
            )

            if self.log_trade_activity:
                logger.info(
                    f"Order FILLED: {order.order_side} {order.size:.6f} at {order.price:.2f}"
                )

        elif order.order_status == OrderStatus.REJECTED:
            if self.log_trade_activity:
                logger.warning(f"Order REJECTED: {order.order_side} {order.size:.6f}")

    def _update_open_orders(self):
        """Update and clean up open orders list"""
        completed_orders = []

        for order_id, order in self.open_orders.items():
            if order.order_status in [
                OrderStatus.FILLED,
                OrderStatus.REJECTED,
                OrderStatus.CANCELED,
            ]:
                completed_orders.append(order_id)

        for order_id in completed_orders:
            del self.open_orders[order_id]

    def _calculate_equity(self) -> float:
        """
        Calculate total equity in quote currency.

        Returns:
            float: Total equity value
        """
        return self.quote_asset_balance + (self.base_asset_balance * self.current_price)

    def _get_position_ratio(self) -> float:
        """
        Get the ratio of base asset to total equity.

        Returns:
            float: Position ratio (0 to 1)
        """
        equity = self._calculate_equity()
        if equity > 0:
            return (self.base_asset_balance * self.current_price) / equity
        return 0

    # def _calculate_reward(self) -> float:
    #     """
    #     Calculate reward for RL agent.

    #     TODO: Implement your reward function here.

    #     Returns:
    #         float: Reward value
    #     """
    #     current_equity = self._calculate_equity()

    #     # Simple reward: change in equity
    #     reward = current_equity - self.previous_equity

    #     # Trade penalty to reduce overtrading
    #     if self.previous_action != self.ACTION_HOLD:  # ✅ Correct
    #         reward -= 0.01  # Small penalty per trade (3/11/25 reduced from 0.5)

    #     # Reward shaping: boost positive rewards
    #     if reward > 0:
    #         reward *= 1.2  # 20% bonus for profitable moves

    #     # Update previous equity
    #     self.previous_equity = current_equity

    #     return reward

    def _calculate_reward(self) -> float:
        """
        Calculate reward based on reward_type configuration.
        
        Supports multiple reward strategies:
        - "simple": Pure equity change (default)
        - "shaped": Equity change + penalties for trading
        - "sharpe": Risk-adjusted returns
        
        Returns:
            float: Reward value
        """
        current_equity = self._calculate_equity()
        equity_change = current_equity - self.previous_equity
        
        if self.reward_type == "simple":
            # Simple: just equity change
            reward = equity_change
            
        elif self.reward_type == "shaped":
            # Shaped: equity change + penalties to encourage HOLD
            reward = equity_change
            
            # Penalty for any trade (encourages HOLD)
            if self.previous_action == self.ACTION_BUY or self.previous_action == self.ACTION_SELL:
                reward -= self.penalty_trade
            
            # Small penalty/bonus for holding
            if self.previous_action == self.ACTION_HOLD:
                reward -= self.penalty_hold  # Usually 0 or negative (becomes bonus)
            
            # Penalty for flipping position (BUY->SELL or SELL->BUY)
            if hasattr(self, 'action_before_previous'):
                if (self.action_before_previous == self.ACTION_BUY and self.previous_action == self.ACTION_SELL) or \
                   (self.action_before_previous == self.ACTION_SELL and self.previous_action == self.ACTION_BUY):
                    reward -= self.penalty_flip
            
        elif self.reward_type == "sharpe":
            # Risk-adjusted: penalize volatility
            reward = equity_change
            
            # Track equity history for volatility calculation
            if not hasattr(self, 'equity_history'):
                self.equity_history = deque(maxlen=20)
            self.equity_history.append(current_equity)
            
            # Calculate Sharpe-like penalty for high volatility
            if len(self.equity_history) >= 10:
                equity_std = np.std(self.equity_history)
                if equity_std > 0:
                    reward = reward / (1 + equity_std)  # Penalize high volatility
            
            # Still apply trade penalty
            if self.previous_action != self.ACTION_HOLD:
                reward -= self.penalty_trade
        
        else:
            # Fallback to simple
            reward = equity_change
        
        # Apply reward scaling
        reward *= self.reward_scale
        
        # Track previous action for next reward calculation
        self.action_before_previous = self.previous_action
        
        # Update previous equity
        self.previous_equity = current_equity
        
        return reward

    def _log_metrics(self):
        """Log current metrics"""
        current_equity = self._calculate_equity()
        equity_change = (
            (current_equity - self.initial_equity) / self.initial_equity
        ) * 100

        logger.info(
            f"Step: {self.step_count} | "
            f"Price: {self.current_price:.2f} | "
            f"Equity: {current_equity:.2f} ({equity_change:+.2f}%) | "
            f"Position: {self._get_position_ratio():.2%} | "
            f"Open Orders: {len(self.open_orders)} | "
            f"Trades: {len(self.trade_history)}"
        )

    def _update_plots(self, time):
        """Update plot data"""
        current_equity = self._calculate_equity()

        if self.use_indicators and self.ma is not None:
            self.exchange.get_plot_data().add(
                topic="Moving Average", time=time, num=self.ma
            )
            self.exchange.get_plot_data().add(
                topic="RSI", time=time, num=self.rsi if self.rsi is not None else 50
            )

        self.exchange.get_plot_data().add(
            topic="Total Equity", time=time, num=current_equity
        )

        # Only update training-specific plots in training mode
        if self.training_mode:
            self.exchange.get_plot_data().add(
                topic="Episode Reward", time=time, num=self.current_episode_reward
            )

    def on_stop(self) -> None:
        """
        Called when the strategy is stopped.
        Use this to save the RL model, states, or clean up resources.
        """
        logger.info("🚨 on_stop() CALLED! Starting export...")
        try:
            final_equity = self._calculate_equity()

            # Safely calculate total return (handle case where initial_equity is 0 or None)
            if self.initial_equity and self.initial_equity > 0:
                total_return = (
                    (final_equity - self.initial_equity) / self.initial_equity
                ) * 100
            else:
                total_return = 0.0
                logger.warning(
                    f"initial_equity is {self.initial_equity}, cannot calculate return percentage"
                )

            logger.info("=" * 60)
            logger.info("RL Strategy Stopped - Final Summary")
            logger.info("=" * 60)
            logger.info(
                f"Initial Equity: {self.initial_equity if self.initial_equity else 'NOT SET'}"
            )
            logger.info(f"Final Equity: {final_equity:.2f}")
            logger.info(f"Total Return: {total_return:+.2f}%")
            logger.info(f"Total Steps: {self.step_count}")
            logger.info(f"Total Trades: {len(self.trade_history)}")
            logger.info(f"Total Episode Reward: {self.current_episode_reward:.2f}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error in on_stop summary calculation: {e}")
            final_equity = 0.0
            total_return = 0.0

        # ALWAYS try to export transitions, even if summary calculation failed
        try:
            logger.info(
                f"on_stop called: training_mode={self.training_mode}, transitions={len(self.transitions)}, exported={self._transitions_exported}, step_count={self.step_count}"
            )
            if (
                self.training_mode
                and len(self.transitions) > 0
                and not self._transitions_exported
            ):
                logger.info(f"Exporting transitions from on_stop...")
                self._export_transitions(final_equity, total_return)
                self._transitions_exported = True
            elif self.training_mode and self._transitions_exported:
                logger.info(f"Transitions already exported, skipping in on_stop")
            elif self.training_mode:
                logger.warning(f"Training mode enabled but no transitions collected!")
        except Exception as e:
            logger.error(f"Error exporting transitions: {e}")
            import traceback

            logger.error(traceback.format_exc())

    def _export_transitions(self, final_equity: float, total_return: float):
        """
        Export collected transitions to a JSON file for training.

        Args:
            final_equity: Final portfolio equity
            total_return: Total return percentage
        """
        try:
            logger.info(f"_export_transitions called")
            logger.info(f"  export_dir: {self.export_dir}")
            logger.info(f"  episode_id: {self.episode_id}")
            logger.info(f"  num_transitions: {len(self.transitions)}")

            # Ensure export directory exists
            self.export_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"  export_dir exists: {self.export_dir.exists()}")

            export_file = self.export_dir / f"episode_{self.episode_id}.json"
            logger.info(f"  export_file: {export_file}")

            # Convert transitions to serializable format
            transitions_data = []
            for idx, (state, action, reward, next_state, done) in enumerate(
                self.transitions
            ):
                try:
                    # Convert numpy arrays and handle NaN/Inf values
                    def safe_convert(arr):
                        """Convert numpy array to list, replacing NaN/Inf with 0"""
                        if isinstance(arr, np.ndarray):
                            arr = arr.copy()
                            arr[np.isnan(arr)] = 0.0
                            arr[np.isinf(arr)] = 0.0
                            return arr.tolist()
                        else:
                            return [
                                (
                                    0.0
                                    if (
                                        isinstance(x, float)
                                        and (np.isnan(x) or np.isinf(x))
                                    )
                                    else x
                                )
                                for x in list(arr)
                            ]

                    # Safe convert reward
                    safe_reward = float(reward)
                    if np.isnan(safe_reward) or np.isinf(safe_reward):
                        safe_reward = 0.0

                    transitions_data.append(
                        {
                            "state": safe_convert(state),
                            "action": int(action),
                            "reward": safe_reward,
                            "next_state": safe_convert(next_state),
                            "done": bool(done),
                        }
                    )
                except Exception as e:
                    logger.error(f"Error converting transition {idx}: {e}")
                    raise

            logger.info(
                f"  Converted {len(transitions_data)} transitions to JSON format"
            )

            # Helper to safely convert floats
            def safe_float(val):
                """Convert to float, replacing NaN/Inf with 0"""
                f = float(val) if val is not None else 0.0
                return 0.0 if (np.isnan(f) or np.isinf(f)) else f

            # Prepare episode summary
            episode_data = {
                "episode_id": self.episode_id,
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "initial_equity": safe_float(self.initial_equity),
                    "final_equity": safe_float(final_equity),
                    "total_return": safe_float(total_return),
                    "total_reward": safe_float(self.current_episode_reward),
                    "num_steps": int(self.step_count),
                    "num_trades": len(self.trade_history),
                    "num_transitions": len(self.transitions),
                },
                "transitions": transitions_data,
            }

            # Write to temp file first, then rename atomically to prevent race conditions
            temp_file = export_file.with_suffix(".json.tmp")
            logger.info(f"  Writing to temp file: {temp_file}")

            # Ensure parent directory exists
            export_file.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(temp_file, "w") as f:
                    json.dump(episode_data, f, indent=2)

                # Atomic rename
                logger.info(f"  Renaming to: {export_file}")
                if export_file.exists():
                    export_file.unlink()  # Remove existing file if any
                temp_file.rename(export_file)

                logger.info(f"✓ Successfully created transition file: {export_file}")
                logger.info(f"  File size: {export_file.stat().st_size} bytes")

            except Exception as write_error:
                logger.error(f"Failed to write transition file: {write_error}")
                # Try writing directly without temp file as fallback
                try:
                    with open(export_file, "w") as f:
                        json.dump(episode_data, f, indent=2)
                    logger.info(
                        f"✓ Created transition file (direct write): {export_file}"
                    )
                except Exception as direct_write_error:
                    logger.error(f"Direct write also failed: {direct_write_error}")
                    raise

            logger.info(
                f"✓ Successfully exported {len(self.transitions)} transitions to {export_file}"
            )
            logger.info(f"  File size: {export_file.stat().st_size} bytes")

        except Exception as e:
            logger.error(f"Failed to export transitions: {e}")
            logger.error(f"  Exception type: {type(e).__name__}")
            logger.error(f"  Exception args: {e.args}")
            import traceback

            logger.error(f"  Traceback:\n{traceback.format_exc()}")
