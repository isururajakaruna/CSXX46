# 🏦 How to Implement Custom Exchange Connectors

This directory is for implementing custom exchange connectors to integrate new trading platforms with ATS.

---

## 📚 **What Goes Here**

Custom exchange implementations for:
- Live trading exchanges (Binance, Coinbase, Kraken, etc.)
- Backtesting engines
- Paper trading simulators
- Custom data feeds

---

## 🎯 **File Structure**

```
custom_modules/exchanges/
├── HOW_TO_IMPLEMENT.md          ← This file
├── my_exchange/
│   ├── __init__.py              ← Empty file (required)
│   ├── exchange.py              ← Your Exchange class (required)
│   └── wallet.py                ← Optional wallet logic
└── another_exchange/
    ├── __init__.py
    └── exchange.py
```

**Naming Convention**: 
- Directory: `snake_case`, e.g., `binance_spot`, `hyperliquid_spot`
- File: Always `exchange.py`
- Class: Always `Exchange`

---

## 🔧 **Implementation Steps**

### **Step 1: Create Exchange Directory**

```bash
mkdir -p custom_modules/exchanges/my_exchange
touch custom_modules/exchanges/my_exchange/__init__.py
touch custom_modules/exchanges/my_exchange/exchange.py
```

---

### **Step 2: Implement Exchange Class**

**📍 Location**: `custom_modules/exchanges/my_exchange/exchange.py`

```python
from ats.exchanges.base_exchange import BaseExchange
from ats.exchanges.data_classes.candle import Candle
from ats.exchanges.data_classes.order import Order, OrderStatus
from ats.exchanges.data_classes.asset_balance import AssetBalance
from typing import Callable, Dict
from ats.utils.logging.logger import logger

class Exchange(BaseExchange):
    """
    Custom exchange connector.
    
    IMPORTANT: Class name MUST be 'Exchange' (exact match)
    """
    
    def __init__(self, config: dict, state=None):
        """
        Initialize exchange with configuration.
        
        Args:
            config: Exchange configuration dictionary
            state: SimpleState object for persistent storage
        """
        super().__init__(config, state)
        
        # Add your initialization logic here
        # e.g., API keys, websocket connections, etc.
    
    def validate_config(self) -> None:
        """
        Validate exchange-specific configuration.
        Raise ConfigValidationException if invalid.
        """
        # Example validation
        # if 'api_key' not in self.config:
        #     raise ConfigValidationException("api_key is required")
        pass
    
    def connect(self) -> None:
        """
        Establish connection to the exchange.
        Set self.is_connected = True when successful.
        
        Raises:
            ExchangeConnectionException: If connection fails
        """
        try:
            # Implement connection logic
            # e.g., authenticate API, establish websocket, etc.
            
            self.is_connected = True
            logger.info(f"Connected to exchange: {self.config.get('name', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    def disconnect(self) -> None:
        """
        Disconnect from the exchange.
        Clean up resources, close connections.
        """
        self.is_connected = False
        # Implement cleanup logic
        logger.info("Disconnected from exchange")
    
    def subscribe_to_candles(self, on_candle: Callable) -> None:
        """
        Subscribe to real-time candle data.
        
        Args:
            on_candle: Callback function to call when new candle arrives
                      Should accept a Candle object as parameter
        """
        # Store the callback
        self._on_candle_callback = on_candle
        
        # Implement subscription logic
        # When new candle data arrives, call:
        # candle = Candle(
        #     open=..., high=..., low=..., close=...,
        #     volume=..., timestamp=...
        # )
        # self._on_candle_callback(candle)
    
    def submit_order(self, order: Order) -> None:
        """
        Submit an order to the exchange.
        
        Args:
            order: Order object to submit
        
        Raises:
            OrderNotSubmittedException: If order submission fails
        """
        try:
            # Implement order submission logic
            # Update order status: order.status = OrderStatus.PENDING
            
            logger.info(f"Order submitted: {order}")
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            raise
    
    def cancel_order(self, order: Order) -> None:
        """
        Cancel an existing order.
        
        Args:
            order: Order object to cancel
        """
        # Implement order cancellation logic
        # Update order status: order.status = OrderStatus.CANCELLED
        logger.info(f"Order cancelled: {order}")
    
    def get_wallet_balance(self) -> Dict[str, AssetBalance]:
        """
        Get current wallet balances.
        
        Returns:
            Dictionary mapping asset symbol to AssetBalance
            Example: {
                'BTC': AssetBalance(free=1.5, locked=0.5),
                'USDT': AssetBalance(free=10000.0, locked=0.0)
            }
        """
        # Implement wallet balance retrieval
        return {}
    
    def get_fees(self) -> tuple:
        """
        Get exchange fee rates.
        
        Returns:
            Tuple of (maker_fee_rate, taker_fee_rate)
            Example: (0.001, 0.002) for 0.1% maker, 0.2% taker
        """
        # Return fee rates
        return (0.001, 0.002)  # Example: 0.1% maker, 0.2% taker
```

---

## 📖 **Required Methods**

### **Core Methods** (Must Implement)

| Method | Purpose |
|--------|---------|
| `validate_config()` | Validate configuration parameters |
| `connect()` | Connect to exchange |
| `disconnect()` | Disconnect and cleanup |
| `subscribe_to_candles()` | Subscribe to price data |
| `submit_order()` | Submit buy/sell orders |
| `cancel_order()` | Cancel pending orders |
| `get_wallet_balance()` | Get account balances |
| `get_fees()` | Get fee rates |

---

## 📋 **Data Classes**

### **Candle**

```python
from ats.exchanges.data_classes.candle import Candle

candle = Candle(
    open=50000.0,
    high=51000.0,
    low=49500.0,
    close=50500.0,
    volume=100.5,
    timestamp=1234567890  # Unix timestamp
)
```

### **Order**

```python
from ats.exchanges.data_classes.order import Order, OrderStatus, OrderType, OrderSide

order = Order(
    symbol='BTC/USDT',
    order_type=OrderType.LIMIT,
    order_side=OrderSide.BUY,
    size=1.0,
    price=50000.0
)

# Update order status
order.status = OrderStatus.PENDING
order.status = OrderStatus.COMPLETED
order.status = OrderStatus.CANCELLED
```

### **AssetBalance**

```python
from ats.exchanges.data_classes.asset_balance import AssetBalance

balance = AssetBalance(
    free=1.5,      # Available balance
    locked=0.5     # Locked in orders
)

# Access
total = balance.free + balance.locked
```

---

## 🎬 **Usage in Trading Configuration**

### **YAML Configuration**

```yaml
exchange:
  namespace: "exchanges:my_exchange"  # Your exchange directory name
  config:
    name: "My Custom Exchange"
    api_key: "your_api_key"
    api_secret: "your_api_secret"
    symbol: "BTC/USDT"
    # ... other exchange-specific config
```

### **JSON Configuration**

```json
{
  "exchange": {
    "namespace": "exchanges:my_exchange",
    "config": {
      "name": "My Custom Exchange",
      "api_key": "your_api_key",
      "api_secret": "your_api_secret",
      "symbol": "BTC/USDT"
    }
  }
}
```

---

## 📖 **Example: Simple Exchange Connector**

**File**: `custom_modules/exchanges/paper_trading/exchange.py`

```python
from ats.exchanges.base_exchange import BaseExchange
from ats.exchanges.data_classes.candle import Candle
from ats.exchanges.data_classes.order import Order, OrderStatus
from ats.exchanges.data_classes.asset_balance import AssetBalance
from typing import Callable, Dict
import time

class Exchange(BaseExchange):
    """
    Simple paper trading exchange for testing.
    """
    
    def __init__(self, config: dict, state=None):
        super().__init__(config, state)
        self.wallet = {
            'BTC': AssetBalance(free=10.0, locked=0.0),
            'USDT': AssetBalance(free=100000.0, locked=0.0)
        }
    
    def validate_config(self) -> None:
        if 'symbol' not in self.config:
            raise ValueError("symbol is required")
    
    def connect(self) -> None:
        self.is_connected = True
        print("Paper trading exchange connected")
    
    def disconnect(self) -> None:
        self.is_connected = False
        print("Paper trading exchange disconnected")
    
    def subscribe_to_candles(self, on_candle: Callable) -> None:
        # Simulate candle generation
        def generate_candles():
            while self.is_connected:
                candle = Candle(
                    open=50000.0,
                    high=51000.0,
                    low=49500.0,
                    close=50500.0,
                    volume=100.0,
                    timestamp=int(time.time())
                )
                on_candle(candle)
                time.sleep(1)  # 1 candle per second
        
        import threading
        thread = threading.Thread(target=generate_candles)
        thread.daemon = True
        thread.start()
    
    def submit_order(self, order: Order) -> None:
        order.status = OrderStatus.COMPLETED
        print(f"Paper order executed: {order}")
    
    def cancel_order(self, order: Order) -> None:
        order.status = OrderStatus.CANCELLED
    
    def get_wallet_balance(self) -> Dict[str, AssetBalance]:
        return self.wallet
    
    def get_fees(self) -> tuple:
        return (0.001, 0.001)  # 0.1% maker/taker
```

---

## 🔍 **Testing Your Exchange**

```python
from custom_modules.exchanges.my_exchange.exchange import Exchange

config = {
    'name': 'Test Exchange',
    'symbol': 'BTC/USDT',
    # ... other config
}

exchange = Exchange(config)
exchange.connect()

# Test wallet balance
balance = exchange.get_wallet_balance()
print(f"BTC Balance: {balance.get('BTC')}")

# Test order submission
from ats.exchanges.data_classes.order import Order, OrderType, OrderSide

order = Order(
    symbol='BTC/USDT',
    order_type=OrderType.LIMIT,
    order_side=OrderSide.BUY,
    size=1.0,
    price=50000.0
)

exchange.submit_order(order)
```

---

## 💡 **Implementation Tips**

### **1. Use PlotData for Visualization**

```python
# In your exchange methods
self._plot_data.add_data_point(
    topic='Candle Close Price',
    data_point=(timestamp, close_price)
)
```

### **2. Use SimpleState for Persistence**

```python
# Save state
self._state.set('last_order_id', order_id)

# Load state
last_id = self._state.get('last_order_id', default=0)
```

### **3. Handle Errors Gracefully**

```python
from ats.exceptions.exchange_exceptions import ExchangeConnectionException

try:
    # Connection logic
except Exception as e:
    raise ExchangeConnectionException(f"Connection failed: {e}")
```

---

## 📚 **Additional Resources**

- **Base Class**: [`ats/exchanges/base_exchange.py`](../../ats/exchanges/base_exchange.py)
- **Backtesting Example**: [`ats/exchanges/exchanges/back_trading/exchange.py`](../../ats/exchanges/exchanges/back_trading/exchange.py)
- **Live Example**: [`ats/exchanges/exchanges/binance_spot/exchange.py`](../../ats/exchanges/exchanges/binance_spot/exchange.py)
- **Official Docs**: [ATS Documentation - Adding a New Exchange](https://ats-doc.gitbook.io/v1/customizations/adding-a-new-exchange)

---

## ⚠️ **Important Notes**

1. **Class Name**: MUST be `Exchange` (exact case)
2. **File Name**: MUST be `exchange.py`
3. **Directory**: Use `snake_case` naming
4. **Thread Safety**: Consider thread-safety for live trading
5. **Error Handling**: Use ATS exception classes
6. **Logging**: Use `from ats.utils.logging.logger import logger`

---

## 🔐 **Security Best Practices**

- Never hardcode API keys in code
- Load sensitive data from environment variables or config
- Use `.env` file for local development (add to `.gitignore`)
- Implement rate limiting to avoid API bans
- Handle API errors and connection issues gracefully

---

**Need Help?** Check the [official ATS documentation](https://ats-doc.gitbook.io/v1) or existing examples in `ats/exchanges/exchanges/`.

