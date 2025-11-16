# 📊 How to Implement Custom Indicators

This directory is for implementing custom technical indicators for trading strategies.

---

## 📚 **What Goes Here**

Custom technical indicators such as:
- Moving averages (SMA, EMA, WMA)
- Oscillators (RSI, MACD, Stochastic)
- Volatility indicators (Bollinger Bands, ATR)
- Volume indicators (OBV, Volume Profile)
- Custom mathematical transformations
- Machine learning-based indicators

---

## 🎯 **File Structure**

```
custom_modules/indicators/
├── HOW_TO_IMPLEMENT.md          ← This file
├── my_indicator.py               ← Your custom indicator
├── advanced_rsi.py               ← Example: Enhanced RSI
└── ml_predictor.py               ← Example: ML-based indicator
```

**Naming Convention**: 
- File: `snake_case`, e.g., `moving_average.py`, `custom_rsi.py`
- Class: PascalCase (auto-converted from filename)
  - `moving_average.py` → `MovingAverage`
  - `custom_rsi.py` → `CustomRsi`

---

## 🔧 **Implementation Steps**

### **Step 1: Create Indicator File**

**📍 Location**: `custom_modules/indicators/my_indicator.py`

```python
from ats.indicators.base_indicator import BaseIndicator

class MyIndicator(BaseIndicator):
    """
    Custom indicator implementation.
    
    The class name should be in PascalCase converted from the filename.
    Example:
        - File: my_indicator.py → Class: MyIndicator
        - File: advanced_macd.py → Class: AdvancedMacd
    """
    
    def __init__(self, config: dict):
        """
        Initialize indicator with configuration.
        
        Args:
            config: Dictionary with indicator configuration
                   Must include 'N' (window size)
        
        Available attributes:
            - self.time_series: Deque storing last N data points
            - self.config: Configuration dictionary
            - self.N: Window size
            - self.result: Latest indicator result
            - self._is_ready: True when window is full
        """
        super().__init__(config)
        
        # Initialize your state variables
        self.sums = 0  # Example: running sum
        # Add more state variables as needed
    
    def running_calc(self, popped_data_point, new_data_point):
        """
        Calculate indicator in a running window fashion.
        
        This method is called every time a new data point arrives.
        It receives:
        - The data point that was removed from the window (or None)
        - The new data point to process
        
        Args:
            popped_data_point: Data point removed from window (None if window not full)
            new_data_point: New data point to process
        
        Returns:
            The processed data point to add to time_series
            (You can return the original or transformed value)
        """
        # Update running calculations efficiently
        if popped_data_point is not None:
            # Remove old value contribution
            self.sums -= popped_data_point
        
        # Add new value contribution
        self.sums += new_data_point
        
        # Calculate indicator result
        if self._is_ready:
            self.result = self.sums / self.N  # Example: Simple moving average
        
        # Return the value to store in time_series
        return new_data_point
```

---

### **Step 2: Use in Strategy**

```python
from custom_modules.indicators.my_indicator import MyIndicator

class Strategy(BaseStrategy):
    def validate_config(self):
        # Initialize indicator
        self.my_indicator = MyIndicator(config={'N': 20})
    
    def on_candle(self, candle):
        # Add new data point
        self.my_indicator.add(candle.close)
        
        # Check if indicator is ready
        if self.my_indicator.is_ready():
            value = self.my_indicator.result
            print(f"Indicator value: {value}")
```

---

## 📖 **Complete Example: Simple Moving Average (SMA)**

**File**: `custom_modules/indicators/simple_moving_average.py`

```python
from ats.indicators.base_indicator import BaseIndicator

class SimpleMovingAverage(BaseIndicator):
    """
    Simple Moving Average (SMA) indicator.
    Calculates the average of the last N data points.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Must include 'N' (window size)
                   Example: {'N': 20} for 20-period SMA
        """
        super().__init__(config)
        self.sums = 0
    
    def running_calc(self, popped_data_point, new_data_point):
        """
        Efficiently calculate SMA using running sum.
        
        Time complexity: O(1) per update
        """
        # Remove old value from sum
        if popped_data_point is not None:
            self.sums -= popped_data_point
        
        # Add new value to sum
        self.sums += new_data_point
        
        # Calculate average when window is full
        if self._is_ready:
            self.result = self.sums / self.N
        
        return new_data_point


# Usage example
sma = SimpleMovingAverage(config={'N': 20})

prices = [100, 102, 101, 105, 103, ...]
for price in prices:
    sma.add(price)
    if sma.is_ready():
        print(f"SMA(20): {sma.result}")
```

---

## 📖 **Advanced Example: Relative Strength Index (RSI)**

**File**: `custom_modules/indicators/relative_strength_index.py`

```python
from ats.indicators.base_indicator import BaseIndicator

class RelativeStrengthIndex(BaseIndicator):
    """
    RSI indicator using exponential moving average.
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Must include 'N' (period, typically 14)
        """
        super().__init__(config)
        self.avg_gain = 0
        self.avg_loss = 0
        self.prev_price = None
        self.first_calc = True
    
    def running_calc(self, popped_data_point, new_data_point):
        """
        Calculate RSI using exponential smoothing.
        """
        if self.prev_price is None:
            self.prev_price = new_data_point
            return new_data_point
        
        # Calculate price change
        change = new_data_point - self.prev_price
        gain = max(change, 0)
        loss = max(-change, 0)
        
        if self.first_calc and len(self.time_series) == self.N - 1:
            # First calculation: simple average
            gains = [max(self.time_series[i] - self.time_series[i-1], 0) 
                     for i in range(1, len(self.time_series))]
            losses = [max(self.time_series[i-1] - self.time_series[i], 0) 
                      for i in range(1, len(self.time_series))]
            
            self.avg_gain = (sum(gains) + gain) / self.N
            self.avg_loss = (sum(losses) + loss) / self.N
            self.first_calc = False
        elif self._is_ready:
            # Exponential smoothing
            alpha = 1 / self.N
            self.avg_gain = (self.avg_gain * (1 - alpha)) + (gain * alpha)
            self.avg_loss = (self.avg_loss * (1 - alpha)) + (loss * alpha)
        
        # Calculate RSI
        if self._is_ready and self.avg_loss != 0:
            rs = self.avg_gain / self.avg_loss
            self.result = 100 - (100 / (1 + rs))
        
        self.prev_price = new_data_point
        return new_data_point


# Usage
rsi = RelativeStrengthIndex(config={'N': 14})

for price in price_data:
    rsi.add(price)
    if rsi.is_ready():
        print(f"RSI(14): {rsi.result:.2f}")
        
        # Trading signals
        if rsi.result < 30:
            print("Oversold - potential buy")
        elif rsi.result > 70:
            print("Overbought - potential sell")
```

---

## 📖 **Advanced Example: Bollinger Bands**

**File**: `custom_modules/indicators/bollinger_bands.py`

```python
from ats.indicators.base_indicator import BaseIndicator
import math

class BollingerBands(BaseIndicator):
    """
    Bollinger Bands: SMA ± (k * standard deviation)
    Returns dict with 'upper', 'middle', 'lower' bands
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Must include:
                - 'N': Period (typically 20)
                - 'k': Number of standard deviations (typically 2)
        """
        super().__init__(config)
        self.k = config.get('k', 2)
        self.sums = 0
        self.sum_squares = 0
    
    def running_calc(self, popped_data_point, new_data_point):
        """
        Calculate Bollinger Bands.
        """
        # Update running sums
        if popped_data_point is not None:
            self.sums -= popped_data_point
            self.sum_squares -= popped_data_point ** 2
        
        self.sums += new_data_point
        self.sum_squares += new_data_point ** 2
        
        # Calculate when ready
        if self._is_ready:
            # Middle band (SMA)
            middle = self.sums / self.N
            
            # Standard deviation
            variance = (self.sum_squares / self.N) - (middle ** 2)
            std_dev = math.sqrt(max(variance, 0))
            
            # Upper and lower bands
            upper = middle + (self.k * std_dev)
            lower = middle - (self.k * std_dev)
            
            self.result = {
                'upper': upper,
                'middle': middle,
                'lower': lower,
                'std_dev': std_dev
            }
        
        return new_data_point


# Usage
bb = BollingerBands(config={'N': 20, 'k': 2})

for price in prices:
    bb.add(price)
    if bb.is_ready():
        bands = bb.result
        print(f"Upper: {bands['upper']:.2f}")
        print(f"Middle: {bands['middle']:.2f}")
        print(f"Lower: {bands['lower']:.2f}")
```

---

## 💡 **Key Concepts**

### **Running Window Calculation**

- Efficient O(1) updates instead of recalculating entire window
- Use `popped_data_point` to remove old values
- Use `new_data_point` to add new values

### **Indicator Lifecycle**

1. **Initialization**: `__init__()` sets up state
2. **Filling Window**: First N data points
3. **Ready State**: When `len(time_series) == N`, `_is_ready = True`
4. **Running Updates**: Efficient O(1) updates after ready

### **State Management**

```python
self.time_series    # Deque of last N processed values
self.result         # Latest indicator output
self._is_ready      # True when window is full
self.N              # Window size
self.config         # Configuration dict
```

---

## 🎯 **Usage in Configuration**

### **YAML**

```yaml
strategy:
  namespace: "strategies:my_strategy"
  config:
    indicators:
      - name: "my_indicator"
        namespace: "indicators:my_indicator"
        config:
          N: 20
```

### **JSON**

```json
{
  "strategy": {
    "namespace": "strategies:my_strategy",
    "config": {
      "indicators": [
        {
          "name": "my_indicator",
          "namespace": "indicators:my_indicator",
          "config": {"N": 20}
        }
      ]
    }
  }
}
```

---

## 🔍 **Testing Your Indicator**

```python
# Test indicator
from custom_modules.indicators.my_indicator import MyIndicator

indicator = MyIndicator(config={'N': 5})

test_data = [10, 12, 11, 13, 14, 15, 16]

for i, value in enumerate(test_data):
    indicator.add(value)
    print(f"Step {i+1}:")
    print(f"  Ready: {indicator.is_ready()}")
    print(f"  Result: {indicator.result}")
    print(f"  Time series length: {len(indicator.time_series)}")
```

---

## 📚 **Additional Resources**

- **Base Class**: [`ats/indicators/base_indicator.py`](../../ats/indicators/base_indicator.py)
- **Examples**: 
  - [`ats/indicators/mean_indicator.py`](../../ats/indicators/mean_indicator.py)
  - [`ats/indicators/rsi_indicator.py`](../../ats/indicators/rsi_indicator.py)
  - [`ats/indicators/std_indicator.py`](../../ats/indicators/std_indicator.py)
- **Official Docs**: [ATS Documentation - Adding an Indicator](https://ats-doc.gitbook.io/v1/customizations/adding-an-indicator)

---

## 💡 **Common Patterns**

### **1. Running Sum (SMA, EMA)**
```python
if popped_data_point:
    self.sums -= popped_data_point
self.sums += new_data_point
```

### **2. Running Variance (Std Dev, Bollinger)**
```python
self.sums += new_data_point
self.sum_squares += new_data_point ** 2
variance = (self.sum_squares / N) - (mean ** 2)
```

### **3. Rate of Change**
```python
if len(self.time_series) > 0:
    change = new_data_point - self.time_series[0]
    percent_change = (change / self.time_series[0]) * 100
```

### **4. Exponential Smoothing**
```python
alpha = 2 / (N + 1)
self.ema = (new_value * alpha) + (self.ema * (1 - alpha))
```

---

## ⚠️ **Important Notes**

1. **Class Name**: Must be PascalCase of filename (auto-converted)
2. **Window Size**: Config must include 'N' parameter
3. **Return Value**: `running_calc()` must return a value for time_series
4. **Update Result**: Set `self.result` when indicator is ready
5. **Efficiency**: Use running calculations, not full recalculation

---

## 🎓 **Performance Tips**

- Use deque operations (O(1)) instead of list operations
- Maintain running sums/statistics instead of recalculating
- Avoid unnecessary loops over time_series
- Consider memory usage for large N values
- Cache intermediate calculations

---

**Need Help?** Check the [official ATS documentation](https://ats-doc.gitbook.io/v1) or existing examples in `ats/indicators/`.

