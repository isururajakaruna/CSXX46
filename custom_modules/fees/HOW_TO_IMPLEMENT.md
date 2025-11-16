# 💰 How to Implement Custom Fee Structures

This directory is for implementing custom fee calculation logic for trading operations.

---

## 📚 **What Goes Here**

Custom fee structure modules that calculate trading fees based on:
- Order type (LIMIT vs MARKET)
- Order side (BUY vs SELL)
- Order size and price
- Exchange-specific fee rules

---

## 🎯 **File Structure**

```
custom_modules/fees/
├── HOW_TO_IMPLEMENT.md          ← This file
├── my_custom_fees.py             ← Your custom fee class
└── exchange_specific_fees.py     ← Example: Exchange-specific fees
```

**Naming Convention**: `snake_case` filename, e.g., `binance_fees.py`, `maker_taker_fees.py`

---

## 🔧 **Implementation Steps**

### **Step 1: Create Fee Module File**

**📍 Location**: `custom_modules/fees/your_fee_name.py`

```python
from ats.exchanges.base_fees import BaseFees
from typing import Dict, Union, Literal

class YourFeeName(BaseFees):
    """
    Custom fee calculation logic.
    
    The class name should be in PascalCase converted from the filename.
    Example: 
        - File: my_custom_fees.py → Class: MyCustomFees
        - File: binance_fees.py → Class: BinanceFees
    """
    
    def __init__(self, config):
        """
        Initialize with configuration.
        
        Args:
            config: Dictionary containing fee configuration
        """
        super().__init__(config)
        # Add your initialization logic here
        # e.g., self.maker_rate = config.get('maker_rate', 0.001)
    
    def _calculate(self, size: Union[int, float], price: Union[int, float],
                   order_type: Literal['LIMIT', 'MARKET'], 
                   order_side: Literal['BUY', 'SELL']) -> Dict:
        """
        Calculate fees for an order.
        
        Args:
            size: Order size (amount of base asset)
            price: Order price (per unit of base asset)
            order_type: 'LIMIT' or 'MARKET'
            order_side: 'BUY' or 'SELL'
        
        Returns:
            Dictionary with 'base_fee' and 'quote_fee' keys
            Example: {'base_fee': 0.01, 'quote_fee': 100.0}
        """
        # Implement your fee calculation logic here
        
        # Example: Simple percentage-based fees
        # order_value = size * price
        # fee_rate = 0.001  # 0.1%
        # 
        # if order_side == 'BUY':
        #     base_fee = size * fee_rate
        #     quote_fee = 0
        # else:  # SELL
        #     base_fee = 0
        #     quote_fee = order_value * fee_rate
        
        return {
            'base_fee': 0.0,      # Replace with calculated base fee
            'quote_fee': 0.0      # Replace with calculated quote fee
        }
```

---

### **Step 2: Use in Trading Configuration**

Reference your custom fee structure in the exchange configuration:

```yaml
exchange:
  namespace: "exchanges:back_trading"
  config:
    # ... other exchange config ...
    fees:
      namespace: "fees:my_custom_fees"  # Your fee module
      config:
        maker_rate: 0.001
        taker_rate: 0.002
```

Or in JSON:

```json
{
  "exchange": {
    "namespace": "exchanges:back_trading",
    "config": {
      "fees": {
        "namespace": "fees:my_custom_fees",
        "config": {
          "maker_rate": 0.001,
          "taker_rate": 0.002
        }
      }
    }
  }
}
```

---

## 📖 **Example Implementation**

**File**: `custom_modules/fees/maker_taker_fees.py`

```python
from ats.exchanges.base_fees import BaseFees
from typing import Dict, Union, Literal

class MakerTakerFees(BaseFees):
    """
    Maker/Taker fee structure.
    - Maker fee: For limit orders that add liquidity
    - Taker fee: For market orders that remove liquidity
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.maker_rate = config.get('maker_rate', 0.001)  # 0.1%
        self.taker_rate = config.get('taker_rate', 0.002)  # 0.2%
    
    def _calculate(self, size: Union[int, float], price: Union[int, float],
                   order_type: Literal['LIMIT', 'MARKET'], 
                   order_side: Literal['BUY', 'SELL']) -> Dict:
        """
        Calculate maker/taker fees.
        """
        order_value = size * price
        
        # Determine fee rate based on order type
        fee_rate = self.maker_rate if order_type == 'LIMIT' else self.taker_rate
        
        # Calculate fees
        if order_side == 'BUY':
            # Buyer pays fee in base asset
            base_fee = size * fee_rate
            quote_fee = 0
        else:  # SELL
            # Seller pays fee in quote asset
            base_fee = 0
            quote_fee = order_value * fee_rate
        
        return {
            'base_fee': base_fee,
            'quote_fee': quote_fee
        }
```

---

## 📋 **Key Concepts**

### **Base Fee vs Quote Fee**

- **Base Fee**: Fee paid in the base asset (e.g., BTC in BTC/USDT)
- **Quote Fee**: Fee paid in the quote asset (e.g., USDT in BTC/USDT)

### **Order Types**

- **LIMIT**: Order placed at specific price (usually lower fees - maker)
- **MARKET**: Order executed immediately at market price (usually higher fees - taker)

### **Order Sides**

- **BUY**: Purchasing base asset with quote asset
- **SELL**: Selling base asset for quote asset

---

## 🔍 **Testing Your Fee Structure**

```python
# Example test
from custom_modules.fees.my_custom_fees import MyCustomFees

config = {'maker_rate': 0.001, 'taker_rate': 0.002}
fee_calculator = MyCustomFees(config)

# Calculate fee for a LIMIT BUY order
fee_calculator.calculate(
    size=1.0,           # Buy 1 BTC
    price=50000.0,      # At $50,000
    order_type='LIMIT',
    order_side='BUY'
)

fees = fee_calculator.get()
print(f"Base fee: {fees['base_fee']}")
print(f"Quote fee: {fees['quote_fee']}")
```

---

## 📚 **Additional Resources**

- **Base Class**: [`ats/exchanges/base_fees.py`](../../ats/exchanges/base_fees.py)
- **Example**: [`ats/exchanges/fees/generic.py`](../../ats/exchanges/fees/generic.py)
- **Official Docs**: [ATS Documentation - Custom Fee Structures](https://ats-doc.gitbook.io/v1/customizations/custom-fee-structures)

---

## 💡 **Common Fee Patterns**

### **Percentage-Based Fees**
```python
fee = order_value * fee_rate  # e.g., 0.1% of trade value
```

### **Tiered Fees** (Volume-based)
```python
if volume < 10000:
    fee_rate = 0.002
elif volume < 100000:
    fee_rate = 0.001
else:
    fee_rate = 0.0005
```

### **Fixed + Percentage**
```python
fee = fixed_amount + (order_value * percentage_rate)
```

---

## ⚠️ **Important Notes**

1. **Class Name**: Must match filename in PascalCase (auto-converted by ATS)
2. **Return Format**: Always return dict with `base_fee` and `quote_fee` keys
3. **Inheritance**: Must extend `BaseFees` from `ats.exchanges.base_fees`
4. **Override**: `_calculate()` method is required (note the underscore prefix)

---

**Need Help?** Check the [official ATS documentation](https://ats-doc.gitbook.io/v1) or existing examples in `ats/exchanges/fees/`.

