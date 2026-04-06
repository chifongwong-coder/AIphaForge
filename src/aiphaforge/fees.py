"""
Fee Models

Supports transaction cost calculation for multiple markets:
- US stocks: per-share commission
- China A-shares: commission + stamp duty + transfer fee
- Crypto spot: maker/taker fee rates
- Crypto futures: maker/taker + funding rate
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional


class MarketType(Enum):
    """Market type."""
    US_STOCK = "us_stock"
    CHINA_A_SHARE = "china_a_share"
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_FUTURES = "crypto_futures"
    GENERIC = "generic"


class BaseFeeModel(ABC):
    """
    Base class for fee models.

    All fee models must inherit from this class and implement the required methods.

    Attributes:
        name: Model name.
        market_type: Market type.
        slippage_pct: Slippage percentage.

    Example:
        >>> fee_model = ChinaAShareFeeModel()
        >>> commission = fee_model.calculate_commission(10.0, 1000, 'buy')
        >>> slippage = fee_model.calculate_slippage(10.0, 1000, 'buy', 1000000)
        >>> total = fee_model.total_cost(10.0, 1000, 'buy', 1000000)
    """

    name: str = "base"
    market_type: MarketType = MarketType.GENERIC

    def __init__(self, slippage_pct: float = 0.001):
        """
        Initialize the fee model.

        Parameters:
            slippage_pct: Slippage percentage, default 0.1%.
        """
        self.slippage_pct = slippage_pct

    @abstractmethod
    def calculate_commission(
        self,
        price: float,
        size: float,
        side: str
    ) -> float:
        """
        Calculate commission.

        Parameters:
            price: Execution price.
            size: Trade quantity.
            side: Trade direction ('buy' or 'sell').

        Returns:
            float: Commission amount.
        """
        pass

    def calculate_slippage(
        self,
        price: float,
        size: float,
        side: str,
        volume: Optional[float] = None
    ) -> float:
        """
        Calculate slippage cost.

        Parameters:
            price: Execution price.
            size: Trade quantity.
            side: Trade direction.
            volume: Current bar volume (optional, for volume-based slippage).

        Returns:
            float: Slippage cost amount.
        """
        notional = price * size
        return notional * self.slippage_pct

    def total_cost(
        self,
        price: float,
        size: float,
        side: str,
        volume: Optional[float] = None
    ) -> float:
        """
        Calculate total transaction cost.

        Parameters:
            price: Execution price.
            size: Trade quantity.
            side: Trade direction.
            volume: Current bar volume.

        Returns:
            float: Total cost = commission + slippage.
        """
        commission = self.calculate_commission(price, size, side)
        slippage = self.calculate_slippage(price, size, side, volume)
        return commission + slippage

    def get_execution_price(
        self,
        price: float,
        side: str,
        volume: Optional[float] = None
    ) -> float:
        """
        Calculate the actual execution price after slippage.

        Parameters:
            price: Target price.
            side: Trade direction.
            volume: Volume.

        Returns:
            float: Actual execution price.
        """
        slippage_amount = price * self.slippage_pct
        if side == 'buy':
            return price + slippage_amount
        else:
            return price - slippage_amount

    def __repr__(self):
        return f"{self.__class__.__name__}(slippage={self.slippage_pct*100:.2f}%)"


class SimpleFeeModel(BaseFeeModel):
    """
    Simple fee model.

    Uses a fixed commission rate. Suitable for quick backtesting.

    Parameters:
        commission_rate: Commission rate, default 0.1%.
        slippage_pct: Slippage percentage, default 0.1%.

    Example:
        >>> fee_model = SimpleFeeModel(commission_rate=0.001)
        >>> cost = fee_model.total_cost(100.0, 100, 'buy')
    """

    name = "simple"
    market_type = MarketType.GENERIC

    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_pct: float = 0.001
    ):
        super().__init__(slippage_pct)
        self.commission_rate = commission_rate

    def calculate_commission(
        self,
        price: float,
        size: float,
        side: str
    ) -> float:
        """Commission = notional * rate."""
        notional = price * size
        return notional * self.commission_rate

    def __repr__(self):
        return (f"SimpleFeeModel(commission={self.commission_rate*100:.2f}%, "
                f"slippage={self.slippage_pct*100:.2f}%)")


class USStockFeeModel(BaseFeeModel):
    """
    US stock fee model.

    Typical broker fee structure:
    - Per-share commission (e.g. $0.005/share)
    - Minimum commission
    - No stamp duty

    Parameters:
        commission_per_share: Per-share commission, default $0.005.
        min_commission: Minimum commission, default $1.0.
        slippage_pct: Slippage percentage, default 0.1%.

    Example:
        >>> fee_model = USStockFeeModel()
        >>> cost = fee_model.total_cost(150.0, 100, 'buy')
    """

    name = "us_stock"
    market_type = MarketType.US_STOCK

    def __init__(
        self,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_pct: float = 0.001
    ):
        super().__init__(slippage_pct)
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

    def calculate_commission(
        self,
        price: float,
        size: float,
        side: str
    ) -> float:
        """Commission = max(shares * per_share_fee, min_commission)."""
        commission = size * self.commission_per_share
        return max(commission, self.min_commission)

    def __repr__(self):
        return (f"USStockFeeModel(per_share=${self.commission_per_share:.4f}, "
                f"min=${self.min_commission:.2f})")


class ChinaAShareFeeModel(BaseFeeModel):
    """
    China A-share fee model.

    Fee structure:
    - Commission: notional * rate (both sides, with minimum)
    - Stamp duty: notional * 0.1% (sell only)
    - Transfer fee: notional * 0.002% (both sides, Shanghai only, simplified to all)

    Parameters:
        commission_rate: Commission rate, default 0.03% (3 basis points).
        min_commission: Minimum commission, default 5 CNY.
        stamp_duty_rate: Stamp duty rate, default 0.1%.
        transfer_fee_rate: Transfer fee rate, default 0.002%.
        slippage_pct: Slippage percentage, default 0.1%.

    Example:
        >>> fee_model = ChinaAShareFeeModel()
        >>> cost = fee_model.total_cost(10.0, 10000, 'sell')
    """

    name = "china_a_share"
    market_type = MarketType.CHINA_A_SHARE

    def __init__(
        self,
        commission_rate: float = 0.0003,
        min_commission: float = 5.0,
        stamp_duty_rate: float = 0.001,
        transfer_fee_rate: float = 0.00002,
        slippage_pct: float = 0.001
    ):
        super().__init__(slippage_pct)
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate
        self.transfer_fee_rate = transfer_fee_rate

    def calculate_commission(
        self,
        price: float,
        size: float,
        side: str
    ) -> float:
        """
        Calculate A-share transaction fees.

        Includes:
        - Commission (both sides)
        - Stamp duty (sell only)
        - Transfer fee (both sides)
        """
        notional = price * size

        # Commission
        commission = max(notional * self.commission_rate, self.min_commission)

        # Transfer fee
        transfer_fee = notional * self.transfer_fee_rate

        # Stamp duty (sell only)
        stamp_duty = 0.0
        if side == 'sell':
            stamp_duty = notional * self.stamp_duty_rate

        return commission + transfer_fee + stamp_duty

    def get_commission_breakdown(
        self,
        price: float,
        size: float,
        side: str
    ) -> dict:
        """
        Get itemized fee breakdown.

        Returns:
            dict: Breakdown of each fee component.
        """
        notional = price * size
        commission = max(notional * self.commission_rate, self.min_commission)
        transfer_fee = notional * self.transfer_fee_rate
        stamp_duty = notional * self.stamp_duty_rate if side == 'sell' else 0.0

        return {
            'notional': notional,
            'commission': commission,
            'transfer_fee': transfer_fee,
            'stamp_duty': stamp_duty,
            'total': commission + transfer_fee + stamp_duty
        }

    def __repr__(self):
        return (f"ChinaAShareFeeModel(commission={self.commission_rate*10000:.1f}bps, "
                f"stamp_duty={self.stamp_duty_rate*1000:.1f}permille)")


class CryptoSpotFeeModel(BaseFeeModel):
    """
    Crypto spot fee model.

    Fee structure:
    - Maker fee: fee rate for limit orders that add liquidity.
    - Taker fee: fee rate for market orders that remove liquidity.
    - Backtests typically use taker fee (market orders).

    Parameters:
        maker_fee: Maker fee rate, default 0.1%.
        taker_fee: Taker fee rate, default 0.1%.
        use_maker: Whether to use maker fee, default False.
        slippage_pct: Slippage percentage, default 0.05%.

    Example:
        >>> fee_model = CryptoSpotFeeModel(maker_fee=0.001, taker_fee=0.001)
        >>> cost = fee_model.total_cost(50000.0, 0.1, 'buy')
    """

    name = "crypto_spot"
    market_type = MarketType.CRYPTO_SPOT

    def __init__(
        self,
        maker_fee: float = 0.001,
        taker_fee: float = 0.001,
        use_maker: bool = False,
        slippage_pct: float = 0.0005
    ):
        super().__init__(slippage_pct)
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.use_maker = use_maker

    @property
    def fee_rate(self) -> float:
        """Currently active fee rate."""
        return self.maker_fee if self.use_maker else self.taker_fee

    def calculate_commission(
        self,
        price: float,
        size: float,
        side: str
    ) -> float:
        """Commission = notional * fee_rate."""
        notional = price * size
        return notional * self.fee_rate

    def __repr__(self):
        fee_type = "maker" if self.use_maker else "taker"
        return (f"CryptoSpotFeeModel({fee_type}={self.fee_rate*100:.2f}%, "
                f"slippage={self.slippage_pct*100:.3f}%)")


class CryptoFuturesFeeModel(BaseFeeModel):
    """
    Crypto futures fee model.

    Fee structure:
    - Maker/taker fees (typically lower than spot)
    - Funding rate (settled every 8 hours)

    Parameters:
        maker_fee: Maker fee rate, default 0.02%.
        taker_fee: Taker fee rate, default 0.04%.
        funding_rate: Funding rate (per 8h), default 0.01%.
        use_maker: Whether to use maker fee.
        slippage_pct: Slippage percentage.

    Note:
        Funding rate is paid between longs and shorts. Positive rate means
        longs pay shorts, negative rate means the opposite.
        In backtesting this is simplified to always deducting the absolute value.

    Example:
        >>> fee_model = CryptoFuturesFeeModel()
        >>> open_cost = fee_model.total_cost(50000.0, 0.1, 'buy')
        >>> funding = fee_model.calculate_funding_cost(50000.0 * 0.1, 8)
    """

    name = "crypto_futures"
    market_type = MarketType.CRYPTO_FUTURES

    def __init__(
        self,
        maker_fee: float = 0.0002,
        taker_fee: float = 0.0004,
        funding_rate: float = 0.0001,
        use_maker: bool = False,
        slippage_pct: float = 0.0005
    ):
        super().__init__(slippage_pct)
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.funding_rate = funding_rate
        self.use_maker = use_maker

    @property
    def fee_rate(self) -> float:
        """Currently active fee rate."""
        return self.maker_fee if self.use_maker else self.taker_fee

    def calculate_commission(
        self,
        price: float,
        size: float,
        side: str
    ) -> float:
        """Calculate commission."""
        notional = price * size
        return notional * self.fee_rate

    def calculate_funding_cost(
        self,
        position_value: float,
        hours_held: float
    ) -> float:
        """
        Calculate funding cost.

        Parameters:
            position_value: Position value.
            hours_held: Number of hours the position was held.

        Returns:
            float: Funding cost.
        """
        funding_periods = hours_held / 8.0
        return abs(position_value * self.funding_rate * funding_periods)

    def total_cost_with_funding(
        self,
        price: float,
        size: float,
        side: str,
        hours_held: float = 0,
        volume: Optional[float] = None
    ) -> float:
        """
        Calculate total cost including funding rate.

        Parameters:
            price: Execution price.
            size: Trade quantity.
            side: Trade direction.
            hours_held: Hours the position was held.
            volume: Bar volume.

        Returns:
            float: Total cost.
        """
        base_cost = self.total_cost(price, size, side, volume)
        funding = self.calculate_funding_cost(price * size, hours_held)
        return base_cost + funding

    def __repr__(self):
        fee_type = "maker" if self.use_maker else "taker"
        return (f"CryptoFuturesFeeModel({fee_type}={self.fee_rate*100:.2f}%, "
                f"funding={self.funding_rate*100:.3f}%/8h)")


class ZeroFeeModel(BaseFeeModel):
    """
    Zero fee model.

    Used for testing or when fees should be ignored.

    Example:
        >>> fee_model = ZeroFeeModel()
        >>> assert fee_model.total_cost(100, 100, 'buy') == 0
    """

    name = "zero"
    market_type = MarketType.GENERIC

    def __init__(self):
        super().__init__(slippage_pct=0.0)

    def calculate_commission(
        self,
        price: float,
        size: float,
        side: str
    ) -> float:
        return 0.0

    def calculate_slippage(
        self,
        price: float,
        size: float,
        side: str,
        volume: Optional[float] = None
    ) -> float:
        return 0.0


# ============================================================================
# Fee Model Factory
# ============================================================================

def get_fee_model(market: str, **kwargs) -> BaseFeeModel:
    """
    Factory function for creating fee models.

    Parameters:
        market: Market type string.
            - 'us_stock' / 'us': US stocks
            - 'china' / 'a_share' / 'cn': China A-shares
            - 'crypto' / 'crypto_spot': Crypto spot
            - 'crypto_futures' / 'futures': Crypto futures
            - 'simple': Simple flat rate
            - 'zero' / 'none': Zero fees
        **kwargs: Arguments passed to the fee model constructor.

    Returns:
        BaseFeeModel: Fee model instance.

    Example:
        >>> fee_model = get_fee_model('china')
        >>> fee_model = get_fee_model('crypto', maker_fee=0.0008)
    """
    market = market.lower().replace('-', '_').replace(' ', '_')

    mapping = {
        'us_stock': USStockFeeModel,
        'us': USStockFeeModel,
        'china': ChinaAShareFeeModel,
        'china_a_share': ChinaAShareFeeModel,
        'a_share': ChinaAShareFeeModel,
        'cn': ChinaAShareFeeModel,
        'crypto': CryptoSpotFeeModel,
        'crypto_spot': CryptoSpotFeeModel,
        'crypto_futures': CryptoFuturesFeeModel,
        'futures': CryptoFuturesFeeModel,
        'simple': SimpleFeeModel,
        'zero': ZeroFeeModel,
        'none': ZeroFeeModel
    }

    if market not in mapping:
        raise ValueError(f"Unknown market type: {market}. Available: {list(mapping.keys())}")

    return mapping[market](**kwargs)
