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

# v2.8.4 M11b: parameterize ``USStockFeeModel.__repr__`` year tags so updating
# the historically-tagged schedule label is a one-line change in this module
# rather than a string edit inside the formatter.  The default ``__init__``
# values still reflect the FY2026 SEC fee + 2026 FINRA TAF schedules.
_REPR_SEC_SCHEDULE_LABEL = "FY2026"
_REPR_FINRA_SCHEDULE_LABEL = "2026"

# v2.8: public surface lock.
__all__ = [
    "BaseFeeModel",
    "ChinaAShareFeeModel",
    "CryptoFuturesFeeModel",
    "CryptoSpotFeeModel",
    "MarketType",
    "SimpleFeeModel",
    "USStockFeeModel",
    "ZeroFeeModel",
    "get_fee_model",
]


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

    def estimate_commission_rate(
        self,
        price: float = 100.0,
        size: float = 100.0,
        side: str = "average",
    ) -> float:
        """Estimate commission as a fraction of notional value.

        Useful for vectorized mode where a single scalar rate is needed.
        The default implementation computes the actual commission and divides
        by notional value.  Subclasses may override for efficiency but the
        default is always correct.

        Parameters:
            price: Representative price for the estimate.
            size: Representative trade size for the estimate.
            side: Trade direction ('buy', 'sell', or 'average').
                When ``'average'``, the buy and sell rates are averaged,
                which gives a more representative estimate for models
                with asymmetric fees (e.g. stamp duty on sell only).

        Returns:
            float: Commission as a fraction of notional (e.g. 0.001 = 0.1%).
        """
        notional = price * size
        if notional <= 0:
            return 0.0
        if side == "average":
            buy_commission = self.calculate_commission(price, size, "buy")
            sell_commission = self.calculate_commission(price, size, "sell")
            return (buy_commission + sell_commission) / (2 * notional)
        commission = self.calculate_commission(price, size, side)
        return commission / notional

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

    Broker fee + regulatory sell-side fees (v2.8.2):

    - **Broker commission**: per-share, with a per-trade minimum
      (e.g. $0.005/share with a $1 minimum).
    - **SEC §31 fee** (sell-only): a regulatory transaction fee on
      sell transactions, computed as ``price * size *
      sec_fee_rate``. The rate is set periodically by the SEC Fee
      Rate Advisory. v2.8.2 ships the FY2026 post-2026-04-04 rate
      ``20.60e-6`` as the default. Pass ``sec_fee_rate=0`` for
      backtests in periods before SEC §31 existed (pre-1971) or
      override with the period-specific rate from the SEC archive.
    - **FINRA TAF** (sell-only): the FINRA Trading Activity Fee,
      computed as ``min(size * finra_taf_per_share, finra_taf_cap)``.
      Originally the NASD TAF (effective 2003-04-01); renamed to
      FINRA TAF in 2007 when FINRA was formed. v2.8.2 ships the 2026
      schedule (``0.000195/share, cap $9.79``) as defaults. Pass
      ``finra_taf_per_share=0`` for pre-2003-04 backtests.

    Buy-side: only the broker commission applies (no regulatory
    fees on buys).

    The regulatory fees are ADDED on top of the broker commission,
    independently of the broker's ``min_commission`` floor. The
    SEC and FINRA charges are statutory and unaffected by broker
    minimums.

    Parameters:
        commission_per_share: Per-share broker commission, default
            $0.005.
        min_commission: Minimum broker commission per trade,
            default $1.0.
        slippage_pct: Slippage percentage, default 0.1%.
        sec_fee_rate: SEC §31 fee as a fraction of notional, default
            ``20.60e-6`` (FY2026 schedule, effective 2026-04-04 per
            the SEC Fee Rate Advisory FY2026-2).
        finra_taf_per_share: FINRA TAF per-share rate (sell-only),
            default ``0.000195`` (2026 schedule, effective
            2026-01-01).
        finra_taf_cap: FINRA TAF per-trade cap, default ``9.79``
            (2026 schedule).

    Example:
        >>> fee_model = USStockFeeModel()
        >>> cost = fee_model.total_cost(150.0, 100, 'sell')

    Notes
    -----
    ``side="average"`` queries (used by the vectorized cost path's
    representative commission rate) yield ``(buy + sell) / 2``,
    which naturally picks up HALF of the sell-side regulatory fees.
    That is the correct estimator for "expected cost per trade
    when buys and sells are equally likely".
    """

    name = "us_stock"
    market_type = MarketType.US_STOCK

    def __init__(
        self,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0,
        slippage_pct: float = 0.001,
        # v2.8.2: SEC §31 + FINRA TAF (sell-only, FY2026 / 2026 schedule)
        sec_fee_rate: float = 20.60e-6,
        finra_taf_per_share: float = 0.000195,
        finra_taf_cap: float = 9.79,
    ):
        super().__init__(slippage_pct)
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.sec_fee_rate = sec_fee_rate
        self.finra_taf_per_share = finra_taf_per_share
        self.finra_taf_cap = finra_taf_cap

    def calculate_commission(
        self,
        price: float,
        size: float,
        side: str
    ) -> float:
        """Commission = broker fee + (sell-only) SEC §31 + FINRA TAF.

        Broker fee: ``max(shares * per_share_fee, min_commission)``.
        Regulatory fees apply only when ``side == "sell"`` and are
        added on top of the broker fee — independent of the
        broker's minimum.

        FINRA TAF cap × ``side="average"`` asymmetry (v2.8.3 Commit I
        item 1): ``BaseFeeModel.estimate_commission_rate(side="average")``
        computes ``(buy_commission + sell_commission) / 2``. On
        trades small enough that the cap does not bind
        (``size * finra_taf_per_share <= finra_taf_cap``) the
        averaged rate equals exactly the midpoint of the per-share
        schedule and the cap is invisible. On larger trades the
        sell-side TAF is clamped at ``finra_taf_cap`` while the
        per-share term would have grown linearly, so the averaged
        rate is BELOW the would-be midpoint of the uncapped
        schedule — the average is no longer a simple sum of the
        per-share rates. Worked example at the default 2026
        schedule (``finra_taf_per_share=0.000195``, cap ``$9.79``):
        the cap binds at ``size ≈ 9.79 / 0.000195 ≈ 50,205`` shares.
        For ``size=100_000`` at ``price=100`` the sell-side TAF is
        ``9.79`` (capped) rather than the uncapped ``19.50``; the
        ``side="average"`` rate is therefore ``4.895/notional``
        smaller per share of TAF contribution than the
        cap-unaware estimate. Callers running representative-cost
        analytics on large blocks should query ``side="sell"``
        explicitly rather than treat ``side="average"`` as a simple
        scaling of the per-share schedule.
        """
        broker_commission = max(
            size * self.commission_per_share, self.min_commission
        )
        if side == "sell":
            sec_fee = price * size * self.sec_fee_rate
            finra_taf = min(
                size * self.finra_taf_per_share, self.finra_taf_cap
            )
            return broker_commission + sec_fee + finra_taf
        return broker_commission

    def __repr__(self):
        return (
            f"USStockFeeModel(per_share=${self.commission_per_share:.4f}, "
            f"min=${self.min_commission:.2f}, "
            f"sec=${self.sec_fee_rate * 1e6:.2f}/$1M "
            f"[{_REPR_SEC_SCHEDULE_LABEL}], "
            f"finra_taf=${self.finra_taf_per_share:.6f}/share "
            f"cap=${self.finra_taf_cap:.2f} [{_REPR_FINRA_SCHEDULE_LABEL}])"
        )


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
