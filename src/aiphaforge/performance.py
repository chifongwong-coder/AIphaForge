"""
Performance Analyzer

Provides comprehensive backtest performance analysis and reporting.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .results import BacktestResult

# Import utility functions
from .utils import (
    TRADING_DAYS_STOCK,
    annualize,
    annualize_return,
    calculate_returns,
)
from .utils import (
    sharpe_ratio as calc_sharpe,
)
from .utils import (
    sortino_ratio as calc_sortino,
)


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for backtest results.

    Attributes:
        result: Backtest result object.
        equity: Equity curve.
        returns: Return series.
        trades: Trade records.

    Example:
        >>> analyzer = PerformanceAnalyzer(result)
        >>> print(analyzer.summary())
        >>> report = analyzer.generate_report()
    """

    def __init__(self, result: BacktestResult, downside_method: str = "full"):
        self.result = result
        self.equity = result.equity_curve
        self.returns = calculate_returns(self.equity) if len(self.equity) > 1 else pd.Series()
        self.trades = result.trades
        self._downside_method = downside_method

        # Cached results
        self._metrics_cache = {}

    # ========== Return Metrics ==========

    @property
    def total_return(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        first_val = self.equity.iloc[0]
        last_val = self.equity.iloc[-1]
        if np.isnan(first_val) or np.isnan(last_val) or np.isinf(first_val) or np.isinf(last_val) or first_val == 0:
            return 0.0
        return (last_val / first_val) - 1

    @property
    def annualized_return(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        n_days = len(self.returns)
        if n_days == 0:
            return 0.0
        total_ret = self.total_return
        return annualize_return(total_ret, n_days, TRADING_DAYS_STOCK)

    @property
    def cagr(self) -> float:
        """Compound annual growth rate."""
        return self.annualized_return

    def monthly_returns(self) -> pd.Series:
        if len(self.equity) < 2:
            return pd.Series()
        monthly = self.equity.resample('ME').last()
        return monthly.pct_change().dropna()

    def yearly_returns(self) -> pd.Series:
        if len(self.equity) < 2:
            return pd.Series()
        yearly = self.equity.resample('YE').last()
        return yearly.pct_change().dropna()

    def rolling_returns(self, window: int = 252) -> pd.Series:
        if len(self.equity) < window:
            return pd.Series()
        return self.equity.pct_change(window)

    # ========== Risk Metrics ==========

    @property
    def volatility(self) -> float:
        """Annualized volatility."""
        if len(self.returns) < 2:
            return 0.0
        return annualize(self.returns.std(), TRADING_DAYS_STOCK, is_volatility=True)

    @property
    def downside_volatility(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        if self._downside_method == "negative_only":
            negative_returns = self.returns[self.returns < 0]
            if len(negative_returns) == 0:
                return 0.0
            daily_downside = np.sqrt((negative_returns ** 2).mean())
        else:
            # "full": root-mean-square of min(r, 0) over ALL observations
            clipped = np.minimum(self.returns, 0.0)
            daily_downside = np.sqrt((clipped ** 2).mean())
        if daily_downside == 0 or np.isnan(daily_downside):
            return 0.0
        return annualize(daily_downside, TRADING_DAYS_STOCK, is_volatility=True)

    @property
    def max_drawdown(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        rolling_max = self.equity.expanding().max()
        drawdown = (self.equity - rolling_max) / rolling_max
        return abs(drawdown.min())

    def drawdown_series(self) -> pd.Series:
        if len(self.equity) < 2:
            return pd.Series()
        rolling_max = self.equity.expanding().max()
        return (self.equity - rolling_max) / rolling_max

    @property
    def max_drawdown_duration(self) -> int:
        """Maximum drawdown duration in days."""
        if len(self.equity) < 2:
            return 0

        drawdown = self.drawdown_series()
        underwater = drawdown < 0

        # Find the longest continuous underwater period
        groups = (~underwater).cumsum()
        underwater_periods = underwater.groupby(groups).sum()

        return int(underwater_periods.max()) if len(underwater_periods) > 0 else 0

    def var(self, confidence: float = 0.95) -> float:
        """Value at Risk."""
        if len(self.returns) < 2:
            return 0.0
        return np.percentile(self.returns, (1 - confidence) * 100)

    def cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)."""
        if len(self.returns) < 2:
            return 0.0
        var = self.var(confidence)
        return self.returns[self.returns <= var].mean()

    # ========== Risk-Adjusted Return Metrics ==========

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio (assuming zero risk-free rate)."""
        return calc_sharpe(self.returns, trading_days=TRADING_DAYS_STOCK)

    @property
    def sortino_ratio(self) -> float:
        return calc_sortino(
            self.returns,
            trading_days=TRADING_DAYS_STOCK,
            downside_method=self._downside_method,
        )

    @property
    def calmar_ratio(self) -> float:
        if self.max_drawdown == 0:
            return 0.0
        annual_ret = self.annualized_return
        if np.isnan(annual_ret):
            return 0.0
        return annual_ret / self.max_drawdown

    def omega_ratio(self, threshold: float = 0.0) -> float:
        if len(self.returns) < 2:
            return 0.0

        excess = self.returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())

        if losses == 0:
            return float('inf') if gains > 0 else 0.0
        return gains / losses

    def information_ratio(self, benchmark_returns: pd.Series) -> float:
        if len(self.returns) < 2:
            return 0.0

        aligned = pd.concat([self.returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 0.0

        excess = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        if excess.std() == 0:
            return 0.0

        return annualize(excess.mean() / excess.std(), TRADING_DAYS_STOCK, is_volatility=True)

    # ========== Trade Statistics ==========

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if self.num_trades == 0:
            return 0.0
        winners = sum(1 for t in self.trades if t.pnl > 0)
        return winners / self.num_trades

    @property
    def profit_factor(self) -> float:
        if not self.trades:
            return 0.0

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def avg_trade_pnl(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl for t in self.trades) / len(self.trades)

    @property
    def avg_win(self) -> float:
        winners = [t.pnl for t in self.trades if t.pnl > 0]
        return np.mean(winners) if winners else 0.0

    @property
    def avg_loss(self) -> float:
        losers = [t.pnl for t in self.trades if t.pnl < 0]
        return np.mean(losers) if losers else 0.0

    @property
    def largest_win(self) -> float:
        if not self.trades:
            return 0.0
        return max(t.pnl for t in self.trades)

    @property
    def largest_loss(self) -> float:
        if not self.trades:
            return 0.0
        return min(t.pnl for t in self.trades)

    @property
    def avg_holding_period(self) -> float:
        """Average holding period in days."""
        if not self.trades:
            return 0.0
        holding_days = [(t.exit_time - t.entry_time).total_seconds() / 86400 for t in self.trades]
        return np.mean(holding_days)

    @property
    def win_loss_ratio(self) -> float:
        if self.avg_loss == 0:
            return float('inf') if self.avg_win > 0 else 0.0
        return abs(self.avg_win / self.avg_loss)

    @property
    def expectancy(self) -> float:
        """Expected return per trade."""
        return self.win_rate * self.avg_win + (1 - self.win_rate) * self.avg_loss

    # ========== Monthly Analysis ==========

    def monthly_returns_table(self) -> pd.DataFrame:
        """Monthly returns table (year x month matrix)."""
        if len(self.equity) < 2:
            return pd.DataFrame()

        monthly = self.monthly_returns()
        if len(monthly) == 0:
            return pd.DataFrame()

        monthly.index = pd.to_datetime(monthly.index)
        table = monthly.to_frame('return')
        table['year'] = table.index.year
        table['month'] = table.index.month

        pivot = table.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

        # Add yearly total
        pivot['Year'] = (1 + pivot.fillna(0)).prod(axis=1) - 1

        return pivot

    def best_month(self) -> Tuple[str, float]:
        monthly = self.monthly_returns()
        if len(monthly) == 0:
            return ("N/A", 0.0)
        best_idx = monthly.idxmax()
        return (str(best_idx), monthly[best_idx])

    def worst_month(self) -> Tuple[str, float]:
        monthly = self.monthly_returns()
        if len(monthly) == 0:
            return ("N/A", 0.0)
        worst_idx = monthly.idxmin()
        return (str(worst_idx), monthly[worst_idx])

    def positive_months_pct(self) -> float:
        """Percentage of months with positive returns."""
        monthly = self.monthly_returns()
        if len(monthly) == 0:
            return 0.0
        return (monthly > 0).sum() / len(monthly)

    # ========== Summary Report ==========

    def summary(self) -> str:
        """Generate a text summary."""
        lines = [
            "=" * 60,
            f"Performance Report - {self.result.strategy_name}",
            "=" * 60,
            "",
            "[Returns]",
            f"  Total Return:        {self.total_return*100:+.2f}%",
            f"  Annualized Return:   {self.annualized_return*100:+.2f}%",
            f"  Ann. Volatility:     {self.volatility*100:.2f}%",
            "",
            "[Risk]",
            f"  Max Drawdown:        {self.max_drawdown*100:.2f}%",
            f"  Max DD Duration:     {self.max_drawdown_duration} days",
            f"  VaR(95%):            {self.var(0.95)*100:.2f}%",
            f"  CVaR(95%):           {self.cvar(0.95)*100:.2f}%",
            "",
            "[Risk-Adjusted Returns]",
            f"  Sharpe Ratio:        {self.sharpe_ratio:.3f}",
            f"  Sortino Ratio:       {self.sortino_ratio:.3f}",
            f"  Calmar Ratio:        {self.calmar_ratio:.3f}",
            "",
            "[Trade Statistics]",
            f"  Num Trades:          {self.num_trades}",
            f"  Win Rate:            {self.win_rate*100:.1f}%",
            f"  Profit Factor:       {self.profit_factor:.2f}",
            f"  Avg Win:             ${self.avg_win:+,.2f}",
            f"  Avg Loss:            ${self.avg_loss:+,.2f}",
            f"  Win/Loss Ratio:      {self.win_loss_ratio:.2f}",
            f"  Avg Holding Period:  {self.avg_holding_period:.1f} days",
            "",
            "[Monthly Statistics]",
            f"  Best Month:          {self.best_month()[0][:7]} ({self.best_month()[1]*100:+.2f}%)",
            f"  Worst Month:         {self.worst_month()[0][:7]} ({self.worst_month()[1]*100:+.2f}%)",
            f"  Positive Months:     {self.positive_months_pct()*100:.1f}%",
            "=" * 60
        ]
        return "\n".join(lines)

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a complete structured report.

        Returns:
            Dict: Report with all metrics organized by category.
        """
        return {
            'overview': {
                'strategy_name': self.result.strategy_name,
                'start_date': str(self.result.start_date),
                'end_date': str(self.result.end_date),
                'initial_capital': self.result.initial_capital,
                'final_capital': self.result.final_capital,
            },
            'returns': {
                'total_return': self.total_return,
                'annualized_return': self.annualized_return,
                'volatility': self.volatility,
                'downside_volatility': self.downside_volatility,
            },
            'risk': {
                'max_drawdown': self.max_drawdown,
                'max_drawdown_duration': self.max_drawdown_duration,
                'var_95': self.var(0.95),
                'cvar_95': self.cvar(0.95),
            },
            'risk_adjusted': {
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'calmar_ratio': self.calmar_ratio,
            },
            'trades': {
                'num_trades': self.num_trades,
                'win_rate': self.win_rate,
                'profit_factor': self.profit_factor,
                'avg_trade_pnl': self.avg_trade_pnl,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'largest_win': self.largest_win,
                'largest_loss': self.largest_loss,
                'win_loss_ratio': self.win_loss_ratio,
                'expectancy': self.expectancy,
                'avg_holding_period': self.avg_holding_period,
            },
            'monthly': {
                'best_month': self.best_month(),
                'worst_month': self.worst_month(),
                'positive_months_pct': self.positive_months_pct(),
            }
        }

    # ========== Visualization Data ==========

    def get_equity_curve_data(
        self,
        benchmark: Optional[pd.Series] = None
    ) -> Dict[str, pd.Series]:
        """Get equity curve data for plotting."""
        data = {'strategy': self.equity}

        if benchmark is not None:
            aligned_bench = benchmark.reindex(self.equity.index).ffill().bfill()
            normalized_bench = aligned_bench / aligned_bench.iloc[0] * self.result.initial_capital
            data['benchmark'] = normalized_bench

        return data

    def get_drawdown_data(self) -> pd.Series:
        return self.drawdown_series()

    def get_monthly_heatmap_data(self) -> pd.DataFrame:
        return self.monthly_returns_table()

    def get_trade_distribution_data(self) -> Dict[str, List]:
        if not self.trades:
            return {'pnl': [], 'pnl_pct': []}

        return {
            'pnl': [t.pnl for t in self.trades],
            'pnl_pct': [t.pnl_pct for t in self.trades]
        }

    # ========== Multi-Asset Analysis ==========

    def per_asset_analysis(self) -> Dict[str, Dict[str, Any]]:
        """Compute per-asset performance metrics from per_asset_pnl.

        The ``sharpe_ratio`` here is computed on dollar PnL deltas
        (not percentage returns), which is standard practice for
        shared-capital multi-asset portfolios where per-asset returns
        have no well-defined denominator.  This is sometimes called
        "dollar-PnL Sharpe" and is widely used in prop trading.

        Returns:
            Dict mapping symbol to a metrics dict with keys:
            ``total_pnl``, ``sharpe_ratio`` (dollar-PnL based),
            ``max_drawdown`` (dollar), ``volatility`` (dollar).
        """
        pnl_dict = self.result.per_asset_pnl
        if pnl_dict is None:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        for sym, pnl_series in pnl_dict.items():
            if len(pnl_series) == 0:
                results[sym] = {
                    'total_pnl': 0.0, 'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0, 'volatility': 0.0,
                }
                continue
            cumulative = pnl_series.cumsum()
            total_pnl = float(cumulative.iloc[-1])
            vol = float(pnl_series.std() * np.sqrt(TRADING_DAYS_STOCK))
            sr = calc_sharpe(pnl_series, trading_days=TRADING_DAYS_STOCK)
            # Drawdown from cumulative PnL curve
            peak = cumulative.expanding().max()
            dd = (cumulative - peak)
            mdd = float(abs(dd.min())) if len(dd) > 0 else 0.0
            results[sym] = {
                'total_pnl': total_pnl,
                'sharpe_ratio': sr,
                'max_drawdown': mdd,
                'volatility': vol,
            }
        return results

    def correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Pairwise PnL-change correlation matrix across assets.

        Returns:
            pd.DataFrame correlation matrix, or None if not multi-asset.
        """
        pnl_dict = self.result.per_asset_pnl
        if pnl_dict is None or len(pnl_dict) < 2:
            return None
        pnl_df = pd.DataFrame(pnl_dict)
        return pnl_df.corr()

    def __repr__(self):
        return (f"PerformanceAnalyzer(strategy={self.result.strategy_name}, "
                f"return={self.total_return*100:+.2f}%, "
                f"sharpe={self.sharpe_ratio:.2f})")


# ========== Convenience Functions ==========

def analyze(result: BacktestResult) -> PerformanceAnalyzer:
    """
    Create a performance analyzer (convenience function).

    Parameters:
        result: BacktestResult object.

    Returns:
        PerformanceAnalyzer instance.
    """
    return PerformanceAnalyzer(result)


def compare_strategies(
    results: Dict[str, BacktestResult]
) -> pd.DataFrame:
    """
    Compare multiple strategies side by side.

    Parameters:
        results: Dict mapping strategy names to BacktestResult objects.

    Returns:
        pd.DataFrame: Comparison table.
    """
    comparisons = []

    for name, result in results.items():
        analyzer = PerformanceAnalyzer(result)
        comparisons.append({
            'Strategy': name,
            'Total Return': f"{analyzer.total_return*100:.2f}%",
            'Annual Return': f"{analyzer.annualized_return*100:.2f}%",
            'Volatility': f"{analyzer.volatility*100:.2f}%",
            'Sharpe': f"{analyzer.sharpe_ratio:.2f}",
            'Max DD': f"{analyzer.max_drawdown*100:.2f}%",
            'Win Rate': f"{analyzer.win_rate*100:.1f}%",
            'Trades': analyzer.num_trades
        })

    return pd.DataFrame(comparisons).set_index('Strategy')
