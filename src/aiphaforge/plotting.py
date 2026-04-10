"""
Visualization
=============

Built-in plotting for backtest results. Requires matplotlib (optional).
Not imported by the main package — use directly::

    from aiphaforge.plotting import plot_result, plot_comparison

Install with: ``pip install aiphaforge[plot]``
"""

from typing import Any, Dict, Optional

import pandas as pd


def _get_plt():
    """Import matplotlib with Agg backend. Raises ImportError if missing."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib required for plotting. "
            "Install with: pip install aiphaforge[plot]")


def plot_result(
    result,
    benchmark: bool = True,
    figsize: tuple = (14, 10),
    title: Optional[str] = None,
):
    """Plot backtest result: equity curve, drawdown, monthly heatmap.

    Parameters:
        result: BacktestResult object.
        benchmark: Whether to overlay benchmark equity curve.
        figsize: Figure size.
        title: Optional figure title.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _get_plt()

    has_benchmark = (benchmark and result.benchmark_equity is not None)
    has_multi_asset = (result.per_asset_pnl is not None
                       and len(result.per_asset_pnl) > 1)
    n_panels = 2 + (1 if has_multi_asset else 0)

    fig, axes = plt.subplots(
        n_panels, 1, figsize=figsize,
        gridspec_kw={'height_ratios': [3, 1] + ([1] if has_multi_asset else [])})

    if n_panels == 2:
        axes = list(axes)

    fig_title = title or f"Backtest: {result.strategy_name}"
    fig.suptitle(fig_title, fontsize=14, fontweight='bold')

    # --- Panel 1: Equity Curve ---
    ax_eq = axes[0]
    ax_eq.plot(result.equity_curve, label='Strategy', linewidth=1.5)
    if has_benchmark:
        ax_eq.plot(result.benchmark_equity,
                   label=result.benchmark_name,
                   linewidth=1, linestyle='--', alpha=0.7)
    ax_eq.set_ylabel('Equity')
    ax_eq.legend(loc='upper left')
    ax_eq.grid(True, alpha=0.3)

    # Annotate key metrics
    metrics_text = (
        f"Return: {result.total_return * 100:+.1f}%  "
        f"Sharpe: {result.sharpe_ratio:.2f}  "
        f"MaxDD: {result.max_drawdown * 100:.1f}%  "
        f"Trades: {result.num_trades}"
    )
    ax_eq.set_title(metrics_text, fontsize=10, loc='left', color='gray')

    # --- Panel 2: Drawdown ---
    ax_dd = axes[1]
    equity = result.equity_curve
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    ax_dd.fill_between(drawdown.index, drawdown, 0,
                       alpha=0.4, color='red', label='Drawdown')
    ax_dd.set_ylabel('Drawdown')
    ax_dd.set_xlabel('Date')
    ax_dd.grid(True, alpha=0.3)

    # --- Panel 3: Per-asset PnL (multi-asset only) ---
    if has_multi_asset:
        ax_pnl = axes[2]
        pnl_df = pd.DataFrame(result.per_asset_pnl)
        cumulative = pnl_df.cumsum()
        for col in cumulative.columns:
            ax_pnl.plot(cumulative[col], label=col, linewidth=1)
        ax_pnl.set_ylabel('Cumulative PnL')
        ax_pnl.legend(loc='upper left', fontsize=8)
        ax_pnl.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison(
    results: Dict[str, Any],
    figsize: tuple = (12, 6),
    title: Optional[str] = None,
):
    """Compare multiple backtest results on one chart.

    Parameters:
        results: Dict mapping strategy name to BacktestResult.
        figsize: Figure size.
        title: Optional figure title.

    Returns:
        matplotlib.figure.Figure
    """
    plt = _get_plt()

    fig, ax = plt.subplots(figsize=figsize)

    for name, result in results.items():
        label = f"{name} ({result.total_return * 100:+.1f}%)"
        ax.plot(result.equity_curve, label=label, linewidth=1.2)

    ax.set_ylabel('Equity')
    ax.set_xlabel('Date')
    ax.set_title(title or 'Strategy Comparison', fontsize=13)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
