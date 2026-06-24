"""Probabilistic and Deflated Sharpe Ratio (Bailey & Lopez de Prado)."""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..results import BacktestResult


@dataclass
class PSRResult:
    """Probabilistic Sharpe Ratio result.

    Attributes:
        observed_sharpe: Annualised Sharpe of the input series.
        benchmark_sharpe: Annualised Sharpe threshold (SR*).
        psr: Probability that the true Sharpe exceeds benchmark_sharpe,
            given the observed series. Range [0, 1] or NaN on
            degenerate input.
        skewness: Sample skewness of returns (Fisher).
        kurtosis: Pearson (non-excess) kurtosis of returns; 3 for
            standard normal.
        n_obs: Number of observations.
    """
    observed_sharpe: float
    benchmark_sharpe: float
    psr: float
    skewness: float
    kurtosis: float
    n_obs: int


@dataclass
class DSRResult:
    """Deflated Sharpe Ratio result.

    Attributes:
        observed_sharpe: Annualised Sharpe of the input series.
        expected_max_null_sharpe: E[max SR] under the null when
            ``n_trials`` independent strategies are tested
            (Bailey & Lopez de Prado 2014, eq. 7), annualised.
        dsr: Probability the observed Sharpe exceeds the expected max
            null. Range [0, 1] or NaN on degenerate input.
        n_trials: Number of strategy trials supplied.
    """
    observed_sharpe: float
    expected_max_null_sharpe: float
    dsr: float
    n_trials: int


def _resolve_returns_and_td(
    source: Union[BacktestResult, pd.Series],
    trading_days: Optional[int],
) -> Tuple[pd.Series, int]:
    """Resolution: explicit kwarg → result.trading_days → 252."""
    if isinstance(source, BacktestResult):
        rets = source.equity_curve.pct_change().dropna()
        td = trading_days if trading_days is not None else int(
            getattr(source, "trading_days", 252))
    elif isinstance(source, pd.Series):
        rets = source.dropna()
        td = trading_days if trading_days is not None else 252
    else:
        raise TypeError(
            f"source must be BacktestResult or pd.Series, "
            f"got {type(source).__name__}")
    return rets, int(td)


def probabilistic_sharpe_ratio(
    source: Union[BacktestResult, pd.Series],
    *,
    benchmark_sharpe: float = 0.0,
    trading_days: Optional[int] = None,
    std_ddof: int = 0,
) -> PSRResult:
    """Bailey & Lopez de Prado 2012, eq. 14.

    PSR(SR*) = Φ((SR_obs − SR*) · √(T−1) /
                  √(1 − γ₃·SR_obs + (γ₄−1)/4·SR_obs²))

    where SR_obs and SR* are *per-period* (non-annualised) Sharpe ratios,
    γ₃ is sample skewness, γ₄ is **Pearson** kurtosis (3 for normal).
    The function takes ``benchmark_sharpe`` as an *annualised* value and
    converts internally.

    Parameters:
        source: A BacktestResult (uses its equity_curve.pct_change()),
            or a pd.Series of per-period returns.
        benchmark_sharpe: Annualised benchmark Sharpe threshold (SR*).
        trading_days: Annualisation factor. Resolution order:
            1. If passed (not None), use it.
            2. Else if source is BacktestResult, use ``source.trading_days``.
            3. Else 252.
        std_ddof: Degrees-of-freedom divisor for the SR denominator's σ.
            v2.2.2 default is ``0`` (biased σ, dividing by T), which is
            the canonical Bailey-LdP eq. 14 form — the √(T-1) factor in
            the z-statistic numerator is the variance standardization
            paired with the biased SR estimator. v2.2.1 and earlier
            silently used ``ddof=1`` (pandas default), which paired
            unbiased σ with the √(T-1) standardization — an internal
            inconsistency. Users who pinned v2.2.1 numerical values
            can opt out by passing ``std_ddof=1``. Numeric shift is
            ~0.2% at T=252 and ~2% at T=20.

    Returns:
        PSRResult. ``psr`` is NaN when:
            - n < 4 observations, or
            - returns std == 0, or
            - the variance denominator becomes non-positive (rare,
              extreme skew/kurt combos).

    Example:
        >>> from aiphaforge.significance import probabilistic_sharpe_ratio
        >>> r = probabilistic_sharpe_ratio(result, benchmark_sharpe=1.0)
        >>> if r.psr > 0.95:
        ...     print(f"Sharpe {r.observed_sharpe:.2f} robust at 95% level")
    """
    from scipy import stats  # lazy import — scipy is an optional dep elsewhere

    rets, td = _resolve_returns_and_td(source, trading_days)
    n = len(rets)
    sqrt_td = np.sqrt(td)

    def _nan_result():
        return PSRResult(
            observed_sharpe=float("nan"), benchmark_sharpe=benchmark_sharpe,
            psr=float("nan"), skewness=float("nan"),
            kurtosis=float("nan"), n_obs=n,
        )

    if n < 4:
        return _nan_result()
    std = float(rets.std(ddof=std_ddof))
    # Guard against ~constant series where std is finite but vanishing
    # (floating-point noise), which makes sr_per blow up and the
    # downstream skew / kurtosis degenerate.
    if not np.isfinite(std) or std < 1e-12:
        return _nan_result()

    sr_per = float(rets.mean() / std)
    benchmark_per = benchmark_sharpe / sqrt_td

    skew = float(stats.skew(rets, bias=False))
    kurt_pearson = float(stats.kurtosis(rets, fisher=False, bias=False))

    denom = 1.0 - skew * sr_per + ((kurt_pearson - 1.0) / 4.0) * (sr_per ** 2)
    if not np.isfinite(denom) or denom <= 0:
        # Degenerate tail / skew; PSR is undefined. Preserve the valid
        # per-period Sharpe as observed_sharpe (annualised) so UI can
        # still show it, but *only* when it is finite.
        obs = sr_per * sqrt_td if np.isfinite(sr_per) else float("nan")
        return PSRResult(
            observed_sharpe=obs,
            benchmark_sharpe=benchmark_sharpe,
            psr=float("nan"),
            skewness=skew if np.isfinite(skew) else float("nan"),
            kurtosis=kurt_pearson if np.isfinite(kurt_pearson) else float("nan"),
            n_obs=n,
        )

    z = (sr_per - benchmark_per) * np.sqrt(n - 1) / np.sqrt(denom)
    psr = float(stats.norm.cdf(z))
    return PSRResult(
        observed_sharpe=sr_per * sqrt_td,
        benchmark_sharpe=benchmark_sharpe,
        psr=psr, skewness=skew, kurtosis=kurt_pearson, n_obs=n,
    )


def deflated_sharpe_ratio(
    source: Union[BacktestResult, pd.Series],
    *,
    n_trials: int,
    trading_days: Optional[int] = None,
    std_ddof: int = 0,
) -> DSRResult:
    """Bailey, Borwein, López de Prado & Zhu 2014.

    Estimates E[max Sharpe] under the null when ``n_trials`` strategies
    are tested, then computes PSR against that elevated bar. Same
    ``trading_days`` resolution order as
    :func:`probabilistic_sharpe_ratio`.

    Parameters:
        source: A BacktestResult or pd.Series of per-period returns.
        n_trials: Number of strategy variants tested before selection.
        trading_days: Annualisation factor (see resolution order above).
        std_ddof: σ degrees-of-freedom; default ``0`` per Bailey-LdP
            canonical form. See :func:`probabilistic_sharpe_ratio` for
            the rationale and the v2.2.2 numeric shift note.

    Returns:
        DSRResult. ``dsr`` is NaN when:
            - n < 4 observations, or
            - n_trials < 1, or
            - returns std == 0.

    Note on autocorrelation:
        The variance formula at eq. 6 assumes i.i.d. returns. Lo (2002)
        gives an autocorrelation correction; this implementation does
        NOT apply it. For strongly autocorrelated return series (e.g.
        overlapping-window strategies, intraday at <1min) the DSR
        understates uncertainty. Flagged for v2.3.
    """
    import math

    from scipy import stats

    rets, td = _resolve_returns_and_td(source, trading_days)
    n = len(rets)
    sqrt_td = np.sqrt(td)

    # n_trials < 2 is mathematically undefined: Bailey 2014 eq.7 has
    # stats.norm.ppf(1 - 1/N), which at N=1 is ppf(0) = -∞ and would
    # trivially return PSR = 1.0 (always "significant"). We return NaN
    # to signal DSR is meaningless; for single-strategy significance
    # the caller should use ``probabilistic_sharpe_ratio`` directly.
    if n < 4 or n_trials < 2:
        return DSRResult(
            observed_sharpe=float("nan"),
            expected_max_null_sharpe=float("nan"),
            dsr=float("nan"), n_trials=int(n_trials),
        )

    std = float(rets.std(ddof=std_ddof))
    if not np.isfinite(std) or std < 1e-12:
        return DSRResult(
            observed_sharpe=float("nan"),
            expected_max_null_sharpe=float("nan"),
            dsr=float("nan"), n_trials=int(n_trials),
        )

    sr_per = float(rets.mean() / std)
    sr_obs_ann = sr_per * sqrt_td

    # Bailey & Lopez de Prado 2014 eq. 6: standard deviation of the Sharpe
    # *estimator* itself (NOT of the returns). Per-period formula includes
    # higher-moment corrections; simplifies to 1/sqrt(T-1) under normality.
    skew = float(stats.skew(rets, bias=False))
    kurt_pearson = float(stats.kurtosis(rets, fisher=False, bias=False))
    # n >= 4 guaranteed by the early-exit above, so (n-1) >= 3
    var_sr_per = (1.0 - skew * sr_per
                  + (kurt_pearson - 1.0) / 4.0 * sr_per ** 2) / (n - 1)
    if not np.isfinite(var_sr_per) or var_sr_per <= 0:
        return DSRResult(
            observed_sharpe=sr_obs_ann,
            expected_max_null_sharpe=float("nan"),
            dsr=float("nan"), n_trials=int(n_trials),
        )
    sd_sr_per = np.sqrt(var_sr_per)
    sd_sr_ann = sd_sr_per * sqrt_td

    # Bailey & Lopez de Prado 2014 eq. 7: expected max Sharpe under N trials
    gamma = 0.5772156649  # Euler-Mascheroni
    e = math.e
    sr_zero_ann = sd_sr_ann * (
        (1.0 - gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    )

    psr_result = probabilistic_sharpe_ratio(
        rets, benchmark_sharpe=sr_zero_ann, trading_days=td)

    return DSRResult(
        observed_sharpe=sr_obs_ann,
        expected_max_null_sharpe=sr_zero_ann,
        dsr=psr_result.psr,
        n_trials=int(n_trials),
    )
