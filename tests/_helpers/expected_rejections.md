# `validate_signal_series` rejection set (v2.8.2 audit)

Authoritative enumeration of input shapes the v2.8.1
`aiphaforge.signals.validate_signal_series` rejects under its
**default** `allow_fractional=True`. This file is the source of
truth for the v2.8.2 Commit C boundary-validation work and the
README v2.8.2 release-notes false-positive enumeration.

Audited against: `src/aiphaforge/signals.py:186-244` at HEAD
`c4a5004` (post-v2.8.2 Commit B).

## Rejected shapes

| Shape | Exception | Trigger |
|---|---|---|
| Non-`DatetimeIndex` (e.g. `RangeIndex`, `Int64Index`, MultiIndex) | `TypeError` | `signal.index` is not a `pd.DatetimeIndex` |
| Duplicate `DatetimeIndex` timestamps | `ValueError` | `signal.index.has_duplicates` |
| Non-numeric dtype (e.g. `object`, `string`, `datetime`) | `TypeError` | `pd.api.types.is_numeric_dtype(signal)` False |

## NOT rejected (passes default validation)

These shapes are common-sense candidates but are **NOT** rejected by
`validate_signal_series` and may slip through:

- All-NaN Series (validator only checks dtype, not content)
- Non-monotonic / out-of-order `DatetimeIndex` (validator does not
  check ordering — downstream may complain)
- Empty Series (passes; trivially has no duplicates / no values)
- Fractional values when `allow_fractional=True` (the default)

## Notes for v2.8.2 Commit C

- `set_signals` calls validate at the boundary with the default
  `allow_fractional=True` — matches the engine's existing tolerance
  for fractional / target-weight signals.
- For per-symbol dict input, each value is validated; the catch
  re-raises with the offending symbol's name in the message.
- The escape hatch is `BacktestEngine(data_validation="none")` —
  matches the `validate_ohlcv` convention.

If a future commit extends `validate_signal_series` to reject
additional shapes (e.g. out-of-order index), this file MUST be
updated synchronously with the README enumeration in `Commit H`.
