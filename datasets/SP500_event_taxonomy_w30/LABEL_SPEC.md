# Event Label Specification

This dataset was generated with a lookback window of 30 trading days.

## Parameters
- lookback_window = 30
- short_window = 6
- regime_persistence = 3
- return_z = 2.5
- volume_z = 2.5
- volatility_z = 2.0
- regime_mean_z = 1.5
- regime_vol_ratio = 1.5

## Label Definitions
- jump: positive return shock relative to the trailing 30-day baseline
- drop: negative return shock relative to the trailing 30-day baseline
- volume_spike: abnormal log-volume surge relative to the trailing 30-day baseline
- volatility_shock: realized-volatility surge relative to the trailing 30-day baseline
- regime_shift: persistent deviation in short-window return mean and realized volatility for at least 3 days

## Notes
- All baselines use only historical data up to t-1.
- The composite anomaly label is the OR of all event columns.
- The dataset keeps the event labels at point level, so each day can have multiple concurrent events.
