# Event Label Specification

This specification defines the taxonomy used to generate point labels for the SP500 research dataset.

## Event Taxonomy

The paper-level labels are:

- `jump`: positive return shock
- `drop`: negative return shock
- `volume_spike`: abnormal trading-volume surge
- `volatility_shock`: sudden increase in realized volatility
- `regime_shift`: persistent structural change in return/volatility behavior

## Window Parameters

The generator is parameterized by a main lookback window `W`.

Default configuration:

- `W = 100`
- `short_window = max(5, round(W / 5))`
- `regime_persistence = max(3, round(W / 20))`
- `return_z = 2.5`
- `volume_z = 2.5`
- `volatility_z = 2.0`
- `regime_mean_z = 1.5`
- `regime_vol_ratio = 1.5`

The output dataset name should include `W`, for example `SP500_event_taxonomy_w100`.

## Labeling Rules

All rolling statistics are computed using only historical data up to `t-1`.

Let `r_t = log(Close_t / Close_{t-1})` and `v_t = log1p(Volume_t)`.

| Event | Rule |
| --- | --- |
| `jump` | $r_t > \mu_r^{(W)}(t-1) + z_r \sigma_r^{(W)}(t-1)$ |
| `drop` | $r_t < \mu_r^{(W)}(t-1) - z_r \sigma_r^{(W)}(t-1)$ |
| `volume_spike` | $v_t > \mu_v^{(W)}(t-1) + z_v \sigma_v^{(W)}(t-1)$ |
| `volatility_shock` | $\mathrm{RV}_t > \mu_{RV}^{(W)}(t-1) + z_{RV} \sigma_{RV}^{(W)}(t-1)$ |
| `regime_shift` | Persistent deviation where short-window mean return and short-window realized volatility both deviate materially from the long-window baseline for at least `regime_persistence` days |

Where:

- $\mu^{(W)}$ and $\sigma^{(W)}$ are rolling mean and standard deviation over the past `W` trading days.
- `RV` is realized volatility estimated from the short window of absolute returns.

## Dataset Output

Each generated label file contains:

- `Date`
- `jump`
- `drop`
- `volume_spike`
- `volatility_shock`
- `regime_shift`
- `is_anomaly`

The composite anomaly target is `is_anomaly = jump OR drop OR volume_spike OR volatility_shock OR regime_shift`.

## Compatibility Note

The existing pipeline only requires binary label columns, so these event labels can be used directly.
The internal `criterion_a/b/c` columns are not required for the taxonomy dataset and should not be used as the primary paper labels.
