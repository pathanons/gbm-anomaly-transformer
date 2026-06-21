# Research Conclusion 1: Student-t Next-Day NLL เป็นสัญญาณ anomaly ที่ใช้ต่อได้

วันที่สรุป: 2026-06-20

## ข้อสรุปหลัก

Student-t next-day NLL score ใช้เป็น anomaly-ranking signal ได้จริงในชุดทดลอง fast weak-label นี้

เหตุผลคือเมื่อใช้ threshold แบบ validation quantile แล้วนำไปตัด test score จุดที่โมเดล flag ว่า anomaly มี overlap กับวันที่ endpoint log-return หลุดเส้น split-specific +/-3 std อย่างชัดเจน โดยเฉพาะบริเวณ operating point แถว `delta=0.02` ถึง `delta=0.015`

ข้อสรุปนี้ควรใช้เป็น evidence สำหรับเดินหน้าต่อเรื่อง thresholding, reporting, และ anomaly visualization ได้แน่นอน แต่ยังไม่ควรใช้เป็น final claim ว่าโมเดลชนะ baseline ทั้งหมด เพราะ ground truth ตอนนี้ยังเป็น weak endpoint label จาก log-return +/-3 std ไม่ใช่ external event label และยังไม่มี multi-seed/baseline/statistical test ครบ

## Experimental Setup

- Model signal: Student-t next-day NLL score
- Input window: 60 trading days
- Target: next-day log return
- Score rule: higher NLL means the realized next-day return is less likely under the predicted Student-t distribution
- Threshold protocol: threshold fitted from validation score quantile, then applied to test score
- Weak ground truth: endpoint log-return outside split-specific mean +/- 3 std
- Test points: 111,988
- Weak endpoint anomalies: 1,488

## Validation-Quantile Threshold Results

| delta | quantile | validation threshold | predicted anomalies | correct overlaps | false alarms | missed weak anomalies | precision | recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | Q95 | -0.788603 | 5,813 | 1,472 | 4,341 | 16 | 25.32% | 98.92% |
| 0.03 | Q97 | -0.200252 | 3,603 | 1,399 | 2,204 | 89 | 38.83% | 94.02% |
| 0.02 | Q98 | 0.334968 | 2,401 | 1,280 | 1,121 | 208 | 53.31% | 86.02% |
| 0.015 | Q98.5 | 0.734406 | 1,783 | 1,155 | 628 | 333 | 64.78% | 77.62% |
| 0.01 | Q99 | 1.243551 | 1,238 | 975 | 263 | 513 | 78.76% | 65.52% |
| 0.005 | Q99.5 | 2.140344 | 682 | 622 | 60 | 866 | 91.20% | 41.80% |

## Interpretation

The score behaves like a usable ranking signal:

- Lower quantile cutoffs such as `delta=0.05` and `delta=0.03` catch most weak anomalies but create many false alarms.
- Higher quantile cutoffs such as `delta=0.01` and `delta=0.005` give higher precision but miss many weak anomalies.
- `delta=0.02` and `delta=0.015` are the most useful current candidates for follow-up because they keep recall meaningful while reducing false alarms substantially.

Operationally:

- `delta=0.02` is better when recall matters more: 1,280/1,488 weak anomalies caught, recall 86.02%.
- `delta=0.015` is better when fewer false alarms matter more: precision 64.78%, recall 77.62%.

## Evidence Artifacts

Main summaries:

- `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/validation_quantile_threshold_summary.csv`
- `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/validation_quantile_confusion_counts.csv`

Visual evidence folders:

- `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_validation_quantile_delta_0p05`
- `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_validation_quantile_delta_0p03`
- `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_validation_quantile_delta_0p02`
- `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_validation_quantile_delta_0p015`
- `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_validation_quantile_delta_0p01`
- `D:/AnomalyTransformerRuns/experiments/phase4_nextday_nll_w60_fast_compare/figures/overlap_circled_validation_quantile_delta_0p005`

Each visual folder contains:

- `detected_anomalies.csv`
- `overlap_anomalies.csv`
- `summary_by_ticker.csv`
- `summary.txt`
- 109 per-ticker plots with circled overlaps

## Limitations

- The weak label is endpoint log-return outside split-specific +/-3 std, not a curated external market-event label.
- The test weak-label thresholds are descriptive labels, so final real-time evaluation should use a locked labeling/evaluation protocol.
- These results come from a fast 3-epoch run and should be repeated with multi-seed experiments.
- This is not yet a baseline-superiority claim.

## Next Actions

1. Promote `delta=0.02` and `delta=0.015` as candidate operating points for the next evaluation batch.
2. Compare against Gaussian NLL, rolling z-score, and GARCH-t under the same validation-quantile protocol.
3. Add event-level recall, detection delay, and false alarms per year.
4. Repeat with multiple seeds before making paper-level claims.
