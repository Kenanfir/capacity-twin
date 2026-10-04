| policy                             | level         |   recall |   recall_ci_lo |   recall_ci_hi |   false_alarm_rate |   margin |
|:-----------------------------------|:--------------|---------:|---------------:|---------------:|-------------------:|---------:|
| trend + conformal (paper)          | P0_unsurveyed | 0        |       0        |       0        |         0          |       39 |
| no-trend + conformal               | P0_unsurveyed | 0        |       0        |       0        |         0          |       27 |
| trend + conformal (paper)          | P1_low        | 0.737179 |       0.432168 |       0.949497 |         0.0646876  |       39 |
| no-trend + conformal               | P1_low        | 0.455128 |       0.159474 |       0.795776 |         0.0327367  |       27 |
| trend + conformal (paper)          | P2_medium     | 0.878205 |       0.701408 |       0.99194  |         0.031545   |       39 |
| no-trend + conformal               | P2_medium     | 0.628205 |       0.252504 |       0.898102 |         0.00902904 |       27 |
| trend + conformal (paper)          | P3_high       | 1        |       1        |       1        |         0.0206799  |       39 |
| no-trend + conformal               | P3_high       | 0.814103 |       0.66964  |       0.939607 |         0.00311273 |       27 |
| oracle (true geometry) + conformal | P_oracle      | 1        |       1        |       1        |         0.0198138  |       39 |