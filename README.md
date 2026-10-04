# Capacity-twin: a synthetic simulation of capacity alerts under geometry uncertainty

Code, seed, and results behind the paper *Conformal-Calibrated Capacity
Alerts Under Geometry Uncertainty: A Decision-Support Simulation for Cultural
Heritage Site Management* (submitted to the Journal of Intelligent Decision
Making and Information Science).

**Everything here is synthetic.** Zone geometry, visitor arrivals, dwell
times, and queues are simulated. No real visitor, occupancy, survey, or
personal data was collected, and this code is not the code of any deployed
MUSERA system. It models a ten-zone site shaped like the Rumah Jawa Kediri
pilot (a gate, a performance venue, four exhibit zones, two vendor clusters, a
plaza hub, and an exit) without claiming real surveyed coordinates.

## What it does

For each of 1,800 independent simulated operating days ("worlds") it draws one
*true* synthetic geometry, runs a hybrid agent-based and discrete-event
simulation of visitor flow against it, and then generates a *survey estimate*
of that geometry at four precision levels (unsurveyed, low, medium, high). A
capacity-alert policy forecasts each zone's occupancy ten minutes ahead,
adds a split-conformal margin calibrated on 900 worlds, and is scored on 900
held-out worlds against the world's own true safety threshold. Comparators:
a zero-margin policy, a no-trend predictor, and an oracle given the true
geometry.

## Run it

```bash
pip install -r requirements.txt
python3 build.py --n-runs 1800 --check
```

This reruns the experiment, regenerates every table and figure, and gates on
the test suite, exact reproducibility of the raw runs across two full runs,
finite occupancy values, and the presence of every figure and table. It exits
0 only if all gates pass. It takes about 6 minutes on a 14-core laptop
(simulation days run in parallel; use `--jobs 1` for the serial path, which
gives identical output). The master seed is `20260918`; each world's draws
depend only on that seed and the world's index.

## Files

| Path | Content |
|---|---|
| `sim/` | arrival process, zones and geometry, simulation engine, metrics, alert policy and conformal calibration |
| `run_experiment.py` | runs the experiment grid and policy evaluation, with bootstrap confidence intervals |
| `analysis/make_outputs.py` | builds the tables and figures |
| `tests/` | unit tests for the arrival process, engine, and policy |
| `results/` | committed outputs of the run above |
| `figures/` | the four generated figures |

## Results files and the paper

| File | Paper |
|---|---|
| `results/summary_table.csv` | Table 4 |
| `results/policy_comparison_table.csv` | Tables 3 and 5 |
| `results/comparator_table.csv` | Table 6 |
| `results/per_zone_table.csv` | Table 7 |
| `results/policy_evaluation.csv` | all recall, precision, false-alarm rates and their 95% cluster-bootstrap intervals, per policy, level, and zone |
| `results/raw_runs.parquet` | per-world, per-level summary of every simulated day |
| `figures/occupancy_by_precision.png` | Fig. 3 |
| `figures/policy_recall_vs_precision.png` | Fig. 4 |
| `figures/case_study_timeseries.png` | Fig. 5 |
| `figures/methodology_diagram.png` | Fig. 2 (a schematic; its citation numbers refer to the paper's reference list) |

## Limits

The result rests on 13 worlds out of 900 that contain a genuine safety
breach, which is why every recall and false-alarm figure carries a confidence
interval. The geometry noise levels are design parameters, not measured survey
errors. See the paper's Limitations discussion.

## License and citation

MIT License (see `LICENSE`). Citation details are in `CITATION.cff`.
