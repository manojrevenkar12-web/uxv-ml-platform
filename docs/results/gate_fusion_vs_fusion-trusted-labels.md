# Validation report: fusion (PROMOTE)

- dataset version: `969966aae5426c70` (948 test frames)
- candidate weights: `bc56bf03685694a6` trained on `969966aae5426c70`
- baseline: fusion-trusted-labels `32d5b294f77852a8`

| check | value / delta | 95% CI | result |
|---|---|---|---|
| min_mAP | 0.673 |  | pass |
| lineage_complete |  |  | pass |
| no_regression[mAP] | +0.037 | [+0.023, +0.051] | pass |
| no_regression[AP/vehicle] | +0.010 | [-0.007, +0.028] | pass |
| no_regression[AP/person] | +0.101 | [+0.070, +0.133] | pass |
| no_regression[AP/building] | -0.000 | [-0.021, +0.019] | pass |
| no_regression[lighting=day/mAP] | +0.033 | [+0.004, +0.065] | pass |
| no_regression[lighting=dusk/mAP] | +0.028 | [-0.004, +0.063] | pass |
| no_regression[lighting=night/mAP] | +0.046 | [+0.030, +0.065] | pass |
| no_regression[fog=heavy/mAP] | +0.052 | [+0.019, +0.092] | pass |
| no_regression[fog=light/mAP] | +0.029 | [-0.010, +0.070] | pass |
| no_regression[fog=none/mAP] | +0.032 | [+0.014, +0.051] | pass |
| no_regression[altitude_band=high/mAP] | -0.025 | [-0.047, -0.002] | pass |
| no_regression[altitude_band=low/mAP] | +0.049 | [+0.017, +0.090] | pass |
| no_regression[altitude_band=mid/mAP] | +0.080 | [+0.063, +0.099] | pass |
