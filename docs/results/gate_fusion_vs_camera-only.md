# Validation report: fusion (PROMOTE)

- dataset version: `969966aae5426c70` (948 test frames)
- candidate weights: `bc56bf03685694a6` trained on `969966aae5426c70`
- baseline: camera-only `f4a32645a8f28ab2`

| check | value / delta | 95% CI | result |
|---|---|---|---|
| min_mAP | 0.673 |  | pass |
| lineage_complete |  |  | pass |
| no_regression[mAP] | +0.050 | [+0.029, +0.070] | pass |
| no_regression[AP/vehicle] | +0.097 | [+0.066, +0.124] | pass |
| no_regression[AP/person] | +0.063 | [+0.028, +0.104] | pass |
| no_regression[AP/building] | -0.009 | [-0.046, +0.026] | pass |
| no_regression[lighting=day/mAP] | -0.000 | [-0.027, +0.028] | pass |
| no_regression[lighting=dusk/mAP] | +0.049 | [-0.000, +0.093] | pass |
| no_regression[lighting=night/mAP] | +0.085 | [+0.054, +0.112] | pass |
| no_regression[fog=heavy/mAP] | +0.230 | [+0.145, +0.298] | pass |
| no_regression[fog=light/mAP] | -0.005 | [-0.052, +0.053] | pass |
| no_regression[fog=none/mAP] | +0.043 | [+0.020, +0.064] | pass |
| no_regression[altitude_band=high/mAP] | +0.016 | [-0.008, +0.038] | pass |
| no_regression[altitude_band=low/mAP] | +0.297 | [+0.216, +0.357] | pass |
| no_regression[altitude_band=mid/mAP] | +0.020 | [-0.008, +0.047] | pass |
