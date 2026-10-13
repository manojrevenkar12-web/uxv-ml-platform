# Validation report: fusion-nosync (PROMOTE)

- dataset version: `969966aae5426c70` (948 test frames)
- candidate weights: `850ff160382928b4` trained on `f67539b7a2b242b7`
- baseline: fusion `bc56bf03685694a6`

| check | value / delta | 95% CI | result |
|---|---|---|---|
| min_mAP | 0.690 |  | pass |
| lineage_complete |  |  | pass |
| no_regression[mAP] | +0.017 | [+0.005, +0.026] | pass |
| no_regression[AP/vehicle] | -0.022 | [-0.039, -0.008] | pass |
| no_regression[AP/person] | +0.042 | [+0.018, +0.065] | pass |
| no_regression[AP/building] | +0.031 | [+0.015, +0.046] | pass |
| no_regression[lighting=day/mAP] | +0.059 | [+0.039, +0.078] | pass |
| no_regression[lighting=dusk/mAP] | -0.010 | [-0.032, +0.017] | pass |
| no_regression[lighting=night/mAP] | -0.005 | [-0.020, +0.009] | pass |
| no_regression[fog=heavy/mAP] | +0.016 | [-0.015, +0.044] | pass |
| no_regression[fog=light/mAP] | -0.024 | [-0.051, +0.008] | pass |
| no_regression[fog=none/mAP] | +0.019 | [+0.005, +0.030] | pass |
| no_regression[altitude_band=high/mAP] | +0.050 | [+0.032, +0.065] | pass |
| no_regression[altitude_band=low/mAP] | +0.003 | [-0.026, +0.032] | pass |
| no_regression[altitude_band=mid/mAP] | -0.011 | [-0.025, +0.004] | pass |
