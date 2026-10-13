# Validation report: camera-only (BLOCK)

- dataset version: `969966aae5426c70` (948 test frames)
- candidate weights: `f4a32645a8f28ab2` trained on `969966aae5426c70`
- baseline: fusion `bc56bf03685694a6`

| check | value / delta | 95% CI | result |
|---|---|---|---|
| min_mAP | 0.622 |  | pass |
| lineage_complete |  |  | pass |
| no_regression[mAP] | -0.050 | [-0.070, -0.029] | **FAIL** |
| no_regression[AP/vehicle] | -0.097 | [-0.124, -0.066] | **FAIL** |
| no_regression[AP/person] | -0.063 | [-0.104, -0.028] | **FAIL** |
| no_regression[AP/building] | +0.009 | [-0.026, +0.046] | pass |
| no_regression[lighting=day/mAP] | +0.000 | [-0.028, +0.027] | pass |
| no_regression[lighting=dusk/mAP] | -0.049 | [-0.093, +0.000] | pass |
| no_regression[lighting=night/mAP] | -0.085 | [-0.112, -0.054] | **FAIL** |
| no_regression[fog=heavy/mAP] | -0.230 | [-0.298, -0.145] | **FAIL** |
| no_regression[fog=light/mAP] | +0.005 | [-0.053, +0.052] | pass |
| no_regression[fog=none/mAP] | -0.043 | [-0.064, -0.020] | **FAIL** |
| no_regression[altitude_band=high/mAP] | -0.016 | [-0.038, +0.008] | pass |
| no_regression[altitude_band=low/mAP] | -0.297 | [-0.357, -0.216] | **FAIL** |
| no_regression[altitude_band=mid/mAP] | -0.020 | [-0.047, +0.008] | pass |
