### Accuracy cost of int8 quantisation

Paired per-seed differences against each model's own fp32 export, mean with a Student-t 95% confidence interval. Negative means quantisation lost accuracy.

| Model | Precision | fp32 % | int8 % | Δ (pp, mean ± 95% CI) |
|---|---|---:|---:|---:|
| efficientnet_lite0 | int8_dynamic | 97.61 | 63.44 | -34.17 ± 4.07 |
| efficientnet_lite0 | int8_static | 97.61 | 97.45 | -0.16 ± 0.23 |
| mobilenetv3_large | int8_dynamic | 97.10 | 70.47 | -26.63 ± 8.12 |
| mobilenetv3_large | int8_static | 97.10 | 91.10 | -6.01 ± 1.08 |
| mobilenetv3_small | int8_dynamic | 97.19 | 14.57 | -82.62 ± 2.92 |
| mobilenetv3_small | int8_static | 97.19 | 32.53 | -64.66 ± 10.34 |
| mobilevit_s | int8_dynamic | 98.36 | 58.41 | -39.94 ± 11.18 |
| mobilevit_s | int8_static | 98.36 | 49.50 | -48.86 ± 9.28 |
| resnet50 | int8_dynamic | 98.12 | 80.49 | -17.63 ± 7.89 |
| resnet50 | int8_static | 98.12 | 97.22 | -0.90 ± 0.50 |
