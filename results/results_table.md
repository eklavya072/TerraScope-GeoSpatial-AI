### Results (ONNX Runtime CPU EP, intra-op threads=1, batch=1)

Accuracy is the mean over 5 seeds with a Student-t 95% confidence interval. Energy figures are ESTIMATED from on-die power telemetry, not metered at the wall. CO2e assumes 481 gCO2e/kWh (world average grid carbon intensity, ~481 gCO2e/kWh).

| Model | Precision | Params | Accuracy % (mean ± 95% CI) | p95 latency (ms) | Model RSS (MB) | Model (MB) | Energy/1k inf (J, estimated) | CO2e/1k inf (g, estimated) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| efficientnet_lite0 | fp32 | 3.38M | 97.61 ± 0.13 | 3.74 | 31 | 13.5 | 19.28 | 0.0026 |
| efficientnet_lite0 | int8_dynamic | 3.38M | 63.44 ± 3.97 | 4.43 | 19 | 3.6 | 25.75 | 0.0034 |
| efficientnet_lite0 | int8_static | 3.38M | 97.45 ± 0.32 | 0.43 | 17 | 3.8 | 3.18 | 0.0004 |
| mobilenetv3_large | fp32 | 4.21M | 97.10 ± 0.17 | 2.73 | 35 | 16.8 | 14.46 | 0.0019 |
| mobilenetv3_large | int8_dynamic | 4.21M | 70.47 ± 8.10 | 2.88 | 19 | 4.4 | 19.32 | 0.0026 |
| mobilenetv3_large | int8_static | 4.21M | 91.10 ± 1.11 | 0.52 | 21 | 4.7 | 3.60 | 0.0005 |
| mobilenetv3_small | fp32 | 1.53M | 97.19 ± 0.10 | 1.40 | 18 | 6.1 | 6.97 | 0.0009 |
| mobilenetv3_small | int8_dynamic | 1.53M | 14.57 ± 2.95 | 1.54 | 14 | 1.7 | 8.02 | 0.0011 |
| mobilenetv3_small | int8_static | 1.53M | 32.53 ± 10.40 | 0.32 | 19 | 1.9 | 1.72 | 0.0002 |
| mobilevit_s | fp32 | 4.94M | 98.36 ± 0.44 | 4.46 | 37 | 20.0 | 26.31 | 0.0035 |
| mobilevit_s | int8_dynamic | 4.94M | 58.41 ± 10.82 | 3.91 | 30 | 5.5 | 24.53 | 0.0033 |
| mobilevit_s | int8_static | 4.94M | 49.50 ± 9.13 | 2.11 | 28 | 5.8 | 13.77 | 0.0018 |
| resnet50 | fp32 | 23.53M | 98.12 ± 0.30 | 12.24 | 162 | 94.0 | 60.84 | 0.0081 |
| resnet50 | int8_dynamic | 23.53M | 80.49 ± 8.15 | 5.36 | 39 | 23.7 | 27.08 | 0.0036 |
| resnet50 | int8_static | 23.53M | 97.22 ± 0.38 | 2.49 | 67 | 24.0 | 14.39 | 0.0019 |
