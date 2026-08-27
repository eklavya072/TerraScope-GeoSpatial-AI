### Which accuracy differences are statistically distinguishable?

Welch's t-test over seeds, Holm-Bonferroni corrected across all pairwise comparisons (family-wise alpha = 0.05).

| Comparison | Δ accuracy (pp) | p | Holm threshold | Distinguishable? |
|---|---:|---:|---:|---|
| efficientnet_lite0 vs mobilenetv3_small | +0.41 | 0.0001 | 0.0050 | **yes** |
| mobilenetv3_large vs resnet50 | -1.01 | 0.0001 | 0.0056 | **yes** |
| efficientnet_lite0 vs mobilenetv3_large | +0.50 | 0.0002 | 0.0063 | **yes** |
| mobilenetv3_small vs resnet50 | -0.93 | 0.0005 | 0.0071 | **yes** |
| mobilenetv3_large vs mobilevit_s | -1.25 | 0.0006 | 0.0083 | **yes** |
| mobilenetv3_small vs mobilevit_s | -1.16 | 0.0014 | 0.0100 | **yes** |
| efficientnet_lite0 vs resnet50 | -0.51 | 0.0063 | 0.0125 | **yes** |
| efficientnet_lite0 vs mobilevit_s | -0.75 | 0.0073 | 0.0167 | **yes** |
| mobilenetv3_large vs mobilenetv3_small | -0.09 | 0.2480 | 0.0250 | no |
| mobilevit_s vs resnet50 | +0.24 | 0.2565 | 0.0500 | no |

8 of 10 pairwise accuracy differences are statistically distinguishable after correction.
