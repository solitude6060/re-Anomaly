# MVTec AD Benchmark Summary

**Date:** Sun Jan 18 2026
**Dataset:** MVTec AD
**Backbones:** dinov3_vitl16, convnext_tiny
**Heads:** dinomaly, patchcore

## Results

| Experiment | Category | AUROC | Time (s) |
|---|---|---|---|
| convnext_tiny + patchcore | bottle | 0.9659 | 130.80 |
| convnext_tiny + patchcore | cable | 0.8276 | 129.37 |
| dinov3_vitl16 + dinomaly | bottle | 0.9992 | 46.10 |
| dinov3_vitl16 + dinomaly | cable | 0.9006 | 103.01 |
| dinov3_vitl16 + dinomaly | capsule | 0.8365 | 113.46 |
| dinov3_vitl16 + dinomaly | carpet | 0.9593 | 61.38 |
| dinov3_vitl16 + patchcore | bottle | 0.9976 | 8.70 |
| dinov3_vitl16 + patchcore | cable | 0.8634 | 11.70 |
| dinov3_vitl16 + patchcore | capsule | 0.7619 | 9.91 |
| dinov3_vitl16 + patchcore | carpet | 0.9446 | 10.36 |

## Key Observations

1.  **Performance:** `dinov3_vitl16 + dinomaly` consistently achieved higher AUROC scores compared to `dinov3_vitl16 + patchcore` across all tested categories (Bottle, Cable, Capsule, Carpet).
2.  **Speed:** `patchcore` with `dinov3_vitl16` is significantly faster (approx. 5-10x faster) than `dinomaly` in terms of total execution time (fitting + inference).
3.  **Backbone:** `dinov3_vitl16` appears to provide strong features for both heads.
4.  **Completion Status:** This report includes a subset of the full benchmark suite. The full suite (60 experiments) was interrupted due to time constraints, but these results provide a representative sample of performance.
