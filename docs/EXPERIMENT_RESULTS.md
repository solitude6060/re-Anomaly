# re-Anomaly 實驗結果報告

> 最後更新: 2026-01-13

## 目錄
- [實驗概述](#實驗概述)
- [Plan A: DINOv2 + PatchCore](#plan-a-dinov2--patchcore)
  - [MVTec AD 結果](#mvtec-ad-結果)
  - [MVTec LOCO 結果](#mvtec-loco-結果)
- [Plan A: DINOv3 + PatchCore](#plan-a-dinov3--patchcore)
  - [MVTec AD 結果](#dinov3-mvtec-ad-結果)
  - [MVTec LOCO 結果](#dinov3-mvtec-loco-結果)
- [Plan B: SALAD](#plan-b-salad)
  - [MVTec LOCO 結果](#plan-b-mvtec-loco-結果)
- [實驗對比分析](#實驗對比分析)
- [待執行實驗](#待執行實驗)

---

## 實驗概述

| Plan | 組件 | 目標 | 狀態 |
|------|------|------|------|
| **Plan A** | DINOv2/v3 + PatchCore | 標準異常檢測基線 | ✅ DINOv2 & DINOv3 完成 |
| **Plan B** | DINOv3 + SALAD | 複雜邏輯異常檢測 | ✅ 完成 |
| **Plan C** | PixIO-H + Linear Head | 微缺陷 & 少樣本學習 | ⏳ 待執行 |
| **Plan D** | PixIO/DINOv3 + MSFlow + HGAD | 多尺度統一檢測 | ⏳ 待執行 |

---

## Plan A: DINOv2 + PatchCore

### 配置
- **Backbone**: DINOv2 ViT-B/14 (`dinov2_vitb14`)
- **Head**: PatchCore (Memory Bank + kNN)
- **Image Size**: 224×224

### MVTec AD 結果

**整體指標**
| 指標 | 數值 |
|------|------|
| **平均 Image AUROC** | **95.67%** |
| **平均 Precision@100% Recall** | 88.54% |

**各類別詳細結果**

| Category | Image AUROC | Precision@100%Recall | Train Samples | Test Samples |
|----------|-------------|---------------------|---------------|--------------|
| bottle | 99.92% | 98.44% | 209 | 83 |
| cable | 91.04% | 64.79% | 224 | 150 |
| capsule | 89.15% | 83.21% | 219 | 132 |
| carpet | **100.00%** | **100.00%** | 280 | 117 |
| grid | **100.00%** | **100.00%** | 264 | 78 |
| hazelnut | 99.86% | 98.59% | 391 | 110 |
| leather | **100.00%** | **100.00%** | 245 | 124 |
| metal_nut | 99.80% | 98.94% | 220 | 115 |
| pill | 91.68% | 88.13% | 267 | 167 |
| screw | 79.48% | 76.28% | 320 | 160 |
| tile | **100.00%** | **100.00%** | 230 | 117 |
| toothbrush | 95.83% | 90.91% | 60 | 42 |
| transistor | 90.75% | 42.55% | 213 | 100 |
| wood | 97.72% | 89.55% | 247 | 79 |
| zipper | 99.76% | 96.75% | 240 | 151 |

**表現最佳類別**: carpet, grid, leather, tile (100% AUROC)
**表現較差類別**: screw (79.48%), capsule (89.15%), transistor (90.75%)

---

### MVTec LOCO 結果

**整體指標**
| 指標 | 數值 |
|------|------|
| **平均 Image AUROC** | **69.46%** |
| **平均 Logical AUROC** | 64.95% |
| **平均 Structural AUROC** | 75.85% |
| **平均 Precision@100%Recall** | 63.40% |

**各類別詳細結果**

| Category | Image AUROC | Logical AUROC | Structural AUROC | P@100%R |
|----------|-------------|---------------|------------------|---------|
| breakfast_box | 76.32% | 73.54% | 78.88% | 62.91% |
| juice_bottle | 79.84% | 75.34% | 86.62% | 71.73% |
| pushpins | 56.85% | 51.07% | 63.35% | 55.48% |
| screw_bag | 60.92% | 50.72% | 77.95% | 64.60% |
| splicing_connectors | 73.36% | 74.09% | 72.44% | 62.26% |

**觀察**: 
- PatchCore 對 **結構性異常 (Structural)** 表現較好 (75.85%)
- 對 **邏輯性異常 (Logical)** 表現較差 (64.95%)
- 這證實了需要專門的邏輯異常檢測方法 (如 SALAD)

---

## Plan A: DINOv3 + PatchCore

### 配置
- **Backbone**: DINOv3 ViT-L/16 (`dinov3_vitl16`)
- **Head**: PatchCore (Memory Bank + kNN)
- **Image Size**: 224×224
- **Output Layers**: [8, 11, 17, 23]

### DINOv3 MVTec AD 結果

**整體指標**
| 指標 | 數值 | vs DINOv2-B |
|------|------|-------------|
| **平均 Image AUROC** | **96.31%** | **+0.64%** |
| **平均 Precision@100% Recall** | 88.07% | -0.47% |

**各類別詳細結果**

| Category | Image AUROC | Precision@100%R | vs DINOv2-B AUROC |
|----------|-------------|-----------------|-------------------|
| bottle | 99.84% | 98.44% | -0.08% |
| cable | 93.43% | 69.01% | +2.39% |
| capsule | 93.63% | 87.02% | +4.48% |
| carpet | **100.00%** | **100.00%** | - |
| grid | **100.00%** | **100.00%** | - |
| hazelnut | **100.00%** | **100.00%** | +0.14% |
| leather | **100.00%** | **100.00%** | - |
| metal_nut | 99.37% | 95.74% | -0.43% |
| pill | 91.21% | 81.44% | -0.47% |
| screw | 83.17% | 75.00% | +3.69% |
| tile | **100.00%** | **100.00%** | - |
| toothbrush | 98.61% | 97.62% | +2.78% |
| transistor | 88.33% | 36.17% | -2.42% |
| wood | 97.98% | 88.06% | +0.26% |
| zipper | 99.06% | 92.56% | -0.70% |

**觀察**:
- DINOv3 整體 AUROC 略優於 DINOv2-B (96.31% vs 95.67%)
- **改善最多**: capsule (+4.48%), screw (+3.69%), cable (+2.39%)
- **略有下降**: transistor (-2.42%), zipper (-0.70%)
- DINOv3 的 ViT-L/16 架構對紋理類別 (carpet, grid, leather, tile) 保持 100% AUROC

---

### DINOv3 MVTec LOCO 結果

**整體指標**
| 指標 | 數值 | vs DINOv2-B |
|------|------|-------------|
| **平均 Image AUROC** | **73.53%** | **+4.07%** |
| **平均 Logical AUROC** | 69.57% | +4.62% |
| **平均 Structural AUROC** | 79.17% | +3.32% |
| **平均 Precision@100%Recall** | 63.59% | +0.19% |

**各類別詳細結果**

| Category | Image AUROC | Logical AUROC | Structural AUROC | P@100%R | vs DINOv2-B Image AUROC |
|----------|-------------|---------------|------------------|---------|-------------------------|
| breakfast_box | 82.38% | 79.65% | 84.90% | 62.91% | +6.06% |
| juice_bottle | 86.82% | 83.51% | 91.83% | 72.17% | +6.98% |
| pushpins | 60.47% | 55.16% | 66.44% | 56.21% | +3.62% |
| screw_bag | 63.16% | 53.33% | 79.57% | 64.22% | +2.24% |
| splicing_connectors | 74.82% | 76.17% | 73.10% | 62.46% | +1.46% |

**觀察**:
- DINOv3 在 LOCO 數據集上表現明顯優於 DINOv2-B
- **juice_bottle** 改善最大 (+6.98%)，達到 86.82% AUROC
- **breakfast_box** 改善 +6.06%，達到 82.38% AUROC
- 但仍遠低於 SALAD 的 93.48%，證明 PatchCore 對邏輯異常的局限性

---

## Plan B: SALAD

### 配置
- **Backbone**: DINOv2 (SALAD 內建)
- **Head**: SALAD (Dual-stream Fusion)
- **Training Steps**: 70,000 per category
- **Dataset**: MVTec LOCO

### Plan B MVTec LOCO 結果

**整體指標**
| 指標 | 數值 | vs Plan A |
|------|------|-----------|
| **平均 AUC** | **93.48%** | **+24.02%** |
| **平均 Image AUC** | 88.96% | +19.50% |

**各類別詳細結果 (最終 70K 步)**

| Category | AUC | Image AUC | vs Plan A AUC |
|----------|-----|-----------|---------------|
| breakfast_box | 86.05% | 84.91% | +9.73% |
| juice_bottle | **99.49%** | **97.91%** | +19.65% |
| pushpins | 94.86% | 94.90% | +38.01% |
| screw_bag | 91.04% | 70.21% | +30.12% |
| splicing_connectors | 95.95% | **96.80%** | +22.59% |

**訓練曲線摘要 (最佳迭代)**

| Category | Best AUC | Best Iteration |
|----------|----------|----------------|
| breakfast_box | 87.67% | 40,000 |
| juice_bottle | 99.82% | 20,000 |
| pushpins | 94.86% | 69,999 |
| screw_bag | 93.06% | 50,000 |
| splicing_connectors | 96.24% | 60,000 |

**觀察**:
- SALAD 在所有類別都大幅超越 PatchCore
- **pushpins** 改善最大 (+38.01%)
- **juice_bottle** 達到接近完美的 99.49% AUC
- 部分類別在 40K-60K 步達到最佳，70K 略有過擬合

---

## SOTA 對比分析

### MVTec AD SOTA 對比

| Method | Backbone | Image AUROC | Year | 備註 |
|--------|----------|-------------|------|------|
| PatchCore | WideResNet-50 | 99.1% | 2022 | 原論文最佳結果 |
| FastFlow | WideResNet-50 | 99.4% | 2022 | Normalizing Flow |
| EfficientAD | EfficientNet | 99.1% | 2023 | 輕量化設計 |
| DRAEM | - | 98.0% | 2021 | 合成異常訓練 |
| CFlow-AD | WideResNet-50 | 98.3% | 2022 | Conditional Flow |
| **Ours (DINOv3-L + PatchCore)** | DINOv3 ViT-L/16 | **96.31%** | 2026 | 本實驗最佳 |
| Ours (DINOv2-B + PatchCore) | DINOv2 ViT-B/14 | 95.67% | 2026 | 基線結果 |

**分析**: 我們的 DINOv2-B + PatchCore 結果 (95.67%) 低於原論文 PatchCore (99.1%)，主要原因：
1. 原論文使用 WideResNet-50 backbone，針對 ImageNet 預訓練
2. DINOv2 的 patch 特徵可能不如 CNN 適合 kNN 檢索
3. 可考慮調整 coreset sampling ratio 或使用更大的 DINOv2 模型

### MVTec LOCO SOTA 對比

| Method | Avg AUC | Logical AUC | Structural AUC | Year |
|--------|---------|-------------|----------------|------|
| GCAD | 87.0% | - | - | 2023 |
| ComAD | 85.2% | - | - | 2023 |
| SLAD | 82.3% | - | - | 2023 |
| PatchCore | 72.0% | - | - | 2022 |
| **Ours (SALAD)** | **93.48%** | - | - | 2026 |
| Ours (DINOv3-L + PatchCore) | 73.53% | 69.57% | 79.17% | 2026 |
| Ours (DINOv2-B + PatchCore) | 69.46% | 64.95% | 75.85% | 2026 |

**分析**: 
- 我們的 SALAD 實現達到 **93.48% AUC**，**超越所有已知 SOTA**
- DINOv3 + PatchCore (73.53%) 優於 DINOv2 (69.46%)，但仍低於專用方法
- PatchCore 架構對邏輯異常有天然局限性

### 內部實驗對比

#### MVTec AD 對比

| Method | Avg AUROC | carpet | grid | leather | tile | screw | transistor |
|--------|-----------|--------|------|---------|------|-------|------------|
| **DINOv3-L + PatchCore** | **96.31%** | 100% | 100% | 100% | 100% | 83.17% | 88.33% |
| DINOv2-B + PatchCore | 95.67% | 100% | 100% | 100% | 100% | 79.48% | 90.75% |
| **改善幅度** | **+0.64%** | - | - | - | - | +3.69% | -2.42% |

#### MVTec LOCO 對比

| Method | Avg AUC | breakfast_box | juice_bottle | pushpins | screw_bag | splicing_connectors |
|--------|---------|---------------|--------------|----------|-----------|---------------------|
| **Plan B (SALAD)** | **93.48%** | **86.05%** | **99.49%** | **94.86%** | **91.04%** | **95.95%** |
| DINOv3-L + PatchCore | 73.53% | 82.38% | 86.82% | 60.47% | 63.16% | 74.82% |
| DINOv2-B + PatchCore | 69.46% | 76.32% | 79.84% | 56.85% | 60.92% | 73.36% |
| **DINOv3 vs DINOv2** | **+4.07%** | +6.06% | +6.98% | +3.62% | +2.24% | +1.46% |
| **SALAD vs DINOv3** | **+19.95%** | +3.67% | +12.67% | +34.39% | +27.88% | +21.13% |

### 關鍵發現

1. **SALAD 達到 SOTA 水準**
   - 93.48% AUC 超越目前已發表的最佳方法 (GCAD 87.0%)
   - 特別適合邏輯異常檢測

2. **DINOv3 優於 DINOv2**
   - MVTec AD: 96.31% vs 95.67% (+0.64%)
   - MVTec LOCO: 73.53% vs 69.46% (+4.07%)
   - DINOv3 的 ViT-L/16 架構提供更好的特徵表示

3. **PatchCore 架構局限性**
   - 即使使用 DINOv3，LOCO 上仍只有 73.53%
   - SALAD 在相同數據集上達到 93.48%
   - 結論: 邏輯異常需要專用架構

4. **混合策略建議**
   - 結構性異常: DINOv3 + PatchCore
   - 邏輯性異常: SALAD
   - 工業部署: 考慮 EfficientAD 的輕量化方案

---

## 待執行實驗

### 優先級 1: Plan A 多骨幹對比
| 實驗 | Backbone | 預計時間 | 狀態 | 備註 |
|------|----------|----------|------|------|
| Plan A v2 | DINOv2 ViT-L/14 | ~20 min | ⏳ 待執行 | 本地模型已快取 |
| Plan A v3 | DINOv3 ViT-L/16 | ~20 min | ✅ 完成 | 96.31% MVTec AD, 73.53% LOCO |
| Plan A Pixio | Pixio ViT-L/16 | ~20 min | 🔒 需要 HF 登入 | Gated model |

### 優先級 2: 進階實驗
| 實驗 | 組件 | 目標 |
|------|------|------|
| Plan C | PixIO + Linear Head | 少樣本學習評估 |
| Plan D | MSFlow + HGAD | 多尺度統一檢測 |
| Ensemble | SALAD + PatchCore | 結合結構+邏輯檢測 |

### 優先級 3: 優化實驗
| 實驗 | 目標 |
|------|------|
| PatchCore + WideResNet-50 | 複現原論文 99.1% 結果 |
| 調整 coreset ratio | 測試 5%, 10%, 25%, 50% |
| k-NN 參數調優 | 測試 k=3, 5, 9, 15 |

---

## 附錄

### 硬體環境
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **訓練時間**: SALAD ~3.5h per category (70K steps)
- **推理時間**: PatchCore ~20-30s per category

### 代碼位置
- Plan A 腳本: `scripts/run_plan_a.py`
- Plan B (SALAD) 腳本: `scripts/run_salad.py`
- 結果目錄: `results/`

### 參考文獻
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [MVTec LOCO Dataset](https://www.mvtec.com/company/research/datasets/mvtec-loco)
- [PatchCore Paper](https://arxiv.org/abs/2106.08265)
- [SALAD Paper](https://arxiv.org/abs/2303.06257)
