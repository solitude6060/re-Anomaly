# re-Anomaly 實驗結果報告

> 最後更新: 2026-01-12

## 目錄
- [實驗概述](#實驗概述)
- [Plan A: DINOv2 + PatchCore](#plan-a-dinov2--patchcore)
  - [MVTec AD 結果](#mvtec-ad-結果)
  - [MVTec LOCO 結果](#mvtec-loco-結果)
- [Plan B: SALAD](#plan-b-salad)
  - [MVTec LOCO 結果](#plan-b-mvtec-loco-結果)
- [實驗對比分析](#實驗對比分析)
- [待執行實驗](#待執行實驗)

---

## 實驗概述

| Plan | 組件 | 目標 | 狀態 |
|------|------|------|------|
| **Plan A** | DINOv2/v3 + PatchCore | 標準異常檢測基線 | ✅ DINOv2 完成 |
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

## 實驗對比分析

### MVTec LOCO 對比

| Method | Avg AUC | breakfast_box | juice_bottle | pushpins | screw_bag | splicing_connectors |
|--------|---------|---------------|--------------|----------|-----------|---------------------|
| **Plan A (PatchCore)** | 69.46% | 76.32% | 79.84% | 56.85% | 60.92% | 73.36% |
| **Plan B (SALAD)** | **93.48%** | **86.05%** | **99.49%** | **94.86%** | **91.04%** | **95.95%** |
| **改善幅度** | **+24.02%** | +9.73% | +19.65% | +38.01% | +30.12% | +22.59% |

### 關鍵發現

1. **SALAD 大幅優於 PatchCore 處理邏輯異常**
   - 平均提升 24.02%
   - 特別是 pushpins (+38%) 和 screw_bag (+30%)

2. **PatchCore 在標準異常檢測仍有價值**
   - MVTec AD 達到 95.67% AUROC
   - 對結構性異常表現良好

3. **混合策略建議**
   - 結構性異常: PatchCore
   - 邏輯性異常: SALAD
   - 或使用 SALAD 的雙流架構統一處理

---

## 待執行實驗

### 優先級 1: Plan A 多骨幹對比
| 實驗 | Backbone | 預計時間 | 狀態 |
|------|----------|----------|------|
| Plan A v2 | DINOv2 ViT-L/14 | ~20 min | ⏳ 待執行 |
| Plan A v3 | DINOv3 ViT-L/16 | ~20 min | ⏳ 待執行 |
| Plan A Pixio | Pixio ViT-L/16 | ~20 min | ⏳ 待執行 |

### 優先級 2: 進階實驗
| 實驗 | 組件 | 目標 |
|------|------|------|
| Plan C | PixIO + Linear Head | 少樣本學習評估 |
| Plan D | MSFlow + HGAD | 多尺度統一檢測 |
| Ensemble | SALAD + PatchCore | 結合結構+邏輯檢測 |

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
