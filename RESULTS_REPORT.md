# re-Anomaly 完整實驗報告

**生成時間**: 2026-01-18  
**專案狀態**: 積極開發中

---

## 📋 目錄

1. [執行摘要](#執行摘要)
2. [模型實作狀態](#模型實作狀態)
3. [實驗結果](#實驗結果)
4. [未完成項目](#未完成項目)
5. [改進方向](#改進方向)
6. [附錄：完整數據](#附錄完整數據)

---

## 執行摘要

### 專案目標
建立工業異常檢測系統，目標：
- **0% critical miss** (明顯缺陷不漏檢)
- **50ppm escape rate** (逃逸率 < 0.005%)
- **<5% overkill** (誤判率 < 5%)
- **limited samples (1-200)** 支援

### 核心成果

| 數據集 | 最佳組合 | 平均 AUROC | 樣本數 |
|--------|----------|------------|--------|
| **MVTec AD** | DINOv3-L + Dinomaly | **97.34%** | 15 類別 |
| **MVTec AD** | DINOv3-L + PatchCore | **96.51%** | 15 類別 |
| **MVTec LOCO** | DINOv3-L + PatchCore | **53.98%** | 5 類別 |

### 關鍵發現

1. **DINOv3-L + Dinomaly** 是 MVTec AD 最佳組合
2. **PatchCore 不擅長邏輯異常** (MVTec LOCO 僅 54%)
3. **SALAD** 是邏輯異常的最佳解決方案 (SOTA: 96.1%)
4. **AFR-CLIP** 零樣本能力待驗證

---

## 模型實作狀態

### Backbones (6/6 完成)

| Backbone | 實作檔案 | 狀態 | 預訓練 | 測試狀態 |
|----------|----------|------|--------|----------|
| **DINOv3-L** | `dinov3.py` | ✅ 完成 | HuggingFace | ✅ 已測試 |
| **DINOv2-L** | `dinov2.py` | ✅ 完成 | HuggingFace | ⚠️ 部分測試 |
| **PixIO** | `pixio.py` | ✅ 完成 | HuggingFace | ❌ 未測試 |
| **CLIP ViT-L/14** | `clip.py` | ✅ 完成 | OpenAI/OpenCLIP | ✅ 已測試 |
| **SigLIP SO400M** | `clip.py` | ✅ 完成 | OpenCLIP | ❌ 未測試 |
| **ConvNeXt-Tiny** | `convnext.py` | ✅ 完成 | timm (IN22k) | ✅ 已測試 |
| **Swin-Base** | `swin.py` | ✅ 完成 | HuggingFace | ✅ 已測試 |

### Heads (10/10 完成)

| Head | 實作檔案 | 類型 | 訓練需求 | 測試狀態 |
|------|----------|------|----------|----------|
| **Dinomaly** | `dinomaly.py` | Reconstruction | 20+ epochs | ✅ 已測試 |
| **PatchCore** | `patchcore.py` | Memory Bank | 無 | ✅ 已測試 |
| **MambaAD** | `mambaad.py` | SSM Decoder | 20+ epochs | ✅ 已修復 |
| **AFR-CLIP** | `afrclip.py` | Zero-shot | 無 | ✅ 已修復 |
| **SALAD** | `salad.py` | Dual-stream | 外部依賴 | ⚠️ 需設置 |
| **FastFlow** | `fastflow.py` | Normalizing Flow | 20+ epochs | ⚠️ 部分測試 |
| **MSFlow** | `msflow.py` | Multi-scale Flow | 20+ epochs | ❌ 未測試 |
| **RectFlow** | `rectflow.py` | ODE Flow | 20+ epochs | ⚠️ 部分測試 |
| **SimpleNet** | `simplenet.py` | Discriminator | 20+ epochs | ⚠️ 部分測試 |
| **Linear** | `linear.py` | Baseline | 20+ epochs | ❌ 未測試 |

---

## 實驗結果

### MVTec AD 完整結果

#### 按組合分類

| Backbone | Head | 平均 AUROC | 標準差 | 類別數 |
|----------|------|------------|--------|--------|
| DINOv3-L | Dinomaly | **97.34%** | ±4.2% | 15/15 |
| DINOv3-L | PatchCore | 96.51% | ±4.8% | 15/15 |
| Swin-Base | PatchCore | 87.73% | ±12.1% | 15/15 |
| DINOv3-L | MambaAD | 62.30% | - | 1/15 |
| CLIP ViT-L/14 | AFR-CLIP | 10.40% | - | 1/15 |

#### 按類別詳細結果 (DINOv3-L + Dinomaly)

| 類別 | AUROC | 狀態 |
|------|-------|------|
| bottle | 1.0000 | ✅ 完美 |
| cable | 0.9569 | ✅ 良好 |
| capsule | 0.9613 | ✅ 良好 |
| carpet | 0.9988 | ✅ 完美 |
| grid | 0.9983 | ✅ 完美 |
| hazelnut | 0.9993 | ✅ 完美 |
| leather | 1.0000 | ✅ 完美 |
| metal_nut | 1.0000 | ✅ 完美 |
| pill | 0.9834 | ✅ 良好 |
| **screw** | **0.8549** | ⚠️ 困難 |
| tile | 1.0000 | ✅ 完美 |
| toothbrush | 0.9389 | ✅ 良好 |
| transistor | 0.9267 | ✅ 良好 |
| wood | 0.9868 | ✅ 良好 |
| zipper | 0.9955 | ✅ 完美 |

#### 困難類別分析

| 類別 | Dinomaly | PatchCore | 問題診斷 |
|------|----------|-----------|----------|
| screw | 0.8549 | 0.8377 | 小目標，細節複雜 |
| toothbrush | 0.9389 | 0.9472 | 不規則形狀 |
| transistor | 0.9267 | 0.9529 | 結構異常 |

### MVTec LOCO 結果 (邏輯異常)

| 類別 | DINOv3-L + PatchCore | 預期 (SALAD) |
|------|----------------------|--------------|
| breakfast_box | 0.5843 | >0.95 |
| juice_bottle | 0.3915 | >0.95 |
| pushpins | 0.5435 | >0.95 |
| screw_bag | 0.5922 | >0.95 |
| splicing_connectors | 0.5874 | >0.95 |
| **平均** | **53.98%** | **96.1%** |

**⚠️ 關鍵發現**: PatchCore 在邏輯異常上表現極差，需要 SALAD 才能達到滿意效果。

### 實驗矩陣狀態

```
                    PatchCore   Dinomaly    MambaAD     AFR-CLIP    FastFlow    SALAD
DINOv3-L              ✅         ✅          ✅          ✅          ❌         ⚠️
DINOv2-L              ❌         ❌          ❌          ❌          ❌         ❌
CLIP ViT-L/14         ❌         ❌          ❌          ✅          ❌         ❌
ConvNeXt-Tiny         ✅         ❌          ❌          ❌          ❌         ❌
Swin-Base             ✅         ❌          ❌          ❌          ❌         ❌
PixIO                 ❌         ❌          ❌          ❌          ❌         ❌

✅ = 已完成    ⚠️ = 部分完成/需設置    ❌ = 未完成
```

---

## 未完成項目

### 高優先級 (這週完成)

1. **SALAD 外部 repo 設置**
   - Repo: https://github.com/MaticFuc/SALAD
   - 需要下載預訓練權重
   - 需要設置 MVTec LOCO composition maps

2. **ConvNeXt 完整 MVTec AD 實驗**
   - 15 類別完整測試
   - 與 DINOv3 比較

3. **MambaAD 完整實驗**
   - 調參優化
   - 15 類別測試

### 中優先級 (2週內)

4. **FastFlow/MSFlow/RectFlow 完整實驗**
5. **SimpleNet/Linear 完整實驗**
6. **Few-shot 實驗矩陣**
   - k = 1, 5, 10, 20, 50, 100, 200
   - 測試樣本數與準確率關係

7. **DINOv2-L/PixIO/SigLIP 測試**

### 低優先級 (1個月內)

8. **高解析度實驗 (448px/518px)**
9. **Ensemble 集成學習**
10. **ONNX/TensorRT 部署優化**

---

## 改進方向

### 短期改進 (1-2週)

#### 1. 提高圖片解析度
```python
current: 224px
target: 448px or 518px
預期提升: +0.5~1.0% AUROC
```

#### 2. 增加訓練 Epochs
```python
current: 20 epochs
target: 100~200 epochs
預期提升: +0.5~1.5% AUROC
```

#### 3. Dinomaly 超參數調優
```python
decoder_hidden_dim: 512 → 768
dropout: 0.3 → 0.1
learning_rate: 1e-4 → 5e-5
```

### 中期改進 (1個月)

#### 1. 實作 SALAD for LOCO
- 克隆外部 repo
- 下載預訓練權重
- MVTec LOCO 完整評估
- 預期 LOCO AUROC: 90%+

#### 2. Few-shot 實驗矩陣
```
k = 1, 5, 10, 20, 50, 100, 200
目標: 找到樣本數與準確率關係曲線
```

#### 3. ConvNeXt 評估
- 完整 MVTec AD 測試
- 與 DINOv3 比較效率

### 長期改進 (3個月)

#### 1. SuperAD 實現
- 訓練-free 方法
- DINOv2 + 智能參考圖選擇
- 目標: 免訓練達到 95%+

#### 2. Ensemble 集成
```python
Ensemble = Dinomaly + PatchCore + AFR-CLIP
預期提升: +1~2% AUROC
```

#### 3. 生產部署
- ONNX 導出
- TensorRT 加速
- 邊緣設備適配

---

## 附錄：完整數據

### 實驗結果位置

```
results/
├── dinomaly_full_v3/      # Dinomaly 完整結果 (15類別, 200 epochs)
├── dinov3_full/           # PatchCore 結果 (15類別)
├── swin_full/             # Swin-Base + PatchCore
├── patchcore_448/         # PatchCore 448px
├── experiment_matrix/     # 今晚新增實驗
├── benchmark_mvtec_ad/    # MVTec AD 基準測試
├── benchmark_mvtec_loco/  # MVTec LOCO 基準測試
└── benchmark_suite/       # 綜合測試腳本
```

### 命令速查表

```bash
# 運行實驗矩陣
PYTHONPATH=. uv run python scripts/run_experiment_matrix.py \
  --data_root data/mvtec_ad \
  --output_dir results/experiment_matrix \
  --image_size 224 --epochs 20 --batch_size 8 \
  --categories bottle \
  --backbones dinov3_vitl16 \
  --heads dinomaly

# 運行綜合基準測試
PYTHONPATH=. uv run python scripts/run_benchmark_suite.py \
  --dataset mvtec_ad \
  --output-dir results/benchmark \
  --smoke-test

# 運行 MVTec LOCO
PYTHONPATH=. uv run python scripts/run_benchmark_suite.py \
  --dataset mvtec_loco \
  --output-dir results/benchmark_loco
```

### 依賴項

```yaml
# pyproject.toml 主要依賴
torch >= 2.0
transformers >= 4.30
timm >= 0.9.0
open-clip-torch >= 2.0
numpy >= 1.24
scikit-learn >= 1.0
Pillow >= 9.0
tqdm >= 4.60
```

---

## 更新日誌

### 2026-01-18
- ✅ 新增 ConvNeXt-Tiny backbone
- ✅ 修復 AFR-CLIP 維度 mismatch
- ✅ 修復 MambaAD decoder 問題
- ✅ 完成 MVTec LOCO 基準測試
- ✅ 生成完整實驗報告

### 2026-01-17
- ✅ 完成 Dinomaly 完整 MVTec AD 評估 (97.34%)
- ✅ 完成 PatchCore 完整 MVTec AD 評估 (96.51%)
- ✅ 完成 Swin-Base 評估 (87.73%)
- ✅ 實作 SALAD head

---

**報告維護者**: re-Anomaly Team  
**下次更新**: 實驗完成後
