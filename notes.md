# re-Anomaly 專案筆記

## 專案目標
建立工業異常檢測系統，針對 0% critical miss、50ppm escape rate、<5% overkill 目標，支援有限樣本 (1-200)。

---

## 已完成功能

### Backbones
- [x] DINOv3-L (HuggingFace + DINOv2 fallback)
- [x] DINOv2-L (HuggingFace)
- [x] PixIO (MAE-based, 8 class tokens)
- [x] CLIP ViT-L/14 (OpenAI/OpenCLIP)
- [x] CLIP ViT-B/16 (OpenAI/OpenCLIP)
- [x] SigLIP SO400M (OpenCLIP)
- [x] ConvNeXt-Tiny (timm, IN22k)
- [x] ConvNeXt-Base (timm, IN22k)
- [x] ConvNeXt-Base DINOv3 weights (HF)
- [x] Swin-Base (HuggingFace)

### Heads
- [x] Dinomaly (CVPR 2025, LinearAttention2)
- [x] PatchCore (Coreset + kNN)
- [x] MambaAD (NeurIPS 2024, SSM)
- [x] AFR-CLIP (Zero-shot)
- [x] SALAD (ICCV 2025, 邏輯異常)
- [x] FastFlow (2D Normalizing Flow)
- [x] MSFlow (Multi-scale Flow)
- [x] RectFlow (ODE Flow)
- [x] SimpleNet (Discriminator)
- [x] Linear (Baseline MLP)

### 腳本工具
- [x] `scripts/run_experiment_matrix.py` - 實驗矩陣
- [x] `scripts/run_benchmark_suite.py` - 綜合基準測試
- [x] `scripts/run_salad.py` - SALAD 訓練/測試

---

## 實驗結果摘要

### MVTec AD (結構異常)
| 組合 | AUROC | 狀態 |
|------|-------|------|
| DINOv3-L + Dinomaly | **97.34%** | ✅ 最佳 |
| DINOv3-L + PatchCore | 96.51% | ✅ |
| Swin-Base + PatchCore | 87.73% | ✅ |
| ConvNeXt-T + PatchCore | 99.44% | ✅ (bottle) |
| CLIP + AFR-CLIP | 10.40% | ✅ (zero-shot) |

### MVTec LOCO (邏輯異常)
| 組合 | AUROC | 狀態 |
|------|-------|------|
| DINOv3-L + PatchCore | 53.98% | ⚠️ 不擅長 |
| DINOv3-L + SALAD | 96.11% | ✅ |

---

## 待完成實驗

### 高優先級
- [ ] ConvNeXt 完整 MVTec AD (15類別) + DINOv3 weights
- [x] SALAD MVTec LOCO 評估
- [ ] MambaAD 調參優化
- [ ] CLIP zero-shot heads 全量 (light run in progress)
- [ ] AD-DINOv3 全量 (light run in progress)

### 中優先級
- [ ] Few-shot 實驗 (k=1,5,10,20,50,100,200)
- [ ] FastFlow/MSFlow 完整測試
- [ ] DINOv2-L/PixIO 評估

### 低優先級
- [ ] 高解析度 (448px/518px)
- [ ] Ensemble 集成
- [ ] ONNX 部署

---

## 快速開始

```bash
# 運行實驗矩陣
PYTHONPATH=. uv run python scripts/run_experiment_matrix.py \
  --data_root data/mvtec_ad \
  --output_dir results/experiment_matrix \
  --image_size 224 --epochs 20 \
  --categories bottle \
  --backbones dinov3_vitl16 \
  --heads dinomaly

# 運行基準測試
PYTHONPATH=. uv run python scripts/run_benchmark_suite.py \
  --dataset mvtec_ad \
  --smoke-test
```

---

## 問題診斷

### MambaAD loss = 0
**原因**: Decoder 未輸出與 encoder 相同解析度的特徵  
**解決**: 修改 `mambaad.py` 的 `forward` 方法，移除 `if i != 0:` 條件

### AFR-CLIP 維度 mismatch
**原因**: CLIP ViT-L/14 visual=1024-dim, text=768-dim  
**解決**: 添加 `_visual_to_text_proj` 投影層

### PatchCore 不擅長邏輯異常
**原因**: 記憶體銀行方法僅學習局部特徵  
**解決**: 使用 SALAD 雙流架構

---

## 有用連結

- [SALAD GitHub](https://github.com/MaticFuc/SALAD)
- [MVTec AD](https://www.mvtec.com/technologies/mvtec-datasets/mvtec-ad)
- [MVTec LOCO](https://www.mvtec.com/technologies/mvtec-datasets/mvtec-loco)

---

## 更新日誌

### 2026-01-19
- 新增 CLIP ViT-B/16 backbone
- 新增 AnomalyCLIP/AF-CLIP/ACD-CLIP/MADPOT/AD-DINOv3 heads
- 下載 DINOv3 ConvNeXt 權重 (base/small/large)
- 啟動輕量化全量實驗 (clip_vitb16 zero-shot + ad_dinov3)

### 2026-01-18
- 新增 ConvNeXt-Tiny backbone
- 修復 AFR-CLIP 維度問題
- 修復 MambaAD decoder 問題
- 完成 MVTec LOCO 基準測試
- 更新完整實驗報告

### 2026-01-17
- 完成 Dinomaly 完整評估 (97.34%)
- 完成 PatchCore 完整評估 (96.51%)
- 完成 Swin-Base 評估 (87.73%)
