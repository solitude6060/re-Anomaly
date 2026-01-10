# re-Anomaly

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## 概述
re-Anomaly 是專為 SMT（表面貼裝技術）工業異常檢測設計的次世代系統，支援多骨幹網路架構。本專案整合了多種先進的視覺骨幹網路與檢測頭，旨在提供極高的檢測精度，實現零漏失率與低過殺率的目標。

## 主要特色
- 多骨幹網路支援：整合了 PixIO, DINOv3, 以及 DINOv2 等先進預訓練模型。
- 多樣化檢測頭：提供 PatchCore, MSFlow, SimpleNet, SALAD, 與 FastFlow 等多種異常檢測演算法。
- 少樣本學習：針對工業場景中樣本獲取困難的問題，最佳化少樣本學習能力。
- 零漏失目標：
  - 0% 漏失率（Zero Leakage）
  - 低於 10% 過殺率（Overkill Rate）
  - 推論延遲小於 100ms（使用 RTX 4090）

## 實驗計畫
| 計畫 | 組合架構 | 目標場景 |
| :--- | :--- | :--- |
| 計畫 A | DINOv2/v3 + PatchCore | 穩定的基線系統，適用於一般異常檢測 |
| 計畫 B | DINOv3 + DAPT + SimpleNet + SALAD 雙流架構 | 專攻邏輯異常與複雜背景 |
| 計畫 C | PixIO-H + 線性頭 (Linear Head) | 微小瑕疵檢測，強化少樣本學習表現 |
| 計畫 D | PixIO/DINOv3 + MSFlow + HGAD | 多尺度特徵融合，構建統一檢測模型 |

## 系統需求
- 作業系統：Linux / Windows
- Python 版本：3.10 或更高版本
- 硬體推薦：NVIDIA RTX 4090 以達到最佳推論性能（低於 100ms）

## 安裝方式
本專案建議使用 uv 進行套件管理，以確保環境的一致性與安裝速度。

1. 安裝基礎依賴：
   ```bash
   uv sync
   ```

2. 安裝所有額外組件：
   ```bash
   uv sync --all-extras
   ```

## 快速開始
以下是使用預設設定啟動實驗計畫 A 的範例：

```bash
# 使用 Hydra 啟動實驗
python main.py experiment=plan_a
```

## 專案結構
```text
.
├── configs/          # Hydra 設定檔 (骨幹、檢測頭、實驗計畫等)
├── src/
│   ├── data/         # 資料處理與載入
│   ├── models/       # 模型定義 (backbones & heads)
│   ├── training/     # 訓練邏輯
│   ├── evaluation/   # 評估指標與驗證
│   ├── deploy/       # 部署相關程式碼 (如 Triton)
│   └── synthesis/    # 異常樣本合成
├── main.py           # 專案主入口
└── pyproject.toml    # 專案依賴管理
```

## 設定說明
本專案採用 Hydra 進行設定管理。所有配置均位於 configs/ 目錄下。

- backbone：選擇使用的骨幹網路（如 DINOv2, PixIO）。
- head：選擇使用的檢測頭（如 PatchCore, SimpleNet）。
- experiment：定義完整的實驗參數計畫。

您可以透過命令列覆寫任何參數：
```bash
python main.py backbone=dinov3 head=patchcore training.batch_size=32
```

## 相關連結
- [English README](README.md)

## 授權條款
本專案採用 MIT License 授權。
