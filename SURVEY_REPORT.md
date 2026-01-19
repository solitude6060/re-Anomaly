# Anomaly Detection Survey (2023-2026)

本文整理 2023-2026 年各領域異常檢測的核心方法，僅收錄「論文 + 官方/主要 code」皆可取得者。若某領域可用的官方 code 較少，會明確標註數量較少。

## 工業/視覺檢測 (MVTec AD/LOCO 等)

- AnomalyCLIP (2023, ICLR 2024)
  - 概念：學習 object-agnostic prompts，將 CLIP 的語義空間對齊「正常/異常」而非物件語義，提升跨類別泛化。
  - Paper: https://arxiv.org/abs/2310.18961
  - Code (official): https://github.com/zqhang/AnomalyCLIP
- AF-CLIP (2025, ACM MM 2025)
  - 概念：強化 CLIP 的局部特徵，加入多尺度聚合 + anomaly-focused adapter，強化局部異常定位。
  - Paper: https://arxiv.org/abs/2507.19949
  - Code (official): https://github.com/Faustinaqq/AF-CLIP
- AD-DINOv3 (2025)
  - 概念：將 DINOv3 作為視覺 backbone，結合輕量對齊與 anomaly-aware calibration。
  - Paper: https://arxiv.org/abs/2509.14084
  - Code (official): https://github.com/Kaisor-Yuan/AD-DINOv3
- ACD-CLIP (2025)
  - 概念：Conv-LoRA 注入局部 inductive bias + 動態 cross-modal 融合，提升 zero-shot anomaly segmentation。
  - Paper: https://arxiv.org/abs/2508.07819
  - Code (official): https://github.com/cockmake/ACD-CLIP

## 醫學影像

- AnyAD (2025)
  - 概念：任意 MRI 模態組合下的 anomaly detection，透過 DINOv2 + 特徵分佈對齊與 INP-guided 重建。
  - Paper: https://arxiv.org/abs/2512.21264
  - Code (official): https://github.com/wuchangw/AnyAD
- MADPOT (2025)
  - 概念：CLIP adapters + Partial Optimal Transport 對齊局部特徵，提升醫療零/小樣本異常偵測。
  - Paper: https://arxiv.org/abs/2507.06733
  - Code (official): https://github.com/mahshid1998/MADPOT
- Conditioned Diffusion Models for UAD (2025)
  - 概念：條件式 diffusion 引導 healthy reconstruction，降低假陽性並提升異常分割。
  - Paper: https://arxiv.org/abs/2312.04215
  - Code (official): https://github.com/FinnBehrendt/Conditioned-Diffusion-Models-UAD
- DinoAtten3D (2025)
  - 概念：DINOv2 slice-level attention 聚合 3D MRI，強化異常分類與小資料泛化。
  - Paper: https://arxiv.org/abs/2509.12512
  - Code (official): https://github.com/Rafsani/DinoAtten3D

## 時序異常偵測 (Time-Series)

- FusAD (2025, ICDE 2026)
  - 概念：時間-頻率融合 + 自適應去噪，多任務 time-series anomaly detection。
  - Paper: https://arxiv.org/abs/2512.14078
  - Code (official): https://github.com/zhangda1018/FusAD
- Pi-Transformer (2025)
  - 概念：physics-informed attention，融合重建誤差與相位/時間錯位訊號。
  - Paper: https://arxiv.org/abs/2509.19985
  - Code (official): https://github.com/sepehr-m/Pi-Transformer

## 影片異常偵測 (Video AD)

- AnomalyCLIP for Video (2023)
  - 概念：CLIP latent-space + temporal modeling，用於影片弱監督異常辨識。
  - Paper: https://arxiv.org/abs/2310.02835
  - Code (official): https://luca-zanella-dvl.github.io/AnomalyCLIP/
- DE-Net (Dynamic Erasing Network) (2023)
  - 概念：多尺度時間特徵 + 動態抑制高置信異常段落，促進完整異常定位。
  - Paper: https://arxiv.org/abs/2312.01764
  - Code (official): https://github.com/ArielZc/DE-Net
- VLLM Anomaly Recognition (2025)
  - 概念：以 VLM 做 zero-shot anomaly recognition / evaluation。
  - Paper: https://arxiv.org/abs/2510.23190
  - Code (official): https://github.com/pascalbenschopTU/VLLM_AnomalyRecognition

## 語言/LLM 與 Log 異常

- GPT-4V-AD (2023)
  - 概念：以 GPT-4V VQA pipeline 做 zero-shot anomaly detection。
  - Paper: https://arxiv.org/abs/2311.02612
  - Code (official): https://github.com/zhangzjn/GPT-4V-AD
- CoLog (2025)
  - 概念：多模態 log anomaly detection，協同 transformer 處理多維 log modalities。
  - Paper: https://arxiv.org/abs/2512.23380
  - Code (official): https://github.com/NasirzadehMoh/CoLog

## 多模態 (視覺 + 感測/Log)

- AnyAD (2025) — MRI multi-sequence
  - Paper: https://arxiv.org/abs/2512.21264
  - Code (official): https://github.com/wuchangw/AnyAD
- CoLog (2025) — log modalities
  - Paper: https://arxiv.org/abs/2512.23380
  - Code (official): https://github.com/NasirzadehMoh/CoLog

---

## 建議新增的 5 個方法 (可加入本 repo 實驗)

以下 5 個方法與本專案的視覺異常檢測場景最相符，且有可用 code：

1. AnomalyCLIP
   - 理由：零樣本異常基準方法，CLIP prompt 學習可直接融入現有 CLIP backbone。
2. AF-CLIP
   - 理由：強調 anomaly-focused adapter + 多尺度，與現有 patch-based pipeline 相容。
3. AD-DINOv3
   - 理由：直接利用 DINOv3，與本 repo 的 DINOv3 backbone 相契合。
4. ACD-CLIP
   - 理由：Conv-LoRA + 動態融合，為 zero-shot segmentation 提供更強基線。
5. MADPOT (醫學/跨域，可作為 cross-domain baseline)
   - 理由：CLIP + POT prompt alignment，適合跨域泛化實驗。

如果你要我把這 5 個方法正式加入實驗矩陣，我會先建立 `heads/` 介面與對應 registry，再做 smoke test 與完整評測。
