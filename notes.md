# Notes: SOTA Anomaly Detection Survey 2024-2025

## User Requirements (Industrial Context)
- **Target**: 0% critical (obvious) miss, 50ppm escape rate, <5% overkill
- **Challenge**: Factory NG samples are scarce, OK samples accumulate slowly
- **Defect types**: Structural, logical, varying sizes and difficulty
- **Labels available**: Only OK/NG image-level labels (no polygon/pixel masks)
- **Sample count**: 1-200 training samples

---

## 1. AFR-CLIP (arXiv 2503.12910, Mar 2025)

### Architecture
**Stateless-to-Stateful Anomaly Feature Rectification**
- Takes CLIP features and rectifies them for anomaly awareness
- Image-guided textual rectification: embeds defect info into stateless prompts
- Text-on-text similarity yields anomaly maps

### Key Innovations
1. **Stateless prompts**: Describe object category without anomaly state
2. **Self Prompting (SP)**: Enhances multi-scale feature perception
3. **Multi-Patch Feature Aggregation (MPFA)**: Handles complex anomalies

### Benchmark Results
- Tested on 11 benchmarks (industrial + medical)
- Claims superiority in zero-shot AD
- **Zero training on target domain required**

### Relevance for Our Project
- **High priority**: Perfect for few-shot/zero-shot scenarios
- Can detect anomalies with 0 training samples
- Good for initial screening when starting with new product

---

## 2. PA-CLIP (arXiv 2503.01292, Mar 2025)

### Implementation Status (2026-01-17)
- Official repo: not found / not public (librarian search)
- Practical plan: treat as "paper-only" for now; can approximate with our PatchCore-style memory bank + synthetic pseudo anomaly generation if we decide to re-derive

### Architecture
**Pseudo-Anomaly Awareness for Zero-Shot AD**
- Reduces background noise via pseudo-anomaly framework
- Multiscale feature aggregation for global+local info
- Two memory banks: normal patterns vs pseudo-anomalies

### Key Innovations
1. **Pseudo-anomaly generation**: Creates synthetic defects for training
2. **Dual memory banks**: Distinguishes background from true anomalies
3. **Decision module**: Minimizes false positives from environmental variations

### Benchmark Results
- MVTec AD: Outperforms existing zero-shot methods
- VisA: Strong performance
- Focuses on reducing false positives (overkill)

### Relevance for Our Project
- **Medium-High priority**: Good for reducing overkill (<5% target)
- Memory bank approach similar to PatchCore but zero-shot
- Could be hybrid with our DINOv3 features

---

## 3. Dinomaly (CVPR 2025)

### Architecture
**Minimalist Reconstruction-Based Framework**
- Pure Transformer: Attention + MLP only
- Uses frozen DINOv2/v3 encoder + trainable decoder
- **LinearAttention2**: Cannot focus (ELU instead of softmax)
- Dropout bottleneck for noise injection

### Four Essential Components
1. **Scalable Foundation Transformers**: Universal features from DINO
2. **Noisy Bottleneck**: Dropout for regularization
3. **Linear Attention**: Prevents local copying
4. **Loose Reconstruction**: Not point-by-point

### Benchmark Results
| Dataset | Image AUROC | Pixel AUROC |
|---------|-------------|-------------|
| MVTec AD (multi-class) | **99.6%** | 97.8% |
| VisA | 98.7% | 98.2% |
| Real-IAD | - | - |

### Key Insight
- First multi-class UAD model competing with single-class SOTAs
- "Less is more" philosophy: simpler is better

### Our Implementation Status
- Already implemented in `src/models/heads/dinomaly.py`
- Tested on screw category: 89.58% (vs 83.77% PatchCore)
- **TODO**: Full MVTec evaluation

### Relevance for Our Project
- **Critical**: Already outperforms PatchCore on hard categories
- Multi-class support reduces deployment complexity
- Need to run full evaluation

---

## 4. SuperAD (CVPR 2025 VAND 3.0 Challenge)

### Implementation Status (2026-01-17)
- Official repo: not public (confirmed by librarian)
- Repro spec from paper:
  - Backbone: DINOv2 ViT-L/14
  - Layers: 6, 12, 18, 24
  - Reference images: 16 per category + greedy coreset selection
  - Input: short side 672 (some categories 448)
  - Background mask: PCA-based threshold tau=1.0 + morph kernel 3x3
- Practical plan: implement "SuperAD-like" variant inside our PatchCore pipeline (reference selection + pca background masking), but label as re-implementation (not official)

### Architecture
**Training-Free Method using DINOv2**
- Carefully selects small number of normal reference images
- Constructs memory bank from DINOv2 features
- Nearest neighbor matching for segmentation

### Key Innovations
1. **Reference image selection**: Quality over quantity
2. **DINOv2 representational power**: No training needed
3. **Designed for MVTec AD 2**: Complex lighting, real anomalies

### Benchmark Results
- Competitive on MVTec AD 2 test sets
- Training-free approach

### Relevance for Our Project
- **High priority**: Training-free = instant deployment
- Similar to our PatchCore but with better reference selection
- Good baseline for few-shot scenarios

---

## 5. MambaAD (NeurIPS 2024)

### Implementation Status (2026-01-17)
- Official repo: https://github.com/lewandofskee/MambaAD
- Our repo: implemented head `src/models/heads/mambaad.py`
- Note: upstream requires extra deps for full scanning:
  - `triton`, `causal_conv1d`, `mamba_ssm`, `numpy-hilbert-curve`, `pyzorder`
  - Our implementation has fallbacks if optional deps missing (scan_type falls back, mamba_ssm fallback)

### Architecture
**State Space Models for Multi-Class UAD**
- Pre-trained encoder + Mamba decoder
- Locality-Enhanced State Space (LSS) modules
- Hybrid State Space (HSS) blocks

### Key Innovations
1. **Mamba architecture**: O(N) vs O(N^2) attention
2. **Long-range + local modeling**: Best of both worlds
3. **Multi-class support**: Single model for all categories

### Benchmark Results
- Competitive with Transformer-based methods
- Lower computational cost
- First Mamba application to anomaly detection

### Relevance for Our Project
- **Medium priority**: Alternative to Transformer decoder
- Could replace Dinomaly's LinearAttention decoder
- More efficient for high-resolution images

---

## 6. AnomalyMoE (arXiv 2508.06203, Aug 2025)

### Architecture
**Language-Free Generalist Model**
- Mixture of Experts (MoE) for different anomaly types
- No language supervision (vs CLIP-based methods)
- Unified model for industrial, logical, medical domains

### Key Innovations
1. **MoE routing**: Different experts for different anomaly types
2. **Language-free**: Avoids CLIP's text-visual alignment issues
3. **Cross-domain generalization**

### Relevance for Our Project
- **Low-Medium priority**: Advanced architecture
- Could be future direction for unified detection
- Complex to implement

---

## 7. Few-Shot Methods Survey

### Implementation Status (2026-01-17)
- UniVAD: official repo https://github.com/FantasticGNU/UniVAD (CC BY-NC-SA 4.0; heavy deps: SAM-HQ + GroundingDINO)
- FADE: official repo https://github.com/BMVC-FADE/BMVC-FADE (MIT; poetry-based)
- AnoPLe: paper claims code link but repo not public / 404 (as of librarian search)
- Production implication: UniVAD license is non-commercial; FADE license is OK; need to decide if our project is research-only when using them

### UniVAD (arXiv 2412.03342, Dec 2024)
- Training-free unified model for few-shot VAD
- Works across industrial, logical, medical domains
- **Key**: No domain-specific training

### AnoPLe (arXiv 2408.13516, Aug 2024)
- Bi-directional prompt learning with CLIP
- Works with only normal samples
- Simulates anomalies for training

### FADE (arXiv 2409.00556, Aug 2024)
- Few-shot/Zero-shot using Large VLM
- GPT-4V style reasoning for anomaly detection
- Interpretable results

### IADGPT (arXiv 2508.10681, Aug 2025)
- Unified LVLM for few-shot IAD
- In-context learning for detection + localization + reasoning
- Most advanced VLM approach

### FIND (ICCV 2025)
- Few-shot multimodal anomaly detection (2D + 3D)
- Works with limited samples
- Addresses modality gap issue

---

## Strategy Recommendations for Limited Samples (1-200)

### Sample Count: 1-10 (Zero/Few-Shot)
**Best approaches:**
1. AFR-CLIP or PA-CLIP (zero-shot, no training)
2. SuperAD (training-free, just needs reference images)
3. UniVAD (few-shot, training-free)

**Expected performance:** 85-92% AUROC

### Sample Count: 10-50 (Few-Shot)
**Best approaches:**
1. DINOv3 + PatchCore (memory bank benefits from more samples)
2. AnoPLe (prompt learning with pseudo-anomalies)
3. SuperAD with more reference images

**Expected performance:** 90-96% AUROC

### Sample Count: 50-200 (Standard Few-Shot)
**Best approaches:**
1. DINOv3 + Dinomaly (trainable, benefits from more data)
2. DINOv3 + PatchCore with larger coreset
3. SALAD (for logical anomalies)

**Expected performance:** 95-99% AUROC

### Hybrid Strategy (Production Recommendation)
1. **Day 1**: Deploy SuperAD/AFR-CLIP (zero-shot baseline)
2. **Week 1 (10-50 samples)**: Switch to PatchCore
3. **Month 1 (50-200 samples)**: Train Dinomaly for hard categories
4. **Ongoing**: Ensemble SuperAD + PatchCore + Dinomaly

---

## New Backbones to Test

Based on research and user requirements:

### Priority 1 (Already Implemented)
- [x] DINOv2-Large (current best)
- [x] DINOv3-Large (96.51% AUROC achieved)
- [x] Swin-Base (87.73% AUROC)
- [x] PixIO (needs testing)

### Priority 2 (Need Implementation/Verification)
- [ ] DINOv2-Giant (1.1B params) - highest quality features
- [ ] SigLIP (CLIP alternative with better vision)
- [ ] CLIP ViT-L/14 (for zero-shot methods)
- [ ] EVA-02 (alternative foundation model)

### Priority 3 (Research)
- [ ] Mamba-based backbones
- [ ] PixIO-H (when released)
- [ ] DINOv3-7B (requires significant GPU)

---

## Updated Experiment Priority

### Immediate (This Week)
1. Run Dinomaly on full MVTec AD (not just screw)
2. Implement AFR-CLIP for zero-shot baseline
3. Test image size 448/518 with PatchCore

### Short-term (Next 2 Weeks)
4. Implement universal dataset interface
5. Run few-shot experiments (k=1,5,10,20,50)
6. Complete SALAD training on MVTec LOCO

### Medium-term (Next Month)
7. Implement MambaAD decoder
8. Test DINOv2-Giant backbone
9. Create production-ready ensemble
