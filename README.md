# Generalizability Evaluation on Datasets using Spatiotemporal Architectures in Deepfake Detection

Official implementation accompanying the research paper: **"Generalizability Evaluation on Datasets using Spatiotemporal Architectures in Deepfake Detection"**  
*Faculty of Artificial Intelligence, FPT University, Danang Campus, Vietnam*

---

## 📌 Abstract

Deepfake detection plays an essential role in mitigating the negative social impacts of AI-manipulated videos. However, existing detectors often struggle to generalize when deployed in real-world scenarios due to dataset bias. In this work, an experimental framework is designed to systematically evaluate the generalizability of datasets using spatiotemporal architectures. 

Our framework employs an **EfficientNet-B0** backbone to extract spatial features, followed by either a **Global Average Pooling (GAP)** layer or a **3-layer Long Short-Term Memory (LSTM)** module to capture temporal dynamics. Intra-dataset and inter-dataset evaluations are conducted across four major benchmarks:
1. **FaceForensics++ (FF++)** (c23 compression)
2. **DeepFake Detection Challenge (DFDC-10)**
3. **Celeb-DF (v2)**
4. **WildDeepfake** (In-the-wild out-of-distribution benchmark)

Additionally, we introduce the **Dataset Generalizability Score (DGS)** to quantify dataset transferability across model architectures.

---

## 🏗️ Model Configurations (Table I Alignment)

All models share the same EfficientNet-B0 visual backbone pre-trained on ImageNet, evaluated under two backbone adaptation strategies (Fully Frozen vs. Fine-tuning Blocks 7 & 8) and two temporal aggregation heads (GAP vs. 3-Layer LSTM):

| Configuration Name | LaTeX Symbol | Backbone Adaptation Strategy | Temporal Modeling Head | Classification Layer |
| :--- | :--- | :--- | :--- | :--- |
| **`M_CNN_F`** | $M_{CNN-F}$ | Frozen (All parameters fixed) | Global Average Pooling | Linear Layer |
| **`M_CNN_T`** | $M_{CNN-T}$ | Fine-tuned (Blocks 7 & 8 learnable) | Global Average Pooling | Linear Layer |
| **`M_CL_F`** | $M_{CL-F}$ | Frozen (All parameters fixed) | 3-Layer LSTM | Linear Layer |
| **`M_CL_T`** | $M_{CL-T}$ | Fine-tuned (Blocks 7 & 8 learnable) | 3-Layer LSTM | Linear Layer |

---

## 📊 Dataset Generalizability Score (DGS)

The **Dataset Generalizability Score (DGS)** evaluates how effectively knowledge learned from a given training dataset transfers to unseen external benchmark datasets:

$$\text{DGS} = \bar{A}_{\text{cross}} \times \left( \frac{\bar{A}_{\text{cross}}}{A_{\text{internal}}} \right) \times (1 - \sigma_{\text{cross}})$$

Where:
- $\bar{A}_{\text{cross}}$: Mean classification performance (e.g. AUC) across all out-of-distribution cross-test datasets.
- $A_{\text{internal}}$: In-distribution test performance on the training dataset's testing split.
- $\sigma_{\text{cross}}$: Standard deviation of cross-dataset performances (penalizes variance across different evaluation domains).

---

## 📂 Preprocessed Datasets & Downloads

See [`dataset_links.md`](dataset_links.md) for direct download links:

* **Preprocessed Datasets (Uniform 30 frames, SCRFD face crops, 224x224):**
  * FaceForensics++ (c23): [Kaggle Dataset](https://www.kaggle.com/datasets/nghgb0101/ff-30frames)
  * DFDC (Subset 10): [Kaggle Dataset](https://www.kaggle.com/datasets/thohintrnhngc/dfdc-preprocessing)
  * Celeb-DF (v2): [Kaggle Dataset](https://www.kaggle.com/datasets/ndhoang2310/new-evaluate-celeb-df)
  * WildDeepfake: [Kaggle Dataset](https://www.kaggle.com/datasets/thohintrnhngc/wild-deepfake-30frame)

---

## 🚀 Quick Start

### 1. Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/thaohientrinhngoc-byte/Deepfakes-Video-Detection.git
cd Deepfakes-Video-Detection
pip install -r requirements.txt
```

### 2. Dataset Preprocessing

To extract 30 uniform frames per video with SCRFD face detection and IoU tracking:

```bash
python preprocess.py \
  --split_file /path/to/dataset_split.csv \
  --output_dir ./processed_dataset \
  --num_frames 30 \
  --target_size 224 \
  --crop_margin 0.4
```

### 3. Model Training

Train any of the 4 paper configurations (`M_CNN_F`, `M_CNN_T`, `M_CL_F`, `M_CL_T`):

```bash
# Train Tuned CNN-LSTM model (M_CL_T)
python train.py \
  --model M_CL_T \
  --data_dir ./processed_dataset \
  --output_dir ./models_checkpoints \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.0001
```

### 4. Cross-Dataset Matrix Evaluation

To evaluate trained checkpoints across all 4 benchmark datasets:

```bash
python evaluate.py \
  --checkpoints_dir ./models_checkpoints \
  --datasets_config datasets.json \
  --output_dir ./results
```

---

## 📄 Citation

If you find this repository useful for your research, please cite:

```bibtex
@inproceedings{nguyen2026generalizability,
  title={Generalizability Evaluation on Datasets using Spatiotemporal Architectures in Deepfake Detection},
  author={Nguyen Dinh Hoang and Ngoc Thao Hien Trinh and Nguyen Huu Gia Bao and Quoc-Trinh Vo},
  booktitle={2026 International Conference on Cognitive and Intelligent Systems (ICogSys 2026)},
  year={2026}
}
```
