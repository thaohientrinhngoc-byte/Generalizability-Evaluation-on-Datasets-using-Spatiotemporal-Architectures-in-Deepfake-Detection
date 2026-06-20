# Deepfake Detection Datasets

This document provides official links to both the raw deepfake benchmarks and the preprocessed versions (30 uniform frames, SCRFD face tracking & 224x224 crop) utilized in the paper experiments.

---

## 1. Preprocessed Datasets (Ready for Training & Testing)

These datasets have been processed through our 5-stage unified pipeline (Uniform 30-frame sampling, SCRFD face detection + IoU tracking, sharpness quality filtering, 224x224 resolution):

| Dataset Name | Benchmark Target | Kaggle Preprocessed Dataset URL |
| :--- | :--- | :--- |
| **FaceForensics++ (FF++)** | Primary Benchmark (c23 compression) | [kaggle.com/datasets/nghgb0101/ff-30frames](https://www.kaggle.com/datasets/nghgb0101/ff-30frames) |
| **DFDC** | DeepFake Detection Challenge (Subset 10) | [kaggle.com/datasets/thohintrnhngc/dfdc-preprocessing](https://www.kaggle.com/datasets/thohintrnhngc/dfdc-preprocessing) |
| **Celeb-DF** | Celeb-DF (v2) | [kaggle.com/datasets/ndhoang2310/new-evaluate-celeb-df](https://www.kaggle.com/datasets/ndhoang2310/new-evaluate-celeb-df) |
| **WildDeepfake** | In-the-Wild Cross-Test Benchmark | [kaggle.com/datasets/thohintrnhngc/wild-deepfake-30frame](https://www.kaggle.com/datasets/thohintrnhngc/wild-deepfake-30frame) |

---

## 2. Raw Source Datasets

Original video sources as released by their respective authors:

| Dataset Name | Source Repository |
| :--- | :--- |
| **FaceForensics++ (c23)** | [Kaggle Benchmark: FF++ c23](https://www.kaggle.com/datasets/xdxd003/ff-c23) |
| **Celeb-DF (v2)** | [Kaggle Benchmark: Celeb-DF v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) |
| **DFDC-10** | [Kaggle Benchmark: DFDC Part 10](https://www.kaggle.com/datasets/pranay22077/dfdc-10) |
| **WildDeepfake** | [HuggingFace Dataset: WildDeepfake](https://huggingface.co/datasets/xingjunm/WildDeepfake?library=datasets) |
