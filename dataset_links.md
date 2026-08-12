# Comprehensive Deepfake Detection Datasets Repository



This document serves as the central directory providing official links to all datasets utilized throughout our research experiments. It includes reliable access to both the original, raw deepfake benchmark videos and their highly optimized, preprocessed versions. For contextual reference, the preprocessed datasets have been rigorously standardized to extract exactly 30 uniform frames per video, utilizing the SCRFD algorithm for precise face tracking, and are finally cropped to a 224x224 resolution to ensure consistency during model training.

---

## 1. Preprocessed Datasets (Optimized for Training & Evaluation)



The datasets listed in the table below have been thoroughly prepared and are completely ready for immediate integration into training and testing workflows. Each individual dataset has successfully passed through our customized 5-stage unified preprocessing pipeline. This comprehensive pipeline guarantees high data quality through the following specific operations: uniform sampling of exactly 30 frames, robust face detection powered by SCRFD combined with Intersection over Union (IoU) tracking, meticulous sharpness quality filtering to discard sub-optimal frames, and a final standardized 224x224 pixel resolution crop.

| Dataset Name | Benchmark Target | Kaggle Preprocessed Dataset URL |
| --- | --- | --- |
| **FaceForensics++ (FF++)**<br> | Primary Benchmark (c23 compression)| [kaggle.com/datasets/nghgb0101/ff-30frames](https://www.kaggle.com/datasets/nghgb0101/ff-30frames)<br> |
| **DFDC**<br> | DeepFake Detection Challenge (Subset 10)| [kaggle.com/datasets/thohintrnhngc/dfdc-preprocessing](https://www.kaggle.com/datasets/thohintrnhngc/dfdc-preprocessing)<br> |
| **Celeb-DF**<br> | Celeb-DF (v2)| [kaggle.com/datasets/ndhoang2310/new-evaluate-celeb-df](https://www.kaggle.com/datasets/ndhoang2310/new-evaluate-celeb-df)<br> |
| **WildDeepfake**<br> | In-the-Wild Cross-Test Benchmark| [kaggle.com/datasets/thohintrnhngc/wild-deepfake-30frame](https://www.kaggle.com/datasets/thohintrnhngc/wild-deepfake-30frame)<br> |

---

## 2. Original Raw Source Datasets



For researchers looking to access the baseline data or to replicate the preprocessing pipeline from scratch, we provide direct links to the untouched, original video collections. These datasets are provided exactly as they were initially released by their respective authors and institutions.

| Dataset Name | Source Repository |
| --- | --- |
| **FaceForensics++ (c23)**<br> | [Kaggle Benchmark: FF++ c23](https://www.kaggle.com/datasets/xdxd003/ff-c23)<br> |
| **Celeb-DF (v2)**<br> | [Kaggle Benchmark: Celeb-DF v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2)<br> |
| **DFDC-10**<br> | [Kaggle Benchmark: DFDC Part 10](https://www.kaggle.com/datasets/pranay22077/dfdc-10)<br> |
| **WildDeepfake**<br> | [HuggingFace Dataset: WildDeepfake](https://huggingface.co/datasets/xingjunm/WildDeepfake?library=datasets)<br> |