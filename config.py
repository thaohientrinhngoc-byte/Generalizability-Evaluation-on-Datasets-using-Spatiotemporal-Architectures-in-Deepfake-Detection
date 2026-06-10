"""
Central Configuration for Deepfake Spatiotemporal Generalizability Evaluation.
"""

import os

# Default Hyperparameters
NUM_FRAMES = 30
FRAME_SIZE = 224
BATCH_SIZE = 32
DEFAULT_LR = 1e-4
RANDOM_SEED = 42

# Preprocessing Defaults
DEFAULT_CROP_MARGIN = 0.4
DEFAULT_MIN_FACE_SCORE = 0.5
DEFAULT_MIN_FACE_PIXELS = 30
DEFAULT_MIN_TOTAL_FRAMES = 60
DEFAULT_MAX_MISSING_FRAC = 0.20
DEFAULT_MAX_CONSEC_GAP = 5

# Paper Model Configuration Naming Alignment
MODEL_NAME_MAP = {
    "M_CNN_F": "M_CNN-F (Frozen CNN Backbone + GAP)",
    "M_CNN_T": "M_CNN-T (Tuned CNN Backbone Blocks 7-8 + GAP)",
    "M_CL_F":  "M_CL-F (Frozen CNN Backbone + 3-Layer LSTM)",
    "M_CL_T":  "M_CL-T (Tuned CNN Backbone Blocks 7-8 + 3-Layer LSTM)",
}

# ImageNet Normalization Constants (Baked into PyTorch model GPU buffers)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
