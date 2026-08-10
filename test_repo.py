"""
Verification test for model initialization, forward pass, and DGS metric.
"""

import torch
from models import SpatiotemporalDeepfakeDetector
from utils import compute_dgs_score, evaluate_predictions


def test_models():
    print("Testing Spatiotemporal Models forward passes...")
    dummy_input = torch.randn(2, 30, 3, 224, 224)  # Batch=2, Frames=30, C=3, H=224, W=224

    for m_type in ["M_CNN_F", "M_CNN_T", "M_CL_F", "M_CL_T"]:
        model = SpatiotemporalDeepfakeDetector(model_type=m_type)
        output = model(dummy_input)
        assert output.shape == (2,), f"Output shape mismatch for {m_type}: expected (2,), got {output.shape}"
        print(f"  [SUCCESS] {m_type} forward pass OK -> output shape: {output.shape}")


def test_dgs():
    print("\nTesting DGS Score calculation...")
    # Example: FF++ internal AUC = 0.8608, cross AUCs = [0.6874, 0.7500, 0.6180]
    a_internal = 0.8608
    a_cross = [0.6874, 0.7500, 0.6180]
    dgs = compute_dgs_score(a_internal, a_cross)
    print(f"  [SUCCESS] Calculated DGS score: {dgs:.4f}")


if __name__ == "__main__":
    test_models()
    test_dgs()
    print("\nAll verification tests passed!")
