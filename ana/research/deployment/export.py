import torch
import torch.nn as nn
from ana.models import ANAModel
from ana.config import ANAConfig
import os

def export_to_onnx(model, dummy_input, filepath="ana_model.onnx"):
    """
    Exports the ANA model to ONNX format.

    Note: Parallel scan ops might need custom ONNX support or fallback to sequential.
    Here we export the sequential version which is more likely to be supported.
    """
    print(f"Exporting model to {filepath}...")
    model.eval()

    # Ensure sequential mode is used for export compatibility
    original_scan = model.config.use_parallel_scan
    model.config.use_parallel_scan = False

    try:
        # Using fixed size for simplicity in this demo framework.
        torch.onnx.export(
            model,
            dummy_input,
            filepath,
            export_params=True,
            opset_version=18, # Use latest stable
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits', 'info_log'],
        )
        print("Export successful.")
    except Exception as e:
        print(f"Export failed: {e}")
    finally:
        model.config.use_parallel_scan = original_scan

if __name__ == "__main__":
    config = ANAConfig(vocab_size=100, d_model=32, num_layers=1)
    model = ANAModel(config)
    dummy = torch.randint(0, 100, (1, 32))
    export_to_onnx(model, dummy)

    if os.path.exists("ana_model.onnx"):
        print("Verified ONNX file exists.")
        os.remove("ana_model.onnx")
        if os.path.exists("ana_model.onnx.data"):
            os.remove("ana_model.onnx.data")
