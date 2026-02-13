import torch
import torch.nn as nn
import os
from ana.models import ANAModel
from ana.config import ANAConfig
from ana.research.core import ExperimentBase, ExperimentRegistry

@ExperimentRegistry.register(phase=6, name="export_onnx")
class OnnxExportExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "export_onnx"

    @property
    def phase(self) -> int:
        return 6

    def setup(self):
        self.model = ANAModel(self.config).to("cpu") # Export usually on CPU
        self.dummy_input = torch.randint(0, self.config.vocab_size, (1, 32))

    def export_to_onnx(self, filepath="ana_model.onnx"):
        """
        Exports the ANA model to ONNX format.
        """
        self.results.log(f"Exporting model to {filepath}...")
        self.model.eval()

        # Ensure sequential mode is used for export compatibility
        original_scan = self.model.config.use_parallel_scan
        self.model.config.use_parallel_scan = False

        try:
            torch.onnx.export(
                self.model,
                self.dummy_input,
                filepath,
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits', 'info_log'],
            )
            self.results.log("Export successful.")
            return True
        except Exception as e:
            self.results.log(f"Export failed: {e}")
            return False
        finally:
            self.model.config.use_parallel_scan = original_scan

    def execute(self):
        filepath = self.results.get_path("ana_research_model.onnx")
        success = self.export_to_onnx(filepath)

        if success and os.path.exists(filepath):
            self.results.log("Verified ONNX file exists.")
            # We don't delete it here as the ResultManager keeps artifacts.
            self.results.save_json("export_results.json", {"success": True, "path": filepath})
        else:
            self.results.save_json("export_results.json", {"success": False})

if __name__ == "__main__":
    config = ANAConfig(vocab_size=100, d_model=32, num_layers=1)
    exp = OnnxExportExperiment(config)
    exp.run()
