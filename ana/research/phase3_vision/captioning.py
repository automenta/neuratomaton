import torch
from ana.models import ANAModel
from ana.config import ANAConfig
from ana.research.phase3_vision.models import ANAVisionCaptioner
from ana.research.core import ExperimentBase, ExperimentRegistry

@ExperimentRegistry.register(phase=3, name="captioning")
class CaptioningExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "captioning"

    @property
    def phase(self) -> int:
        return 3

    def setup(self):
        self.model = ANAVisionCaptioner(self.config).to(self.device)

    def execute(self):
        self.results.log("Running Captioning Model Demo...")
        images = torch.randn(1, 3, 224, 224).to(self.device)
        text_ids = torch.randint(0, self.config.vocab_size, (1, 10)).to(self.device)
        out = self.model(images, text_ids)
        self.results.log(f"Captioning Output: {out.shape}")
        self.results.save_json("captioning_results.json", {"output_shape": list(out.shape)})

if __name__ == "__main__":
    config = ANAConfig(d_model=64, patch_size=16, vocab_size=100)
    exp = CaptioningExperiment(config)
    exp.run()
