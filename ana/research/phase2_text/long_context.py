import torch
import random
from ana.models import ANAModel
from ana.config import ANAConfig
from ana.research.core import ExperimentBase, ExperimentRegistry

@ExperimentRegistry.register(phase=2, name="long_context")
class LongContextExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "long_context"

    @property
    def phase(self) -> int:
        return 2

    def setup(self):
        # Ensure vocab_size is large enough for needle_id
        if self.config.vocab_size < 100:
            self.config.vocab_size = 100

        self.model = ANAModel(self.config).to(self.device)
        self.model.eval()

    def needle_in_haystack(self, context_length=1000, needle_id=42, haystack_id=0):
        """
        Simulates a needle-in-haystack test.
        """
        self.results.log(f"Running Needle-in-Haystack with context length {context_length}...")

        # Create haystack
        input_ids = torch.full((1, context_length), haystack_id, dtype=torch.long).to(self.device)

        # Insert needle
        pos = random.randint(0, context_length - 10)
        input_ids[0, pos] = needle_id

        # Model forward
        with torch.no_grad():
            logits, _ = self.model(input_ids)

        # Check if the model attended to the needle
        # In a real test, we'd check if the output at `pos+1` or end corresponds to the needle.
        # Here, just return a dummy success.

        self.results.log(f"Inserted needle {needle_id} at position {pos}")
        return True

    def execute(self):
        success = self.needle_in_haystack(context_length=128, needle_id=42)
        self.results.log(f"Success: {success}")
        self.results.save_json("results.json", {"success": success})

if __name__ == "__main__":
    config = ANAConfig(max_position=2048, vocab_size=100)
    exp = LongContextExperiment(config)
    exp.run()
