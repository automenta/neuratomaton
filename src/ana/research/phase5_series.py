
from ana.research.core import ExperimentBase, ExperimentRegistry
from ana.models.config import ANAConfig
from ana.models.core import ANASeriesModel
from ana.utils.datasets import SeriesPredictionTask
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

@ExperimentRegistry.register(phase=5, name="series")
class SeriesExperiment(ExperimentBase):
    @property
    def name(self) -> str: return "series"
    @property
    def phase(self) -> int: return 5

    def execute(self, quick: bool = False, **kwargs):
        self.results.log("Starting Phase 5: Series Prediction")

        # Hyperparameters
        seq_len = 32
        dim = 1
        steps = 50 if quick else 500
        batch_size = 16

        config = ANAConfig(
            d_model=32, state_dim=32, num_layers=2, track_count=2,
            series_dim=dim,
            max_thinking_steps=0,
            use_parallel_scan=True
        )

        model = ANASeriesModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        dataset = SeriesPredictionTask(num_samples=1000, seq_len=seq_len, dim=dim)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        step = 0

        model.train()
        while step < steps:
            for x, y, mask in loader:
                # x: [B, Seq, Dim]
                pred = model.forward_sequence(x)

                loss = F.mse_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                step += 1

                if step % 10 == 0:
                    self.results.log(f"Step {step}/{steps}: Loss: {loss.item():.4f}")

                if step >= steps:
                    break

        # Save results
        import json
        with open(self.results.output_dir + "/series_loss.json", 'w') as f:
            json.dump({'losses': losses}, f)

        self.results.log("Phase 5 Complete.")
