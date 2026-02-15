
from ana.research.core import ExperimentBase, ExperimentRegistry
from ana.models.config import ANAConfig
from ana.models.core import ANASeriesModel
from ana.utils.datasets import SeriesPredictionTask
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json

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
        best_loss = float('inf')
        step = 0

        model.train()
        while step < steps:
            for x, y, mask in loader:
                # x: [B, Seq, Dim]
                pred, info = model.forward_sequence(x)

                loss = F.mse_loss(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    # torch.save(model.state_dict(), os.path.join(self.results.output_dir, "best_series_model.pt"))

                step += 1

                if step % 10 == 0:
                    self.results.log(f"Step {step}/{steps}: Loss: {loss.item():.4f}")

                if step >= steps:
                    break

        # Save results
        with open(os.path.join(self.results.output_dir, "series_loss.json"), 'w') as f:
            json.dump({'losses': losses}, f)

        # Visualize
        self.plot_loss(losses)
        self.visualize_predictions(model, dataset)

        self.results.log("Phase 5 Complete.")

    def plot_loss(self, losses):
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title("Series Prediction Training Loss")
        plt.xlabel("Step")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.savefig(os.path.join(self.results.output_dir, "loss.png"))
        plt.close()

    def visualize_predictions(self, model, dataset):
        model.eval()
        x, y, mask = dataset[0]
        # x: [Seq, Dim] -> Add batch dim
        x_b = x.unsqueeze(0)

        with torch.no_grad():
            pred, info = model.forward_sequence(x_b)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(y[:, 0].numpy(), label="Ground Truth", linestyle='--')
        plt.plot(pred[0, :, 0].numpy(), label="Prediction")
        plt.title("Series Prediction (Sample 0)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.results.output_dir, "prediction.png"))
        plt.close()

        # Plot Gating if available
        # Need to implement extraction from info_log
        pass
