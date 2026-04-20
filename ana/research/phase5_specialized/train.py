import torch
import torch.nn as nn
import torch.optim as optim
from ana.research.phase5_specialized.models import ANASeriesModel
from ana.config import ANAConfig
from ana.research.core import ExperimentBase, ExperimentRegistry

@ExperimentRegistry.register(phase=5, name="train_series")
class SeriesTrainerExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "train_series"

    @property
    def phase(self) -> int:
        return 5

    def setup(self):
        self.model = ANASeriesModel(self.config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for seq in dataloader:
            seq = seq.to(self.device)
            # Input: x_t, Target: x_{t+1} (Autoregressive)
            x = seq[:, :-1, :]
            y = seq[:, 1:, :]

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def execute(self):
        self.results.log("Running Series Training...")
        # Dummy data: Sine wave
        t = torch.linspace(0, 10, 100)
        data = torch.sin(t).view(1, 100, 1)
        dataloader = [data]

        loss = self.train_epoch(dataloader)
        self.results.log(f"Series Training Loss: {loss:.4f}")
        self.results.save_json("series_results.json", {"loss": loss})

if __name__ == "__main__":
    config = ANAConfig(series_dim=1, d_model=32)
    exp = SeriesTrainerExperiment(config)
    exp.run()
