import torch
import torch.nn as nn
import torch.optim as optim
from ana.research.phase5_specialized.models import ANASeriesModel
from ana.config import ANAConfig

class SeriesTrainer:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss() # Typically MSE for regression/time-series

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

if __name__ == "__main__":
    config = ANAConfig(series_dim=1, d_model=32)
    model = ANASeriesModel(config)
    trainer = SeriesTrainer(model)

    # Dummy data: Sine wave
    t = torch.linspace(0, 10, 100)
    data = torch.sin(t).view(1, 100, 1)
    dataloader = [data]

    loss = trainer.train_epoch(dataloader)
    print(f"Series Training Loss: {loss:.4f}")
