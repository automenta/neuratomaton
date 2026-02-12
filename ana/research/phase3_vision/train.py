import torch
import torch.nn as nn
import torch.optim as optim
from ana.research.phase3_vision.models import ANAVisionModel
from ana.config import ANAConfig

class VisionTrainer:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

if __name__ == "__main__":
    # Dummy data
    config = ANAConfig(d_model=64, patch_size=16)
    model = ANAVisionModel(config, num_classes=10)
    trainer = VisionTrainer(model)

    # Dummy dataloader
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 10, (4,))
    dataloader = [(images, labels)]

    loss = trainer.train_epoch(dataloader)
    print(f"Vision Training Loss: {loss:.4f}")
