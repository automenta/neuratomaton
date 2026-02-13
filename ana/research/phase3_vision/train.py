import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from ana.research.phase3_vision.models import ANAVisionModel
from ana.config import ANAConfig
from ana.research.core import ExperimentBase, ExperimentRegistry

@ExperimentRegistry.register(phase=3, name="train_vision")
class VisionTrainerExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "train_vision"

    @property
    def phase(self) -> int:
        return 3

    def setup(self):
        self.model = ANAVisionModel(self.config, num_classes=10).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
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

    def save_predictions(self, images, labels, outputs, epoch=0):
        """
        Saves a grid of images with predicted vs actual labels.
        """
        self.model.eval()
        with torch.no_grad():
            preds = outputs.argmax(dim=1)

        fig, axes = plt.subplots(1, min(len(images), 4), figsize=(12, 4))
        if len(images) == 1: axes = [axes]

        for i, ax in enumerate(axes):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            # Normalize for display if needed (assuming standard normalization)
            # img = img * std + mean
            # For dummy random data, just clip
            img = (img - img.min()) / (img.max() - img.min())

            ax.imshow(img)
            ax.set_title(f"Pred: {preds[i].item()} | True: {labels[i].item()}")
            ax.axis('off')

        self.results.save_plot(f"epoch_{epoch}_preds.png")

    def execute(self):
        # Dummy data
        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 10, (4,))
        dataloader = [(images, labels)]

        # Train
        loss = self.train_epoch(dataloader)
        self.results.log(f"Vision Training Loss: {loss:.4f}")
        self.results.save_json("training_results.json", {"loss": loss})

        # Visualize
        outputs = self.model(images.to(self.device))
        self.save_predictions(images, labels, outputs)

if __name__ == "__main__":
    config = ANAConfig(d_model=64, patch_size=16)
    exp = VisionTrainerExperiment(config)
    exp.run()
