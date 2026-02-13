import torch
import torch.nn as nn
import torch.optim as optim
from ana.research.phase3_vision.models import ANAVisionModel
from ana.config import ANAConfig
import matplotlib.pyplot as plt
import os

class VisionTrainer:
    def __init__(self, model, device="cpu"):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        self.results_dir = "results/phase3_vision"
        os.makedirs(self.results_dir, exist_ok=True)

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

        save_path = os.path.join(self.results_dir, f"epoch_{epoch}_preds.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Predictions saved to {save_path}")

if __name__ == "__main__":
    # Dummy data
    config = ANAConfig(d_model=64, patch_size=16)
    model = ANAVisionModel(config, num_classes=10)
    trainer = VisionTrainer(model)

    # Dummy dataloader
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 10, (4,))
    dataloader = [(images, labels)]

    # Train
    loss = trainer.train_epoch(dataloader)
    print(f"Vision Training Loss: {loss:.4f}")

    # Visualize
    outputs = model(images) # re-run for viz
    trainer.save_predictions(images, labels, outputs)
