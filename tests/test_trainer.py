import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os
import shutil
from ana.training.utils import Trainer

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

def test_trainer_fit_and_checkpointing():
    # Setup
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoint_dir = "tests/checkpoints_test"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device="cpu",
        checkpoint_dir=checkpoint_dir,
        log_interval=1
    )

    # Data
    x = torch.randn(10, 10)
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=2)

    # Train
    trainer.fit(dataloader, epochs=2)

    # Check checkpoints
    assert os.path.exists(os.path.join(checkpoint_dir, "checkpoint_epoch_0.pt"))
    assert os.path.exists(os.path.join(checkpoint_dir, "checkpoint_epoch_1.pt"))

    # Load Checkpoint
    loaded_model = SimpleModel()
    loaded_optimizer = torch.optim.Adam(loaded_model.parameters(), lr=0.01)
    loaded_trainer = Trainer(
        model=loaded_model,
        optimizer=loaded_optimizer,
        device="cpu",
        checkpoint_dir=checkpoint_dir
    )

    loaded_trainer.load_checkpoint("checkpoint_epoch_1.pt")

    # Verify weights
    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.allclose(p1, p2)

    # Cleanup
    shutil.rmtree(checkpoint_dir)

if __name__ == "__main__":
    test_trainer_fit_and_checkpointing()
