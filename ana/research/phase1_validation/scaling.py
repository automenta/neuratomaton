import torch
import torch.nn as nn
import torch.optim as optim
from ana.models import ANAModel
from ana.config import ANAConfig
import time

class ScalingExperiment:
    def __init__(self, device="cpu"):
        self.device = device
        self.results = {}

    def get_model_size(self, model):
        return sum(p.numel() for p in model.parameters())

    def run_experiment(self, configs):
        for name, config in configs.items():
            print(f"\n--- Training Model: {name} ---")
            model = ANAModel(config).to(self.device)
            size = self.get_model_size(model)
            print(f"Parameters: {size:,}")

            # Dummy training loop
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            start_time = time.time()

            # Simulate 10 steps
            for i in range(10):
                input_ids = torch.randint(0, config.vocab_size, (config.batch_size, 64)).to(self.device)
                logits, _ = model(input_ids)
                loss = logits.mean() # Dummy loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            elapsed = time.time() - start_time
            print(f"Training completed in {elapsed:.2f}s")
            self.results[name] = {"params": size, "time": elapsed}

if __name__ == "__main__":
    # Small configs for quick verification
    configs = {
        "Tiny": ANAConfig(d_model=32, num_layers=1, state_dim=32),
        "Small": ANAConfig(d_model=64, num_layers=2, state_dim=64)
    }
    exp = ScalingExperiment()
    exp.run_experiment(configs)
