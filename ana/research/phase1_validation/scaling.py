import torch
import torch.nn as nn
import torch.optim as optim
from ana.models import ANAModel
from ana.config import ANAConfig
import time
import matplotlib.pyplot as plt
import os

class ScalingExperiment:
    def __init__(self, device="cpu"):
        self.device = device
        self.results = {}
        self.results_dir = "results/phase1_scaling"
        os.makedirs(self.results_dir, exist_ok=True)

    def get_model_size(self, model):
        return sum(p.numel() for p in model.parameters())

    def run_experiment(self, configs):
        model_sizes = []
        training_times = []

        for name, config in configs.items():
            print(f"\n--- Training Model: {name} ---")
            model = ANAModel(config).to(self.device)
            size = self.get_model_size(model)
            print(f"Parameters: {size:,}")

            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
            start_time = time.time()

            # Simulate 10 steps
            for i in range(10):
                input_ids = torch.randint(0, config.vocab_size, (config.batch_size, 64)).to(self.device)
                logits, _ = model(input_ids)
                loss = logits.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            elapsed = time.time() - start_time
            print(f"Training completed in {elapsed:.2f}s")
            self.results[name] = {"params": size, "time": elapsed}
            model_sizes.append(size)
            training_times.append(elapsed)

        # Generate plot
        self.plot_scaling(model_sizes, training_times)

    def plot_scaling(self, sizes, times):
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times, marker='o', linestyle='-', color='purple')
        plt.title("Training Time vs. Model Size (Scaling Law Check)")
        plt.xlabel("Parameters")
        plt.ylabel("Time (s) for 10 steps")
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, "scaling_plot.png"))
        print(f"Scaling plot saved to {os.path.join(self.results_dir, 'scaling_plot.png')}")
        plt.close()

if __name__ == "__main__":
    configs = {
        "Tiny": ANAConfig(d_model=32, num_layers=1, state_dim=32),
        "Small": ANAConfig(d_model=64, num_layers=2, state_dim=64),
        "Medium": ANAConfig(d_model=128, num_layers=4, state_dim=128) # Added for more data points
    }
    exp = ScalingExperiment()
    exp.run_experiment(configs)
