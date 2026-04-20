import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from ana.models import ANAModel
from ana.config import ANAConfig
from ana.research.core import ExperimentBase, ExperimentRegistry

@ExperimentRegistry.register(phase=1, name="scaling")
class ScalingExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "scaling"

    @property
    def phase(self) -> int:
        return 1

    def get_model_size(self, model):
        return sum(p.numel() for p in model.parameters())

    def execute(self):
        # Define default configs
        configs = {
            "Tiny": ANAConfig(d_model=32, num_layers=1, state_dim=32),
            "Small": ANAConfig(d_model=64, num_layers=2, state_dim=64),
            "Medium": ANAConfig(d_model=128, num_layers=4, state_dim=128)
        }

        model_sizes = []
        training_times = []
        results = {}

        for name, config in configs.items():
            self.results.log(f"\n--- Training Model: {name} ---")
            model = ANAModel(config).to(self.device)
            size = self.get_model_size(model)
            self.results.log(f"Parameters: {size:,}")

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
            self.results.log(f"Training completed in {elapsed:.2f}s")

            results[name] = {"params": size, "time": elapsed}
            model_sizes.append(size)
            training_times.append(elapsed)

        self.results.save_json("scaling_results.json", results)
        self.plot_scaling(model_sizes, training_times)

    def plot_scaling(self, sizes, times):
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, times, marker='o', linestyle='-', color='purple')
        plt.title("Training Time vs. Model Size (Scaling Law Check)")
        plt.xlabel("Parameters")
        plt.ylabel("Time (s) for 10 steps")
        plt.grid(True)
        self.results.save_plot("scaling_plot.png")

if __name__ == "__main__":
    config = ANAConfig()
    exp = ScalingExperiment(config)
    exp.run()
