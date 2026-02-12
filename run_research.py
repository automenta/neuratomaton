import argparse
import sys
import os

# Ensure ana package is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="ANA Research Agenda Execution Framework")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], help="Research Phase (1-4)")
    parser.add_argument("--experiment", type=str, help="Experiment name")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda)")

    args = parser.parse_args()

    if args.phase == 1:
        from ana.research.phase1_validation.benchmarks import BenchmarkRunner
        from ana.research.phase1_validation.scaling import ScalingExperiment
        from ana.config import ANAConfig
        from ana.models import ANAModel

        if args.experiment == "benchmarks":
            print("Running Benchmarks...")
            config = ANAConfig()
            model = ANAModel(config)
            runner = BenchmarkRunner(model, device=args.device)
            runner.run_all()

        elif args.experiment == "scaling":
            print("Running Scaling Experiment...")
            configs = {
                "Tiny": ANAConfig(d_model=32, num_layers=1),
                "Small": ANAConfig(d_model=64, num_layers=2)
            }
            exp = ScalingExperiment(device=args.device)
            exp.run_experiment(configs)

        else:
            print("Available Phase 1 experiments: benchmarks, scaling")

    elif args.phase == 2:
        from ana.research.phase2_text.long_context import needle_in_haystack
        from ana.research.phase2_text.inference import InferenceEngine
        from ana.config import ANAConfig
        from ana.models import ANAModel

        if args.experiment == "long_context":
            print("Running Long Context Experiment...")
            config = ANAConfig(max_position=4096, vocab_size=100)
            model = ANAModel(config).to(args.device)
            needle_in_haystack(model, context_length=128)

        elif args.experiment == "inference":
            print("Running Inference Demo...")
            config = ANAConfig(vocab_size=100)
            model = ANAModel(config).to(args.device)
            engine = InferenceEngine(model, device=args.device)
            engine.run_demo()

        else:
            print("Available Phase 2 experiments: long_context, inference")

    elif args.phase == 3:
        from ana.research.phase3_vision.models import ANAVisionModel
        from ana.research.phase3_vision.train import VisionTrainer
        from ana.config import ANAConfig
        import torch

        if args.experiment == "train_vision":
            print("Running Vision Training...")
            config = ANAConfig(d_model=64, patch_size=16)
            model = ANAVisionModel(config, num_classes=10)
            trainer = VisionTrainer(model, device=args.device)

            # Dummy data
            images = torch.randn(4, 3, 224, 224)
            labels = torch.randint(0, 10, (4,))
            loss = trainer.train_epoch([(images, labels)])
            print(f"Loss: {loss:.4f}")

        else:
            print("Available Phase 3 experiments: train_vision")

    elif args.phase == 4:
        from ana.research.phase4_rl.agent import ANARLAgent
        from ana.research.phase4_rl.train import RLTrainer
        from ana.config import ANAConfig
        import torch

        if args.experiment == "train_rl":
            print("Running RL Training...")
            config = ANAConfig(observation_space=10, action_space=4, d_model=32)
            agent = ANARLAgent(config)
            trainer = RLTrainer(agent, device=args.device)

            obs = torch.randn(1, 10)
            loss = trainer.train_step(obs, 1, 1.0, obs, False)
            print(f"Loss: {loss:.4f}")

        else:
            print("Available Phase 4 experiments: train_rl")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
