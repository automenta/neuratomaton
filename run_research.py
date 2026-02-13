import argparse
import sys
import os

# Ensure ana package is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="ANA Research Agenda Execution Framework")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6], help="Research Phase (1-6)")
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
        from ana.research.phase3_vision.models import ANAVisionModel, ANAVisionCaptioner
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

        elif args.experiment == "captioning":
            print("Running Captioning Model Demo...")
            config = ANAConfig(d_model=64, patch_size=16, vocab_size=100)
            model = ANAVisionCaptioner(config).to(args.device)
            images = torch.randn(1, 3, 224, 224).to(args.device)
            text_ids = torch.randint(0, 100, (1, 10)).to(args.device)
            out = model(images, text_ids)
            print(f"Captioning Output: {out.shape}")

        else:
            print("Available Phase 3 experiments: train_vision, captioning")

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

    elif args.phase == 5:
        from ana.research.phase5_specialized.models import ANASeriesModel
        from ana.research.phase5_specialized.train import SeriesTrainer
        from ana.config import ANAConfig
        import torch

        if args.experiment == "train_series":
            print("Running Series Training...")
            config = ANAConfig(series_dim=1, d_model=32)
            model = ANASeriesModel(config)
            trainer = SeriesTrainer(model, device=args.device)

            # Dummy data
            data = torch.randn(1, 100, 1)
            loss = trainer.train_epoch([data])
            print(f"Loss: {loss:.4f}")

        else:
             print("Available Phase 5 experiments: train_series")

    elif args.phase == 6:
        # Using Phase 6 for "Production & Ecosystem" (Deployment)
        from ana.research.deployment.export import export_to_onnx
        from ana.models import ANAModel
        from ana.config import ANAConfig
        import torch

        if args.experiment == "export_onnx":
            print("Running ONNX Export...")
            config = ANAConfig(vocab_size=100, d_model=32, num_layers=1)
            model = ANAModel(config)
            dummy = torch.randint(0, 100, (1, 32))
            export_to_onnx(model, dummy, filepath="ana_research_model.onnx")
            if os.path.exists("ana_research_model.onnx"):
                os.remove("ana_research_model.onnx")
                # Also check for potential .data files
                if os.path.exists("ana_research_model.onnx.data"):
                    os.remove("ana_research_model.onnx.data")
                print("Cleanup successful.")

        else:
            print("Available Phase 6 experiments: export_onnx")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
