
import argparse
from ana.config import ANAConfig, TrainingConfig, DataConfig
from ana.train import run_training
import sys

def main():
    parser = argparse.ArgumentParser(description="ANA Research Framework")

    # ANA Config
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--state_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_tracks", type=int, default=2)
    parser.add_argument("--use_hololink", type=str, default="True", help="True/False")
    parser.add_argument("--use_controller", type=str, default="True", help="True/False")
    parser.add_argument("--use_parallel_scan", type=str, default="False", help="True/False")
    parser.add_argument("--orthogonal_init", type=str, default="False", help="True/False")
    parser.add_argument("--controller_hidden_dim", type=int, default=64)
    parser.add_argument("--controller_layers", type=int, default=2)
    parser.add_argument("--hololink_decay", type=float, default=1.0)

    # Training Config
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if "cuda" in "cuda" else "cpu")
    parser.add_argument("--output_dir", type=str, default="archive/results")
    parser.add_argument("--stage", type=str, default="2a", choices=["2a", "2b", "3a"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_type", type=str, default="ana", choices=["ana", "baseline"], help="Model type: ana or baseline")
    parser.add_argument("--curriculum_epochs", type=int, default=5)
    parser.add_argument("--start_force_prob", type=float, default=1.0)

    # Data Config
    parser.add_argument("--dataset_size", type=int, default=2000)
    parser.add_argument("--min_noise", type=int, default=10)
    parser.add_argument("--max_noise", type=int, default=50)

    args = parser.parse_args()

    # Convert bool strings
    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    ana_config = ANAConfig(
        d_model=args.d_model,
        state_dim=args.state_dim,
        num_layers=args.num_layers,
        num_tracks=args.num_tracks,
        use_hololink=str2bool(args.use_hololink),
        use_controller=str2bool(args.use_controller),
        use_parallel_scan=str2bool(args.use_parallel_scan),
        orthogonal_init=str2bool(args.orthogonal_init),
        controller_hidden_dim=args.controller_hidden_dim,
        controller_layers=args.controller_layers,
        hololink_decay=args.hololink_decay
    )

    train_config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device,
        output_dir=args.output_dir,
        stage=args.stage,
        seed=args.seed,
        curriculum_epochs=args.curriculum_epochs,
        start_force_prob=args.start_force_prob
    )

    data_config = DataConfig(
        dataset_size=args.dataset_size,
        min_noise=args.min_noise,
        max_noise=args.max_noise
    )

    print("Running Experiment with Config:")
    print(ana_config)
    print(train_config)
    print(f"Model Type: {args.model_type}")
    print(data_config)

    run_training(ana_config, train_config, data_config, model_type=args.model_type)

if __name__ == "__main__":
    main()
