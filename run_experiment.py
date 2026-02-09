
import argparse
from ana.config import ANAConfig
from ana.train import train_stage_2a, train_stage_2b, train_stage_3a
import torch

def main():
    parser = argparse.ArgumentParser(description="ANA Research Framework")
    parser.add_argument("--stage", type=str, default="2a", choices=["2a", "2b", "3a"], help="Training stage")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")

    # Ablations
    parser.add_argument("--no-hololink", action="store_true", help="Disable HoloLink")
    parser.add_argument("--no-controller", action="store_true", help="Disable Controller")
    parser.add_argument("--parallel", action="store_true", help="Use Parallel Scan (Fast Log-N)")

    # Architecture
    parser.add_argument("--track-count", type=int, default=2, help="Number of parallel tracks")

    args = parser.parse_args()

    config = ANAConfig()

    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    if args.no_hololink:
        config.use_hololink = False
    if args.no_controller:
        config.use_controller = False

    if args.parallel:
        config.use_parallel_scan = True

    if args.track_count:
        config.track_count = args.track_count

    # Set device
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Running Stage {args.stage} with Config: {config}")

    if args.stage == "2a":
        train_stage_2a(config)
    elif args.stage == "2b":
        train_stage_2b(config)
    elif args.stage == "3a":
        train_stage_3a(config)

if __name__ == "__main__":
    main()
