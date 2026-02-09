#!/usr/bin/env python3
import argparse
import torch
from ana.config import ANAConfig, TrainingConfig, DataConfig
from ana.train import run_training
from ana.eval import run_all_evals
from ana.analysis import analyze_gating, analyze_attention_pattern
from ana.benchmarks import BenchmarkEvaluator, compare_models
from benchmark import run_benchmarks

def main():
    parser = argparse.ArgumentParser(description="ANA Research Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    train_parser = subparsers.add_parser("train", help="Run training")
    train_parser.add_argument("stage", choices=["2a", "2b", "3a"], help="Training stage")
    train_parser.add_argument("--model", choices=["ana", "baseline"], default="ana", help="Model type")
    train_parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    train_parser.add_argument("--state-dim", type=int, default=64, help="State dimension")
    train_parser.add_argument("--layers", type=int, default=2, help="Number of layers")
    train_parser.add_argument("--tracks", type=int, default=2, help="Number of tracks")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--parallel", action="store_true", help="Use parallel scan")
    train_parser.add_argument("--no-hololink", action="store_true", help="Disable HoloLink")
    train_parser.add_argument("--no-controller", action="store_true", help="Disable controller")
    train_parser.add_argument("--thinking-steps", type=int, default=0, help="Max thinking steps")
    
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    eval_parser.add_argument("--vocab-size", type=int, default=40, help="Vocabulary size")
    eval_parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    bench_parser.add_argument("--vocab-size", type=int, default=50, help="Vocabulary size")
    bench_parser.add_argument("--compare", action="store_true", help="Compare ANA vs Baseline")
    bench_parser.add_argument("--output", type=str, default="archive/benchmarks", help="Output directory")
    
    analyze_parser = subparsers.add_parser("analyze", help="Run analysis")
    analyze_parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    analyze_parser.add_argument("--vocab-size", type=int, default=40, help="Vocabulary size")
    
    study_parser = subparsers.add_parser("study", help="Run research studies")
    study_parser.add_argument("--type", choices=["scaling", "ablation", "full"], default="ablation", help="Study type")
    study_parser.add_argument("--scale", choices=["tiny", "small", "medium", "large"], default="small")
    study_parser.add_argument("--ablation", choices=["full", "no_hololink", "no_controller", "static_only", "with_thinking"], default="full")
    study_parser.add_argument("--output", type=str, default="archive/research")
    
    args = parser.parse_args()
    
    if args.command == "train":
        ana_config = ANAConfig(
            d_model=args.d_model,
            state_dim=args.state_dim,
            num_layers=args.layers,
            track_count=args.tracks,
            use_parallel_scan=args.parallel,
            use_hololink=not args.no_hololink,
            use_controller=not args.no_controller,
            max_thinking_steps=args.thinking_steps,
        )
        
        train_config = TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            stage=args.stage
        )
        
        data_config = DataConfig()
        
        run_training(ana_config, train_config, data_config, args.model)
    
    elif args.command == "eval":
        from ana.models import ANAModel
        
        ana_config = ANAConfig(vocab_size=args.vocab_size, d_model=args.d_model)
        model = ANAModel(ana_config)
        
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        results = run_all_evals(model, device, args.vocab_size)
        
        print("\n=== Evaluation Results ===")
        for task, score in results.items():
            print(f"{task}: {score:.4f}")
    
    elif args.command == "benchmark":
        from ana.models import ANAModel
        from ana.config import ANAConfig
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        config = ANAConfig(
            d_model=64, state_dim=64, vocab_size=args.vocab_size,
            num_layers=2, track_count=2, use_parallel_scan=True
        )
        
        if args.compare:
            compare_models(config, config, device, args.output)
        else:
            model = ANAModel(config).to(device)
            
            if args.checkpoint:
                print(f"Loading checkpoint: {args.checkpoint}")
                model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            
            evaluator = BenchmarkEvaluator(model, device, args.vocab_size)
            results = evaluator.run_all_benchmarks()
            
            import os
            os.makedirs(args.output, exist_ok=True)
            evaluator.save_results(os.path.join(args.output, "benchmark_results.json"))
    
    elif args.command == "analyze":
        from ana.models import ANAModel
        
        ana_config = ANAConfig(vocab_size=args.vocab_size)
        model = ANAModel(ana_config)
        
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
        
        model.eval()
        
        input_ids = torch.randint(0, args.vocab_size, (1, 50))
        
        import os
        os.makedirs("archive/analysis", exist_ok=True)
        analyze_gating(model, input_ids, "archive/analysis/gate_dynamics.png")
        analyze_attention_pattern(model, input_ids, "archive/analysis/attention_pattern.png")
    
    elif args.command == "study":
        from ana.experiments import run_scaling_study, run_ablation_study, run_full_study
        
        base_config = {
            'vocab_size': 50,
            'min_noise': 10,
            'max_noise': 50,
            'dataset_size': 2000,
        }
        
        if args.type == "scaling":
            run_scaling_study(base_config, args.scale, args.output)
        elif args.type == "ablation":
            run_ablation_study(base_config, args.ablation, args.output)
        else:
            run_full_study(args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
