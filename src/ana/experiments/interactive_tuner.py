import cmd
import os
import sys
import threading
import subprocess
import signal
import json
import logging
from typing import Optional

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ana.experiments.discovery import DiscoveryEngine
from ana.models.config import ANAConfig

class InteractiveTuner(cmd.Cmd):
    intro = 'Welcome to the ANA Interactive Tuner. Type help or ? to list commands.\n'
    prompt = '(ANA-Tuner) '

    def __init__(self, output_dir: str = "results/interactive"):
        super().__init__()

        if not OPTUNA_AVAILABLE:
            print("Error: Optuna is not installed. Please install it with `pip install optuna`.")
            sys.exit(1)

        self.engine = DiscoveryEngine(output_dir=output_dir)
        self.study_name = "ana_optimization"
        self.dashboard_process: Optional[subprocess.Popen] = None

        # Load or Create Study
        try:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.engine.storage_url,
                load_if_exists=True,
                direction='maximize',
                sampler=TPESampler(multivariate=True),
                pruner=HyperbandPruner()
            )
            print(f"Loaded Study '{self.study_name}' from {self.engine.storage_url}")

            # Safe access to best value
            try:
                best_val = self.study.best_value
            except ValueError:
                best_val = "N/A"
            print(f"Current Best Value: {best_val}")

        except Exception as e:
            print(f"Error loading study: {e}")
            self.study = None

    def do_status(self, arg):
        """Show current optimization status."""
        if not self.study:
            print("No active study.")
            return

        n_trials = len(self.study.trials)
        print(f"Trials: {n_trials}")

        try:
            print(f"Best Value: {self.study.best_value:.4f}")
            print("Best Params:")
            for k, v in self.study.best_params.items():
                print(f"  {k}: {v}")
        except ValueError:
            print("No successful trials yet (all pruned or failed).")

    def do_optimize(self, arg):
        """Run optimization for N trials. Usage: optimize [n_trials]"""
        if not self.study:
            print("No active study.")
            return

        try:
            n_trials = int(arg) if arg else 10
        except ValueError:
            print("Invalid number of trials.")
            return

        print(f"Starting optimization for {n_trials} trials...")
        # We reuse run_optimization but override n_trials logic by calling study.optimize directly?
        # Actually, run_optimization creates its own study object usually.
        # Let's modify DiscoveryEngine or just call it directly here since we have the study object.

        # We need the objective function from DiscoveryEngine.
        # It's defined inside run_optimization, which is not ideal for reuse.
        # But we can extract the inner logic or just instantiate a new engine run.

        # Better approach: We wrap the engine's run_optimization method but inject our n_trials.
        # But engine.run_optimization creates a new study object.
        # We should probably refactor DiscoveryEngine to accept an existing study or just copy the objective here.

        # For simplicity and robustness, I will call engine.run_optimization with the specific trial count.
        # Wait, run_optimization hardcodes n_trials based on quick flag.
        # I should probably just reimplement the objective call here to use the interactive study object.

        # Re-defining objective locally to use the interactive study session
        from ana.utils.datasets import MultiQueryAssociativeRecall
        from ana.models.core import ANAModel
        from torch.utils.data import DataLoader

        task = MultiQueryAssociativeRecall(num_samples=1000, vocab_size=40, num_pairs=8, num_queries=3)
        train_loader = DataLoader(task, batch_size=16, shuffle=True)
        val_loader = DataLoader(task, batch_size=16, shuffle=False)

        def objective(trial):
            d_model = trial.suggest_categorical('d_model', [32, 64, 128])
            track_count = trial.suggest_int('track_count', 1, 4)
            use_hololink = trial.suggest_categorical('use_hololink', [True, False])
            use_controller = trial.suggest_categorical('use_controller', [True, False])
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)

            config = ANAConfig(
                d_model=d_model, state_dim=d_model, num_layers=2,
                track_count=track_count, use_hololink=use_hololink,
                use_controller=use_controller, learning_rate=lr, dropout=dropout
            )

            model = ANAModel(config).to(self.engine.device)

            # Pruning
            steps = 200 # Fixed for interactive
            check_interval = 50

            # Helper for pruning check
            def pruning_callback(step, loss, model):
                if step > 0 and step % check_interval == 0:
                    _, acc = self.engine.evaluate_model(model, val_loader)
                    trial.report(acc, step)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            try:
                self.engine.train_model(model, train_loader, max_steps=steps, lr=lr, callback=pruning_callback)
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e:
                return 0.0

            _, acc = self.engine.evaluate_model(model, val_loader)
            return acc

        try:
            self.study.optimize(objective, n_trials=n_trials)
            print("Optimization batch complete.")
            self.do_status(None)
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user.")

    def do_dashboard(self, arg):
        """Launch Optuna Dashboard in a background process."""
        if self.dashboard_process:
            print("Dashboard is already running.")
            return

        try:
            # Check if optuna-dashboard is installed
            subprocess.run(["optuna-dashboard", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            cmd = ["optuna-dashboard", self.engine.storage_url]
            self.dashboard_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Dashboard launched at http://127.0.0.1:8080 (Storage: {self.engine.storage_url})")
            print("Type 'stop_dashboard' to close it.")
        except FileNotFoundError:
            print("Error: optuna-dashboard executable not found. Install with `pip install optuna-dashboard`.")
        except Exception as e:
            print(f"Error launching dashboard: {e}")

    def do_stop_dashboard(self, arg):
        """Stop the background dashboard process."""
        if self.dashboard_process:
            self.dashboard_process.terminate()
            self.dashboard_process = None
            print("Dashboard stopped.")
        else:
            print("Dashboard is not running.")

    def do_explain(self, arg):
        """Generate detailed explanation report."""
        if not self.study or len(self.study.trials) == 0:
            print("No trials to explain.")
            return

        print("Generating detailed explanation...")
        # Re-use DiscoveryEngine logic
        # Ideally we'd call a specific method on engine, but let's just trigger the report generation
        # We need to populate engine.tuning_results first

        try:
            best_p = self.study.best_params
            best_v = self.study.best_value
        except ValueError:
             print("No successful trials to explain.")
             return

        self.engine.tuning_results = {
            'best_params': best_p,
            'best_value': best_v,
            'trials': [{'params': t.params, 'value': t.value} for t in self.study.trials]
        }

        try:
            from optuna.importance import get_param_importances
            self.engine.param_importance = get_param_importances(self.study)
        except:
            pass

        self.engine.generate_scientific_report()
        print(f"Report generated at: {os.path.join(self.engine.output_dir, 'DISCOVERY_REPORT.md')}")

    def do_exit(self, arg):
        """Exit the tuner."""
        if self.dashboard_process:
            self.dashboard_process.terminate()
        print("Goodbye.")
        return True

    def do_quit(self, arg):
        return self.do_exit(arg)

if __name__ == '__main__':
    tuner = InteractiveTuner()
    try:
        tuner.cmdloop()
    except KeyboardInterrupt:
        tuner.do_exit(None)
