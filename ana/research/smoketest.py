import os
import shutil
import subprocess
import sys
import glob

def run_command(command):
    print(f"--- Running: {command} ---")
    try:
        # Run command and capture output
        result = subprocess.run(
            command,
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print("SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def check_artifact(base_dir, filename_pattern):
    """
    Checks if a file matching the pattern exists in the most recent timestamped subdirectory
    of base_dir.
    """
    if not os.path.exists(base_dir):
        print(f"MISSING DIR: {base_dir}")
        return False

    # Get subdirectories (timestamps)
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        print(f"NO RUNS FOUND in {base_dir}")
        return False

    # Sort by name (timestamp is sortable)
    latest_dir = sorted(subdirs)[-1]

    # Check for file
    filepath = os.path.join(latest_dir, filename_pattern)
    exists = os.path.exists(filepath)

    print(f"Checking artifact {filepath}: {'FOUND' if exists else 'MISSING'}")
    return exists

def smoketest():
    print("Starting Comprehensive ANA Research Framework Smoketest...\n")

    # Ensure results directory is clean
    if os.path.exists("results"):
        shutil.rmtree("results")

    all_passed = True

    # Phase 1: Benchmarks
    if not run_command("python3 run_research.py --phase 1 --experiment benchmarks"):
        all_passed = False
    elif not check_artifact("results/phase1_benchmarks", "benchmark_report.md"):
        all_passed = False

    # Phase 1: Scaling
    if not run_command("python3 run_research.py --phase 1 --experiment scaling"):
        all_passed = False
    elif not check_artifact("results/phase1_scaling", "scaling_plot.png"):
        all_passed = False

    # Phase 1: Baseline Comparison
    if not run_command("python3 run_research.py --phase 1 --experiment baseline_comparison"):
        all_passed = False
    elif not check_artifact("results/phase1_baseline_comparison", "comparison_report.md"):
        all_passed = False

    # Phase 2: Long Context
    if not run_command("python3 run_research.py --phase 2 --experiment long_context"):
        all_passed = False
    elif not check_artifact("results/phase2_long_context", "results.json"):
        all_passed = False

    # Phase 2: Inference (Non-interactive)
    if not run_command("python3 run_research.py --phase 2 --experiment inference"):
        all_passed = False
    elif not check_artifact("results/phase2_inference", "inference_results.json"):
        all_passed = False

    # Phase 3: Vision Training
    if not run_command("python3 run_research.py --phase 3 --experiment train_vision"):
        all_passed = False
    elif not check_artifact("results/phase3_train_vision", "training_results.json"):
        all_passed = False

    # Phase 3: Captioning
    if not run_command("python3 run_research.py --phase 3 --experiment captioning"):
        all_passed = False
    elif not check_artifact("results/phase3_captioning", "captioning_results.json"):
        all_passed = False

    # Phase 4: RL Training
    if not run_command("python3 run_research.py --phase 4 --experiment train_rl"):
        all_passed = False
    elif not check_artifact("results/phase4_train_rl", "learning_curve.png"):
        all_passed = False

    # Phase 5: Series Training
    if not run_command("python3 run_research.py --phase 5 --experiment train_series"):
        all_passed = False
    elif not check_artifact("results/phase5_train_series", "series_results.json"):
        all_passed = False

    # Phase 6: ONNX Export
    if not run_command("python3 run_research.py --phase 6 --experiment export_onnx"):
        all_passed = False
    elif not check_artifact("results/phase6_export_onnx", "export_results.json"):
        all_passed = False

    print("\n--- Summary ---")
    if all_passed:
        print("ALL TESTS PASSED. Framework is fully functional.")
        # Cleanup
        if os.path.exists("results"):
            shutil.rmtree("results")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    smoketest()
