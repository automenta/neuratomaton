import os
import shutil
import subprocess
import sys

def run_command(command, description):
    print(f"--- Running {description} ---")
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

def check_file(filepath):
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
    if not run_command("python3 run_research.py --phase 1 --experiment benchmarks", "Phase 1: Benchmarks"):
        all_passed = False
    if not check_file("results/phase1_benchmarks/benchmark_report.md"):
        all_passed = False

    # Phase 1: Scaling
    if not run_command("python3 run_research.py --phase 1 --experiment scaling", "Phase 1: Scaling"):
        all_passed = False
    if not check_file("results/phase1_scaling/scaling_plot.png"):
        all_passed = False

    # Phase 2: Long Context
    if not run_command("python3 run_research.py --phase 2 --experiment long_context", "Phase 2: Long Context"):
        all_passed = False

    # Phase 2: Inference (Non-interactive)
    if not run_command("python3 run_research.py --phase 2 --experiment inference", "Phase 2: Inference"):
        all_passed = False

    # Phase 3: Vision Training
    if not run_command("python3 run_research.py --phase 3 --experiment train_vision", "Phase 3: Vision Training"):
        all_passed = False
    if not check_file("results/phase3_vision/epoch_0_preds.png"):
        all_passed = False

    # Phase 3: Captioning
    if not run_command("python3 run_research.py --phase 3 --experiment captioning", "Phase 3: Captioning"):
        all_passed = False

    # Phase 4: RL Training
    if not run_command("python3 run_research.py --phase 4 --experiment train_rl", "Phase 4: RL Training"):
        all_passed = False
    if not check_file("results/phase4_rl/learning_curve.png"):
        all_passed = False

    # Phase 5: Series Training
    if not run_command("python3 run_research.py --phase 5 --experiment train_series", "Phase 5: Series Training"):
        all_passed = False

    # Phase 6: ONNX Export
    if not run_command("python3 run_research.py --phase 6 --experiment export_onnx", "Phase 6: ONNX Export"):
        all_passed = False
    # Note: run_research.py cleans up the ONNX file, so we can't check for it here unless we modify the script.
    # We rely on the command returning success (which implies export succeeded).

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
