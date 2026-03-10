#!/usr/bin/env python
"""
Main entry point for ANA experiments
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ana.experiments.main import run_comprehensive_comparison


def main():
    print("ANA: Adaptive Neural Automaton")
    print("Industrial/Academic Research Platform")
    print("="*50)
    
    print("\\nRunning comprehensive experiments...")
    results = run_comprehensive_comparison()
    
    print("\\nExperiment completed successfully!")
    print("Results saved in the 'results/' directory")
    
    return results


if __name__ == "__main__":
    main()