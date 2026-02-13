#!/usr/bin/env python
"""
Advanced ANA experiments entry point
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ana.experiments.advanced import run_advanced_comprehensive_experiment


def main():
    print("ANA: Adaptive Neural Automaton")
    print("Advanced Industrial/Academic Research Platform")
    print("="*60)
    
    print("\\nRunning advanced comprehensive experiments...")
    results = run_advanced_comprehensive_experiment()
    
    print("\\nAdvanced experiment completed successfully!")
    print("Results saved in the 'results/' directory")
    
    return results


if __name__ == "__main__":
    main()