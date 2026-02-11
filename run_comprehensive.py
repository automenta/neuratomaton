#!/usr/bin/env python3
"""
ANA Project Comprehensive Execution Script

Runs all research tracks and generates comprehensive results with contingency handling.

Usage:
    python run_comprehensive.py [--tracks {all|core|extended}] [--skip {track1,track2,...}]

Tracks:
    core      - Original 3 tracks (curriculum, hybrid, cuda)
    extended  - Extended tracks (bioplausible, hololink, edge)
    all       - All 7 tracks

Research Philosophy: Every result generates value.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import time
from datetime import datetime


TRACKS = {
    'curriculum': {
        'name': 'Scale-Aware Curriculum',
        'script': 'experiments/scale_aware/curriculum_bench.py',
        'category': 'core',
        'risk': 'low',
        'contingencies': {
            'success': 'Workshop paper + open-source toolkit',
            'partial': 'Curriculum learning paper',
            'failure': 'Training difficulty analysis'
        }
    },
    'hybrid': {
        'name': 'Hybrid Architecture',
        'script': 'experiments/hybrid/mixed_tasks.py',
        'category': 'core',
        'risk': 'medium',
        'contingencies': {
            'success': 'ICLR paper + library release',
            'partial': 'Routing analysis paper',
            'failure': 'Architecture failure analysis'
        }
    },
    'cuda': {
        'name': 'CUDA Optimization',
        'script': 'experiments/cuda_benchmarks/speedup.py',
        'category': 'core',
        'risk': 'medium',
        'contingencies': {
            'success': 'SysML paper + library',
            'partial': 'Optimization paper',
            'failure': 'Bottleneck analysis'
        }
    },
    'bioplausible': {
        'name': 'Bio-Plausible Learning',
        'script': 'experiments/bioplausible/continual_learning.py',
        'category': 'extended',
        'risk': 'high',
        'contingencies': {
            'success': 'ICML paper (major contribution)',
            'partial': 'Domain-specific paper',
            'failure': 'Bio-plausibility limitations'
        }
    },
    'hololink': {
        'name': 'HoloLink Standalone',
        'script': 'experiments/hololink/standalone.py',
        'category': 'extended',
        'risk': 'low',
        'contingencies': {
            'success': 'Publication + library',
            'partial': 'Use-case paper',
            'failure': 'Attention comparison'
        }
    },
    'edge': {
        'name': 'Edge AI Deployment',
        'script': 'experiments/edge/deployment.py',
        'category': 'extended',
        'risk': 'medium',
        'contingencies': {
            'success': 'Industry paper + toolkit',
            'partial': 'Feasibility study',
            'failure': 'Deployment limitations'
        }
    },
    'synergy': {
        'name': 'Synergy Analysis',
        'script': 'experiments/exp_synergy_kv.py',
        'category': 'core',
        'risk': 'low',
        'contingencies': {
            'success': 'NeurIPS paper',
            'partial': 'Workshop paper',
            'failure': 'arXiv analysis'
        }
    }
}


def run_command(cmd, description, timeout=3600):
    """Run a command with timeout"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Timeout: {timeout}s")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"\n✓ {description} completed in {elapsed:.1f}s")
        return True, result.stdout, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ {description} timed out after {elapsed:.1f}s")
        return False, f"Timeout after {timeout}s", elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"✗ {description} failed after {elapsed:.1f}s")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False, str(e), elapsed
    except FileNotFoundError:
        print(f"✗ Script not found: {cmd[0]}")
        return False, "Script not found", 0


def analyze_output(output, track_name):
    """Analyze experiment output to determine outcome"""
    if not output:
        return 'unknown'
    
    output_lower = output.lower()
    
    # Success indicators
    success_indicators = [
        '✓ goal achieved',
        '✓ success',
        '✓ track successful',
        'advantage found',
        'outperforms',
        'successful'
    ]
    
    # Partial indicators
    partial_indicators = [
        '⚠ partial',
        '⚠ inconclusive',
        'moderate speedup',
        'comparable',
        'partial success'
    ]
    
    # Failure indicators
    failure_indicators = [
        '✗ no advantages',
        '✗ track failed',
        '✗ not feasible',
        'outperformed',
        'failed',
        'limitations'
    ]
    
    for indicator in success_indicators:
        if indicator in output_lower:
            return 'success'
    
    for indicator in partial_indicators:
        if indicator in output_lower:
            return 'partial'
    
    for indicator in failure_indicators:
        if indicator in output_lower:
            return 'failure'
    
    return 'unknown'


def generate_contingency_plan(results):
    """Generate next steps based on results"""
    plan = {
        'immediate': [],
        'short_term': [],
        'long_term': []
    }
    
    for track_name, track_result in results.items():
        outcome = track_result.get('outcome', 'unknown')
        track_info = TRACKS.get(track_name, {})
        contingencies = track_info.get('contingencies', {})
        
        if outcome == 'success':
            plan['immediate'].append(f"✓ {track_info.get('name', track_name)}: {contingencies.get('success', 'Proceed to publication')}")
        elif outcome == 'partial':
            plan['short_term'].append(f"⚠ {track_info.get('name', track_name)}: {contingencies.get('partial', 'Additional analysis needed')}")
        elif outcome == 'failure':
            plan['short_term'].append(f"✗ {track_info.get('name', track_name)}: {contingencies.get('failure', 'Publish limitations')}")
        else:
            plan['short_term'].append(f"? {track_info.get('name', track_name)}: Review output manually")
    
    # Generate overall recommendations
    successful_tracks = sum(1 for r in results.values() if r.get('outcome') == 'success')
    partial_tracks = sum(1 for r in results.values() if r.get('outcome') == 'partial')
    failed_tracks = sum(1 for r in results.values() if r.get('outcome') == 'failure')
    
    plan['summary'] = {
        'total_tracks': len(results),
        'successful': successful_tracks,
        'partial': partial_tracks,
        'failed': failed_tracks,
        'publication_paths': successful_tracks + partial_tracks
    }
    
    # Overall recommendation
    if successful_tracks >= 4:
        plan['overall'] = "✓ EXCELLENT: Multiple top-tier publications possible"
    elif successful_tracks >= 2:
        plan['overall'] = "✓ GOOD: Solid publication record achievable"
    elif successful_tracks + partial_tracks >= 3:
        plan['overall'] = "✓ ACCEPTABLE: Workshop + arXiv publications"
    else:
        plan['overall'] = "⚠ MINIMAL: Focus on negative results and analysis"
    
    return plan


def generate_comprehensive_report(results, plan):
    """Generate comprehensive final report"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = {
        'timestamp': timestamp,
        'execution_summary': {
            'total_tracks': len(TRACKS),
            'tracks_run': len(results),
            'successful': sum(1 for r in results.values() if r.get('outcome') == 'success'),
            'partial': sum(1 for r in results.values() if r.get('outcome') == 'partial'),
            'failed': sum(1 for r in results.values() if r.get('outcome') == 'failure'),
            'unknown': sum(1 for r in results.values() if r.get('outcome') == 'unknown')
        },
        'track_results': results,
        'contingency_plan': plan,
        'publication_roadmap': generate_publication_roadmap(results),
        'next_steps': generate_next_steps(results)
    }
    
    return report


def generate_publication_roadmap(results):
    """Generate publication roadmap based on results"""
    
    roadmap = {
        'tier_1': [],  # Top-tier (NeurIPS, ICLR, ICML)
        'tier_2': [],  # Specialized (SysML, workshops)
        'tier_3': []   # arXiv
    }
    
    for track_name, track_result in results.items():
        outcome = track_result.get('outcome', 'unknown')
        track_info = TRACKS.get(track_name, {})
        
        if outcome == 'success':
            if track_info.get('risk') == 'high':
                roadmap['tier_1'].append({
                    'track': track_info.get('name'),
                    'venue': 'NeurIPS/ICLR/ICML',
                    'paper_title': f"{track_info.get('name')}: Major Contribution"
                })
            else:
                roadmap['tier_1'].append({
                    'track': track_info.get('name'),
                    'venue': 'NeurIPS/ICLR',
                    'paper_title': f"{track_info.get('name')}: Empirical Study"
                })
        elif outcome == 'partial':
            roadmap['tier_2'].append({
                'track': track_info.get('name'),
                'venue': 'Workshop/Specialized',
                'paper_title': f"{track_info.get('name')}: Analysis"
            })
        else:
            roadmap['tier_3'].append({
                'track': track_info.get('name'),
                'venue': 'arXiv',
                'paper_title': f"{track_info.get('name')}: Study"
            })
    
    return roadmap


def generate_next_steps(results):
    """Generate actionable next steps"""
    
    steps = []
    
    # Successful tracks
    successful = [n for n, r in results.items() if r.get('outcome') == 'success']
    if successful:
        steps.append({
            'priority': 'high',
            'action': 'Draft papers for successful tracks',
            'tracks': successful
        })
    
    # Partial tracks
    partial = [n for n, r in results.items() if r.get('outcome') == 'partial']
    if partial:
        steps.append({
            'priority': 'medium',
            'action': 'Analyze partial results, determine if improvement possible',
            'tracks': partial
        })
    
    # Failed tracks
    failed = [n for n, r in results.items() if r.get('outcome') == 'failure']
    if failed:
        steps.append({
            'priority': 'low',
            'action': 'Document limitations and negative results',
            'tracks': failed
        })
    
    # Overall
    if len(results) > 0:
        steps.append({
            'priority': 'high',
            'action': 'Compile comprehensive paper(s) across all tracks',
            'tracks': 'all'
        })
        steps.append({
            'priority': 'medium',
            'action': 'Release open-source code and models',
            'tracks': 'all'
        })
        steps.append({
            'priority': 'low',
            'action': 'Submit to arXiv for visibility',
            'tracks': 'all'
        })
    
    return steps


def print_comprehensive_report(report):
    """Print formatted comprehensive report"""
    
    print("\n" + "="*80)
    print("ANA PROJECT - COMPREHENSIVE EXECUTION REPORT")
    print("="*80)
    print(f"Timestamp: {report['timestamp']}")
    
    # Execution summary
    summary = report['execution_summary']
    print(f"\n{'-'*80}")
    print("EXECUTION SUMMARY")
    print(f"{'-'*80}")
    print(f"Total Tracks: {summary['total_tracks']}")
    print(f"Tracks Run: {summary['tracks_run']}")
    print(f"  ✓ Successful: {summary['successful']}")
    print(f"  ⚠ Partial: {summary['partial']}")
    print(f"  ✗ Failed: {summary['failed']}")
    print(f"  ? Unknown: {summary['unknown']}")
    
    # Track results
    print(f"\n{'-'*80}")
    print("TRACK RESULTS")
    print(f"{'-'*80}")
    
    for track_name, track_result in report['track_results'].items():
        track_info = TRACKS.get(track_name, {})
        outcome = track_result.get('outcome', 'unknown')
        status_symbol = {'success': '✓', 'partial': '⚠', 'failure': '✗', 'unknown': '?'}.get(outcome, '?')
        
        print(f"\n{status_symbol} {track_info.get('name', track_name)}")
        print(f"  Risk Level: {track_info.get('risk', 'unknown')}")
        print(f"  Outcome: {outcome}")
        print(f"  Duration: {track_result.get('duration', 0):.1f}s")
    
    # Publication roadmap
    print(f"\n{'-'*80}")
    print("PUBLICATION ROADMAP")
    print(f"{'-'*80}")
    
    roadmap = report['publication_roadmap']
    
    if roadmap['tier_1']:
        print(f"\n🏆 Tier 1 (Top-Tier): {len(roadmap['tier_1'])} papers")
        for paper in roadmap['tier_1']:
            print(f"  • {paper['paper_title']} → {paper['venue']}")
    
    if roadmap['tier_2']:
        print(f"\n📚 Tier 2 (Workshop): {len(roadmap['tier_2'])} papers")
        for paper in roadmap['tier_2']:
            print(f"  • {paper['paper_title']} → {paper['venue']}")
    
    if roadmap['tier_3']:
        print(f"\n📄 Tier 3 (arXiv): {len(roadmap['tier_3'])} papers")
        for paper in roadmap['tier_3']:
            print(f"  • {paper['paper_title']} → {paper['venue']}")
    
    # Contingency plan
    plan = report['contingency_plan']
    print(f"\n{'-'*80}")
    print("CONTINGENCY PLAN")
    print(f"{'-'*80}")
    print(f"\nOverall Assessment: {plan['overall']}")
    
    if plan['immediate']:
        print(f"\nImmediate Actions:")
        for action in plan['immediate']:
            print(f"  • {action}")
    
    if plan['short_term']:
        print(f"\nShort-Term Actions:")
        for action in plan['short_term']:
            print(f"  • {action}")
    
    if plan['long_term']:
        print(f"\nLong-Term Actions:")
        for action in plan['long_term']:
            print(f"  • {action}")
    
    # Next steps
    print(f"\n{'-'*80}")
    print("NEXT STEPS")
    print(f"{'-'*80}")
    
    for step in report['next_steps']:
        priority_symbol = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(step['priority'], '⚪')
        print(f"\n{priority_symbol} [{step['priority'].upper()}] {step['action']}")
        if step.get('tracks') != 'all':
            print(f"  Tracks: {', '.join(step['tracks'])}")
    
    # Value statement
    print(f"\n{'='*80}")
    print("VALUE STATEMENT")
    print(f"{'='*80}")
    
    total_publications = len(roadmap['tier_1']) + len(roadmap['tier_2']) + len(roadmap['tier_3'])
    
    if total_publications >= 4:
        print(f"\n✓ EXCELLENT: {total_publications} publication paths identified")
        print(f"  Scientific Impact: High")
        print(f"  Practical Value: High")
        print(f"  Community Contribution: High")
    elif total_publications >= 2:
        print(f"\n✓ GOOD: {total_publications} publication paths identified")
        print(f"  Scientific Impact: Medium")
        print(f"  Practical Value: Medium")
        print(f"  Community Contribution: Medium")
    else:
        print(f"\n⚠ MINIMAL: {total_publications} publication paths identified")
        print(f"  Scientific Impact: Low-Medium")
        print(f"  Practical Value: Low")
        print(f"  Community Contribution: Medium (negative results)")
    
    print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive ANA project experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive.py                    # Run all tracks
  python run_comprehensive.py --tracks core     # Run core tracks only
  python run_comprehensive.py --skip cuda edge  # Skip specific tracks
        """
    )
    
    parser.add_argument(
        '--tracks',
        choices=['all', 'core', 'extended'],
        default='all',
        help='Which tracks to run'
    )
    
    parser.add_argument(
        '--skip',
        type=str,
        default='',
        help='Comma-separated list of tracks to skip'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Timeout per track in seconds (default: 3600)'
    )
    
    args = parser.parse_args()
    
    # Determine which tracks to run
    skip_tracks = set(args.skip.split(',')) if args.skip else set()
    
    if args.tracks == 'core':
        tracks_to_run = {k: v for k, v in TRACKS.items() 
                        if v['category'] == 'core' and k not in skip_tracks}
    elif args.tracks == 'extended':
        tracks_to_run = {k: v for k, v in TRACKS.items() 
                        if v['category'] == 'extended' and k not in skip_tracks}
    else:
        tracks_to_run = {k: v for k, v in TRACKS.items() if k not in skip_tracks}
    
    print("="*80)
    print("ANA PROJECT - COMPREHENSIVE EXECUTION")
    print("="*80)
    print(f"Tracks to run: {len(tracks_to_run)}")
    print(f"Timeout per track: {args.timeout}s")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    results = {}
    total_time = 0
    
    # Run each track
    for track_name, track_info in tracks_to_run.items():
        print(f"\n\nTrack {len(results)+1}/{len(tracks_to_run)}: {track_info['name']}")
        print(f"Risk Level: {track_info['risk']}")
        print(f"Script: {track_info['script']}")
        
        script_path = Path(track_info['script'])
        if not script_path.exists():
            print(f"⚠ Script not found: {track_info['script']}")
            results[track_name] = {
                'outcome': 'failure',
                'duration': 0,
                'output': 'Script not found'
            }
            continue
        
        success, output, duration = run_command(
            [sys.executable, str(script_path)],
            f"{track_info['name']} ({track_name})",
            timeout=args.timeout
        )
        
        outcome = analyze_output(output, track_name)
        
        results[track_name] = {
            'outcome': outcome,
            'duration': duration,
            'output': output,
            'success': success
        }
        
        total_time += duration
    
    # Generate reports
    plan = generate_contingency_plan(results)
    report = generate_comprehensive_report(results, plan)
    
    # Print report
    print_comprehensive_report(report)
    
    # Save report
    output_dir = Path('experiments')
    output_dir.mkdir(exist_ok=True)
    
    report_file = output_dir / f'comprehensive_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Comprehensive report saved: {report_file}")
    print(f"✓ Total execution time: {total_time/60:.1f} minutes")
    print(f"\nView results:")
    print(f"  Report: {report_file}")
    print(f"  Papers: papers/")
    print(f"  Code: ana/")


if __name__ == '__main__':
    main()
