import sys
from pathlib import Path
import torch
import time
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent / "ana" / "eqprop"))

from ana.bio_ana import create_bio_ana, get_bio_config


def detailed_profile():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_bio_config('nano')
    model = create_bio_ana('nano').to(device)
    model.eval()
    
    input_ids = torch.randint(0, 50, (2, 64), device=device)
    
    print("="*60)
    print("DETAILED COMPONENT TIMING")
    print("="*60)
    
    timings = {}
    
    with torch.no_grad():
        start = time.perf_counter()
        x = model.embedding(input_ids)
        timings['embedding'] = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        x = model._add_position_encoding(x)
        timings['pos_encoding'] = (time.perf_counter() - start) * 1000
        
        track_states = {'syntax': None, 'semantic': None, 'logic': None}
        
        track_timings = []
        for t in range(64):
            xt = x[:, t, :]
            start = time.perf_counter()
            track_out, track_states = model.tracks(
                xt,
                h_syntax=track_states['syntax'],
                h_semantic=track_states['semantic'],
                h_logic=track_states['logic'],
                steps=20,
            )
            track_timings.append((time.perf_counter() - start) * 1000)
        
        timings['tracks_total'] = sum(track_timings)
        timings['tracks_avg'] = sum(track_timings) / len(track_timings)
        timings['tracks_first_5'] = sum(track_timings[:5]) / 5
        timings['tracks_last_5'] = sum(track_timings[-5:]) / 5
        
        if model.hololink:
            start = time.perf_counter()
            _ = model.hololink(track_out, write_mode=False)
            timings['hololink'] = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        mixed = model.mixer(track_out)
        out = model.norm(xt + mixed)
        timings['mix_norm'] = (time.perf_counter() - start) * 1000
        
        start = time.perf_counter()
        output_seq = torch.stack([out] * 64, dim=1)
        logits = model.output_head(output_seq)
        timings['output_head'] = (time.perf_counter() - start) * 1000
    
    total = sum(v for v in timings.values() if isinstance(v, float))
    
    print(f"\nComponent breakdown:")
    for k, v in timings.items():
        if isinstance(v, float):
            pct = v / total * 100
            print(f"  {k:20s}: {v:8.3f}ms ({pct:5.1f}%)")
    
    print(f"\n{'='*60}")
    print("TRACK UPDATE PATTERN")
    print(f"{'='*60}")
    print(f"  First 5 tokens:  {timings['tracks_first_5']:.3f}ms avg")
    print(f"  Last 5 tokens:   {timings['tracks_last_5']:.3f}ms avg")
    print(f"  Ratio:           {timings['tracks_last_5']/timings['tracks_first_5']:.2f}x")
    
    print(f"\n{'='*60}")
    print("INSIGHTS")
    print(f"{'='*60}")
    
    if timings['tracks_total'] / total > 0.8:
        print("\n✓ TRACKS are the primary bottleneck (>80% of time)")
        print("  → Focus on optimizing track dynamics")
    
    if timings['tracks_last_5'] < timings['tracks_first_5'] * 0.9:
        print("\n✓ Later tokens converge faster")
        print("  → Adaptive relaxation is effective")
    
    if timings['hololink'] and timings['hololink'] < timings['tracks_total'] * 0.05:
        print("\n✓ HoloLink is negligible overhead")
        print("  → No optimization needed here")
    
    print(f"\n{'='*60}")
    print("NEXT-LEVEL OPTIMIZATIONS")
    print(f"{'='*60}")
    print("\n1. Fused track update kernel:")
    print("   - Combine all three track operations into single CUDA kernel")
    print("   - Reduces kernel launch overhead")
    print("   - Estimated speedup: 1.3-1.5x")
    
    print("\n2. Sparse relaxation iterations:")
    print("   - Only update tracks when input changes significantly")
    print("   - Skip iterations for padding/stop tokens")
    print("   - Estimated speedup: 1.2x")
    
    print("\n3. Cached track states:")
    print("   - Reuse previous states when input is similar")
    print("   - Use for repeated tokens/subwords")
    print("   - Estimated speedup: 1.1x")
    
    print("\n4. TorchScript compilation:")
    print("   - Convert tracks to TorchScript for JIT optimization")
    print("   - Bypass Python GIL overhead")
    print("   - Estimated speedup: 1.2x")
    
    return timings


def track_convergence_analysis():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_bio_ana('nano').to(device)
    model.eval()
    
    input_ids = torch.randint(0, 50, (2, 32), device=device)
    
    print(f"\n{'='*60}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    x = model.embedding(input_ids)
    x = model._add_position_encoding(x)
    
    track_states = {'syntax': None, 'semantic': None, 'logic': None}
    
    convergence_history = []
    
    with torch.no_grad():
        for t in range(32):
            xt = x[:, t, :]
            
            prev_state = None
            iters_needed = []
            
            for i in range(50):
                track_out, track_states = model.tracks(
                    xt,
                    h_syntax=track_states['syntax'],
                    h_semantic=track_states['semantic'],
                    h_logic=track_states['logic'],
                    steps=1,
                )
                
                if prev_state is not None:
                    max_diff = max(
                        torch.abs(track_states['syntax'] - prev_state['syntax']).max().item(),
                        torch.abs(track_states['semantic'] - prev_state['semantic']).max().item(),
                        torch.abs(track_states['logic'] - prev_state['logic']).max().item(),
                    )
                    if max_diff < 0.01:
                        iters_needed.append(i + 1)
                        break
                
                prev_state = {k: v.clone() for k, v in track_states.items()}
            
            if iters_needed:
                convergence_history.append(iters_needed[0])
    
    print(f"\nTokens analyzed: {len(convergence_history)}")
    print(f"Iterations to converge (threshold=0.01):")
    print(f"  Min:    {min(convergence_history)}")
    print(f"  Max:    {max(convergence_history)}")
    print(f"  Mean:   {sum(convergence_history)/len(convergence_history):.1f}")
    print(f"  Median: {sorted(convergence_history)[len(convergence_history)//2]}")
    
    print(f"\nDistribution:")
    for threshold in [5, 10, 15, 20, 30]:
        count = sum(1 for x in convergence_history if x <= threshold)
        print(f"  ≤ {threshold:2d} iters: {count:2d}/{len(convergence_history)} ({count/len(convergence_history)*100:.0f}%)")
    
    print(f"\n{'='*60}")
    print("OPTIMAL ITERATION SCHEDULE")
    print(f"{'='*60}")
    
    median_iters = sorted(convergence_history)[len(convergence_history)//2]
    print(f"\nRecommended fixed iterations: {median_iters}")
    print(f"  (vs current default of 20)")
    print(f"  → Speedup: {20/median_iters:.2f}x")
    
    print(f"\nAdaptive schedule recommendation:")
    print(f"  Tokens 0-25%:   {min(20, median_iters + 5)} iters")
    print(f"  Tokens 25-50%:  {median_iters} iters")
    print(f"  Tokens 50-75%:  {max(3, median_iters // 2)} iters")
    print(f"  Tokens 75-100%: {max(2, median_iters // 3)} iters")


if __name__ == "__main__":
    timings = detailed_profile()
    track_convergence_analysis()
