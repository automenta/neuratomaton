"""
ANA Research Experiments

Core experiments for validating multi-track SSM generalization.
"""
import torch
import torch.nn.functional as F
from ana import ANAConfig, ANAModel
from ana.models import BaselineSSM


def train_curriculum(
    task='reverse',
    train_lengths=(2, 3, 4, 5, 6),
    test_lengths=(7, 8, 10, 12),
    steps=300,
    lr=1e-2,
    d_model=32,
    state_dim=32,
    track_count=2,
    use_hololink=True,
    use_controller=True,
    vocab_size=10,
    batch_size=16,
    verbose=True
):
    """
    Train with curriculum learning on multiple lengths.
    Test generalization to unseen lengths.
    
    Returns: dict with training and generalization results
    """
    config = ANAConfig(
        d_model=d_model,
        vocab_size=vocab_size,
        state_dim=state_dim,
        track_count=track_count,
        use_hololink=use_hololink,
        use_controller=use_controller,
    )
    model = ANAModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if verbose:
        params = sum(p.numel() for p in model.parameters())
        print(f"Training ANA ({params:,} params)")
        print(f"  Train lengths: {train_lengths}")
        print(f"  Test lengths: {test_lengths}")
    
    # Training loop with curriculum
    for step in range(steps):
        L = train_lengths[step % len(train_lengths)]
        
        if task == 'reverse':
            train = torch.randint(1, vocab_size - 1, (batch_size, L))
            targ = train.flip(dims=[1])
        elif task == 'copy':
            train = torch.randint(1, vocab_size - 1, (batch_size, L))
            targ = train.clone()
        else:
            raise ValueError(f"Unknown task: {task}")
        
        optimizer.zero_grad()
        logits, _ = model(train)
        loss = F.cross_entropy(logits.view(-1, vocab_size), targ.view(-1), ignore_index=0)
        loss.backward()
        optimizer.step()
        
        if verbose and (step + 1) % 100 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == targ).float().mean()
            print(f"  Step {step+1}: loss={loss.item():.4f}, acc={100*acc:.0f}%")
    
    # Generalization test
    model.eval()
    results = {'train_lengths': train_lengths, 'test_lengths': test_lengths, 'generalization': {}}
    
    with torch.no_grad():
        for L in test_lengths:
            accs = []
            for _ in range(20):
                test = torch.randint(1, vocab_size - 1, (batch_size, L))
                if task == 'reverse':
                    test_targ = test.flip(dims=[1])
                else:
                    test_targ = test.clone()
                
                logits, _ = model(test)
                acc = (logits.argmax(-1) == test_targ).float().mean()
                accs.append(acc.item())
            
            k = L / max(train_lengths)
            mean_acc = sum(accs) / len(accs)
            results['generalization'][L] = {'k': k, 'accuracy': mean_acc}
            
            if verbose:
                print(f"  Length {L} (k={k:.1f}): {100*mean_acc:.1f}%")
    
    return model, results


def ablation_hololink(train_lengths=(2, 3, 4, 5, 6), test_lengths=(7, 8, 10), steps=200):
    """Compare ANA with and without HoloLink."""
    print("\n" + "="*60)
    print("ABLATION: HoloLink")
    print("="*60)
    
    results = {}
    
    for use_holo in [True, False]:
        name = "HoloLink ON" if use_holo else "HoloLink OFF"
        print(f"\n--- {name} ---")
        
        _, res = train_curriculum(
            train_lengths=train_lengths,
            test_lengths=test_lengths,
            steps=steps,
            use_hololink=use_holo,
            verbose=False
        )
        results[name] = res
        
        for L, data in res['generalization'].items():
            print(f"  Length {L} (k={data['k']:.1f}): {100*data['accuracy']:.1f}%")
    
    return results


def ablation_tracks(train_lengths=(2, 3, 4, 5, 6), test_lengths=(7, 8, 10), steps=200):
    """Compare different track counts."""
    print("\n" + "="*60)
    print("ABLATION: Track Count")
    print("="*60)
    
    results = {}
    
    for num_tracks in [1, 2, 3]:
        print(f"\n--- {num_tracks} Track(s) ---")
        
        _, res = train_curriculum(
            train_lengths=train_lengths,
            test_lengths=test_lengths,
            steps=steps,
            track_count=num_tracks,
            verbose=False
        )
        results[num_tracks] = res
        
        for L, data in res['generalization'].items():
            print(f"  Length {L} (k={data['k']:.1f}): {100*data['accuracy']:.1f}%")
    
    return results


def compare_baseline(train_lengths=(2, 3, 4, 5, 6), test_lengths=(7, 8, 10), steps=300):
    """Compare ANA vs single-track baseline."""
    print("\n" + "="*60)
    print("COMPARISON: ANA vs BaselineSSM")
    print("="*60)
    
    config = ANAConfig(d_model=32, vocab_size=10, state_dim=32, track_count=2)
    
    results = {'ana': {}, 'baseline': {}}
    
    for name, ModelClass in [('ANA', ANAModel), ('Baseline', BaselineSSM)]:
        print(f"\n--- {name} ---")
        model = ModelClass(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        for step in range(steps):
            L = train_lengths[step % len(train_lengths)]
            train = torch.randint(1, 9, (16, L))
            targ = train.flip(dims=[1])
            
            optimizer.zero_grad()
            logits, _ = model(train)
            F.cross_entropy(logits.view(-1, 10), targ.view(-1)).backward()
            optimizer.step()
        
        model.eval()
        with torch.no_grad():
            for L in test_lengths:
                accs = []
                for _ in range(20):
                    test = torch.randint(1, 9, (16, L))
                    test_targ = test.flip(dims=[1])
                    logits, _ = model(test)
                    acc = (logits.argmax(-1) == test_targ).float().mean()
                    accs.append(acc.item())
                mean_acc = sum(accs) / len(accs)
                results[name.lower()][L] = mean_acc
                print(f"  Length {L}: {100*mean_acc:.1f}%")
    
    return results


def run_all_experiments():
    """Run the full experiment suite."""
    print("="*60)
    print("ANA RESEARCH EXPERIMENTS")
    print("="*60)
    
    # E1: Curriculum learning baseline
    print("\n[E1] Curriculum Learning")
    _, e1_results = train_curriculum(steps=300)
    
    # E2: HoloLink ablation
    e2_results = ablation_hololink(steps=200)
    
    # E3: Track count ablation
    e3_results = ablation_tracks(steps=200)
    
    # E4: Baseline comparison
    e4_results = compare_baseline(steps=300)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("Curriculum Learning (E1):")
    for L, data in e1_results['generalization'].items():
        print(f"  k={data['k']:.1f}: {100*data['accuracy']:.1f}%")
    
    return {
        'curriculum': e1_results,
        'hololink_ablation': e2_results,
        'track_ablation': e3_results,
        'baseline_comparison': e4_results
    }


if __name__ == "__main__":
    run_all_experiments()
