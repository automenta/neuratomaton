#!/usr/bin/env python3
"""
ANA v2: Comprehensive Test Suite

Tests each component individually before testing the full model.
"""

import sys
import torch

print("=" * 60)
print("ANA v2: Comprehensive Tests")
print("=" * 60)

failures = []

def test(name, fn):
    try:
        fn()
        print(f"   ✅ {name}")
        return True
    except Exception as e:
        print(f"   ❌ {name}: {e}")
        failures.append((name, str(e)))
        return False

print("\n1. IMPORTS")
def test_imports():
    from ana.v2.core import (
        ANAConfig, GumbelSoftmax, HolographicMemory, 
        ProgramStack, Interpreter, LinearRecurrentTrack,
        ANALayer, ANAModel
    )
test("Core imports", test_imports)

print("\n2. CONFIG")
def test_config():
    from ana.v2.core import ANAConfig
    c = ANAConfig(d_model=64, vocab_size=20)
    assert c.d_model == 64
    assert c.total_track_dim == 128  # 32+64+32
test("Config creation", test_config)

print("\n3. GUMBEL-SOFTMAX")
def test_gumbel_shape():
    from ana.v2.core import GumbelSoftmax
    logits = torch.randn(4, 4)
    samples = GumbelSoftmax.sample(logits, temperature=0.5, hard=True)
    assert samples.shape == logits.shape
test("Shape", test_gumbel_shape)

def test_gumbel_sum():
    from ana.v2.core import GumbelSoftmax
    logits = torch.randn(4, 4)
    samples = GumbelSoftmax.sample(logits, temperature=0.5, hard=True)
    assert torch.allclose(samples.sum(dim=-1), torch.ones(4), atol=1e-5)
test("Sum to 1", test_gumbel_sum)

def test_gumbel_zero_temp():
    from ana.v2.core import GumbelSoftmax
    logits = torch.randn(4, 4)
    samples = GumbelSoftmax.sample(logits, temperature=0.0)
    max_idx = logits.argmax(dim=-1)
    for i in range(4):
        assert samples[i, max_idx[i]] == 1.0
test("Zero temperature (argmax)", test_gumbel_zero_temp)

print("\n4. HOLOGRAPHIC MEMORY")
def test_holo_init():
    from ana.v2.core import HolographicMemory
    h = HolographicMemory(dim=64)
    assert h.dim == 64
    assert h.memory.shape[0] == 1000
test("Init", test_holo_init)

def test_holo_bind():
    from ana.v2.core import HolographicMemory
    h = HolographicMemory(dim=64)
    key = torch.randn(1, 64)
    value = torch.randn(1, 64)
    bound = h.bind(key, value)
    assert bound.shape == (1, 64)
test("Bind shape", test_holo_bind)

def test_holo_write_read():
    from ana.v2.core import HolographicMemory
    h = HolographicMemory(dim=64)
    key = torch.randn(1, 64)
    value = torch.randn(1, 64)
    h.write(key, value)
    assert h.write_idx == 1
    retrieved = h.read(key)
    assert retrieved.shape == (1, 64)
test("Write and read", test_holo_write_read)

def test_holo_reset():
    from ana.v2.core import HolographicMemory
    h = HolographicMemory(dim=64)
    key = torch.randn(1, 64)
    value = torch.randn(1, 64)
    h.write(key, value)
    h.reset()
    assert h.write_idx == 0
test("Reset", test_holo_reset)

print("\n5. PROGRAM STACK")
def test_stack_init():
    from ana.v2.core import ProgramStack
    s = ProgramStack(dim=32, max_depth=5)
    assert s.depth() == 0
test("Init", test_stack_init)

def test_stack_push():
    from ana.v2.core import ProgramStack
    s = ProgramStack(dim=32, max_depth=5)
    state = torch.randn(1, 32)
    assert s.push(state) == True
    assert s.depth() == 1
test("Push", test_stack_push)

def test_stack_pop():
    from ana.v2.core import ProgramStack
    s = ProgramStack(dim=32, max_depth=5)
    state = torch.randn(1, 32)
    s.push(state)
    frame = s.pop()
    assert frame is not None
    assert s.depth() == 0
test("Pop", test_stack_pop)

def test_stack_max_depth():
    from ana.v2.core import ProgramStack
    s = ProgramStack(dim=32, max_depth=3)
    for _ in range(5):
        s.push(torch.randn(1, 32))
    assert s.depth() == 3
test("Max depth", test_stack_max_depth)

print("\n6. INTERPRETER")
def test_interpreter_init():
    from ana.v2.core import ANAConfig, Interpreter
    c = ANAConfig()
    i = Interpreter(c)
    assert i.temperature == 1.0
test("Init", test_interpreter_init)

def test_interpreter_execute():
    from ana.v2.core import ANAConfig, Interpreter, ProgramStack, HolographicMemory
    c = ANAConfig(d_model=64, stack_dim=32)
    i = Interpreter(c)
    stack = ProgramStack(32, 5)
    holo = HolographicMemory(64)
    
    x = torch.randn(2, 64)
    opcode_logits = torch.randn(2, 4)
    h_prev = torch.zeros(2, 32)
    
    alpha, beta, h_next, info = i.execute(x, opcode_logits, stack, holo, h_prev)
    
    assert alpha.shape == (2, 3)
    assert beta.shape == (2, 3)
    assert h_next.shape == (2, 32)
    assert 'opcode' in info
test("Execute", test_interpreter_execute)

print("\n7. LINEAR RECURRENT TRACK")
def test_track_init():
    from ana.v2.core import LinearRecurrentTrack
    t = LinearRecurrentTrack(input_dim=64, state_dim=32)
    assert t.state_dim == 32
test("Init", test_track_init)

def test_track_step():
    from ana.v2.core import LinearRecurrentTrack
    t = LinearRecurrentTrack(input_dim=64, state_dim=32)
    x = torch.randn(2, 64)
    y, h = t._step(x, None, None, None)
    assert y.shape == (2, 64)
    assert h.shape == (2, 32)
test("Step forward", test_track_step)

def test_track_sequence():
    from ana.v2.core import LinearRecurrentTrack
    t = LinearRecurrentTrack(input_dim=64, state_dim=32)
    x = torch.randn(2, 5, 64)
    y, h = t._sequence(x, None, None)
    assert y.shape == (2, 5, 64)
    assert h.shape == (2, 32)
test("Sequence forward", test_track_sequence)

def test_track_modulation():
    from ana.v2.core import LinearRecurrentTrack
    t = LinearRecurrentTrack(input_dim=64, state_dim=32)
    x = torch.randn(2, 64)
    alpha_mod = torch.ones(2, 1)
    beta_mod = torch.ones(2, 1)
    y, h = t._step(x, None, alpha_mod, beta_mod)
    assert y.shape == (2, 64)
test("With modulation", test_track_modulation)

print("\n8. ANA LAYER")
def test_layer_init():
    from ana.v2.core import ANAConfig, ANALayer
    c = ANAConfig(d_model=64, track_dims=(16, 32, 16))
    layer = ANALayer(c)
    assert len(layer.tracks) == 3
test("Init", test_layer_init)

def test_layer_forward():
    from ana.v2.core import ANAConfig, ANALayer
    c = ANAConfig(d_model=64, track_dims=(16, 32, 16), vocab_size=20)
    layer = ANALayer(c)
    x = torch.randn(2, 5, 64)
    out, states, info = layer(x)
    assert out.shape == (2, 5, 64)
    assert len(states) == 3
test("Forward", test_layer_forward)

print("\n9. FULL MODEL")
def test_model_init():
    from ana.v2.core import ANAConfig, ANAModel
    c = ANAConfig(d_model=64, vocab_size=20)
    m = ANAModel(c)
    params = sum(p.numel() for p in m.parameters())
    assert params > 0
    print(f"(params: {params:,})", end=" ")
test("Init", test_model_init)

def test_model_forward():
    from ana.v2.core import ANAConfig, ANAModel
    c = ANAConfig(d_model=64, vocab_size=20, track_dims=(16, 32, 16))
    m = ANAModel(c)
    input_ids = torch.randint(0, 20, (2, 5))
    logits = m(input_ids)
    assert logits.shape == (2, 5, 20)
test("Forward shape", test_model_forward)

def test_model_backward():
    from ana.v2.core import ANAConfig, ANAModel
    c = ANAConfig(d_model=64, vocab_size=20, track_dims=(16, 32, 16))
    m = ANAModel(c)
    input_ids = torch.randint(0, 20, (2, 5))
    targets = torch.randint(0, 20, (2, 5))
    
    logits = m(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 20), targets.view(-1)
    )
    loss.backward()
    
    grads = sum(1 for p in m.parameters() if p.grad is not None)
    assert grads > 0
test("Backward pass", test_model_backward)

print("\n10. TASK GENERATION")
def test_tasks():
    from ana.v2.tasks import (
        generate_copy_task, generate_reverse_task,
        generate_associative_recall_task
    )
    
    task = generate_copy_task(num_train=10, num_test=5)
    assert task.train_seqs.shape[0] == 10
    assert task.test_seqs.shape[0] == 5
    
    task = generate_reverse_task(num_train=10, num_test=5)
    assert task.train_seqs.shape[0] == 10
    
    task = generate_associative_recall_task(num_train=10, num_test=5)
    assert task.train_seqs.shape[0] == 10
test("Task generation", test_tasks)

print("\n11. TRAINING SETUP")
def test_trainer():
    from ana.v2.core import ANAConfig
    from ana.v2.train import Trainer, SimpleDataset
    from ana.v2.tasks import generate_copy_task
    from torch.utils.data import DataLoader
    
    task = generate_copy_task(num_train=10, num_test=5)
    config = ANAConfig(d_model=32, vocab_size=task.vocab_size, 
                       track_dims=(8, 16, 8), num_layers=1)
    
    dataset = SimpleDataset(task.train_seqs, task.train_targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    trainer = Trainer(config, lr=1e-3)
    
    # Single batch
    for x, targets in loader:
        trainer.model.to(trainer.device)
        x, targets = x.to(trainer.device), targets.to(trainer.device)
        logits = trainer.model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            targets.view(-1),
            ignore_index=0
        )
        assert loss.item() > 0
        break
test("Trainer setup", test_trainer)

print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)} tests")
    for name, err in failures:
        print(f"  - {name}: {err}")
    sys.exit(1)
else:
    print("ALL TESTS PASSED! ANA v2 is ready. 🖕🚀")
print("=" * 60)
