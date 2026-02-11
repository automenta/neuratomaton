"""
Edge AI Deployment Experiments

Tests whether ANA can enable associative memory on resource-constrained edge devices.

Key Experiments:
1. Quantization (INT8/INT4) for memory reduction
2. Latency measurement on simulated edge hardware
3. Power consumption estimation
4. Deployment feasibility analysis
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from pathlib import Path
import time
import copy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ana.model_v3 import ANAv2Model
from ana.config_v2 import ANAv2Config


class EdgeAssociativeModel(nn.Module):
    """Lightweight associative model for edge deployment"""
    def __init__(self, vocab_size=30, d_model=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Simplified tracks
        self.track1 = nn.Linear(d_model, d_model)
        self.track2 = nn.Linear(d_model, d_model)
        
        # Memory
        self.memory = nn.Parameter(torch.zeros(100, d_model))
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        h = torch.zeros(x.size(0), x.size(2), device=x.device)
        
        outputs = []
        for t in range(x.size(1)):
            # Track update
            h = torch.sigmoid(self.track1(h)) * h + torch.sigmoid(self.track2(x[:, t])) * x[:, t]
            
            # Memory access
            sim = torch.matmul(h, self.memory.T)
            mem_out = torch.matmul(F.softmax(sim, dim=-1), self.memory)
            
            # Combine
            h = h + 0.1 * mem_out
            outputs.append(h)
        
        out = torch.stack(outputs, dim=1)
        return self.output(out)


class NeedleHaystackDataset(Dataset):
    def __init__(self, num_samples=500, vocab_size=30, seq_len=64):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            keys = torch.randint(1, 10, (2,))
            values = torch.randint(10, 20, (2,))
            
            kv_sequence = [keys[0].item(), values[0].item(), keys[1].item(), values[1].item()]
            
            noise = torch.randint(1, 30, (10,)).tolist()
            
            query = keys[np.random.randint(2)].item()
            target = values[keys.tolist().index(query)].item()
            
            sequence = kv_sequence + noise + [query, target]
            sequence = sequence[:self.seq_len]
            
            input_ids = torch.tensor(sequence[:self.seq_len-1], dtype=torch.long)
            target_id = torch.tensor(sequence[-1], dtype=torch.long)
            
            data.append((input_ids, target_id))
        
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


class EdgeDeploymentBenchmark:
    def __init__(self):
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def quantize_model(self, model, quant_type='int8'):
        """Quantize model for edge deployment"""
        model = model.to('cpu')
        model.eval()
        
        if quant_type == 'int8':
            # Dynamic quantization
            quantized_model = quantization.quantize_dynamic(
                model, {nn.Linear, nn.Embedding}, dtype=torch.qint8
            )
        elif quant_type == 'int4':
            # Simulated int4 (not natively supported)
            quantized_model = copy.deepcopy(model)
            for name, param in quantized_model.named_parameters():
                if param.dim() == 2:  # Linear weights
                    scale = param.abs().max() / 7.0
                    quantized_param = torch.round(param / scale) * scale
                    param.data.copy_(quantized_param)
        else:
            quantized_model = model
        
        return quantized_model
    
    def measure_latency(self, model, input_ids, num_runs=100):
        """Measure inference latency"""
        model = model.to(self.device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_ids)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        elapsed = (time.time() - start_time) / num_runs
        
        return elapsed * 1000  # Convert to ms
    
    def measure_memory_usage(self, model):
        """Estimate memory usage"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        total_size = param_size + buffer_size
        return total_size / 1024  # KB
    
    def estimate_power_consumption(self, latency_ms, device_type='microcontroller'):
        """Estimate power consumption based on latency"""
        # Rough estimates based on typical power consumption
        if device_type == 'microcontroller':
            # ARM Cortex-M4: ~100mW active
            power_mw = 100
        elif device_type == 'rpi':
            # Raspberry Pi 4: ~3W
            power_mw = 3000
        else:
            # Desktop GPU: ~200W
            power_mw = 200000
        
        # Energy per inference
        energy_joules = power_mw * (latency_ms / 1000) / 1000
        
        return energy_joules
    
    def check_deployment_feasibility(self, model, latency_ms, memory_kb, target_device='microcontroller'):
        """Check if model is feasible for target device"""
        constraints = {
            'microcontroller': {'max_memory_kb': 256, 'max_latency_ms': 100},
            'rpi': {'max_memory_kb': 4096, 'max_latency_ms': 50},
            'desktop': {'max_memory_kb': 16384, 'max_latency_ms': 10}
        }
        
        constraint = constraints.get(target_device, constraints['microcontroller'])
        
        feasible_memory = memory_kb <= constraint['max_memory_kb']
        feasible_latency = latency_ms <= constraint['max_latency_ms']
        
        return {
            'device': target_device,
            'memory_feasible': feasible_memory,
            'latency_feasible': feasible_latency,
            'overall_feasible': feasible_memory and feasible_latency,
            'max_memory_kb': constraint['max_memory_kb'],
            'max_latency_ms': constraint['max_latency_ms']
        }
    
    def run_edge_deployment_experiment(self):
        print("="*80)
        print("EDGE AI DEPLOYMENT EXPERIMENTS")
        print("="*80)
        
        # Create model
        model = EdgeAssociativeModel(vocab_size=30, d_model=64)
        
        # Create dataset
        dataset = NeedleHaystackDataset(num_samples=500, vocab_size=30)
        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Train model briefly
        print("\n1. Training Model...")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        for epoch in range(5):
            total_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                logits = model(inputs)
                
                batch, seq, vocab = logits.shape
                loss = F.cross_entropy(logits[:, -1, :], targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 2 == 0:
                print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        # Evaluate accuracy
        print("\n2. Evaluating Model...")
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in train_loader:
                logits = model(inputs)
                pred = logits[:, -1, :].argmax(dim=-1)
                correct += (pred == targets).sum().item()
                total += targets.size(0)
        
        accuracy = correct / total
        print(f"  Accuracy: {accuracy:.2%}")
        
        # Create test input
        test_input = torch.randint(0, 30, (1, 63))
        
        # 3. Measure baseline performance
        print("\n3. Measuring Baseline Performance...")
        baseline_latency = self.measure_latency(model, test_input, num_runs=100)
        baseline_memory = self.measure_memory_usage(model)
        
        print(f"  Baseline Latency: {baseline_latency:.2f} ms")
        print(f"  Baseline Memory: {baseline_memory:.2f} KB")
        
        # 4. Quantize models
        print("\n4. Quantizing Models...")
        
        # INT8 quantization
        model_int8 = self.quantize_model(model, quant_type='int8')
        int8_latency = self.measure_latency(model_int8, test_input, num_runs=100)
        int8_memory = self.measure_memory_usage(model_int8)
        
        print(f"  INT8 Latency: {int8_latency:.2f} ms ({(int8_latency/baseline_latency):.2f}x)")
        print(f"  INT8 Memory: {int8_memory:.2f} KB ({(int8_memory/baseline_memory):.2f}x)")
        
        # INT4 quantization (simulated)
        model_int4 = self.quantize_model(model, quant_type='int4')
        int4_latency = self.measure_latency(model_int4, test_input, num_runs=100)
        int4_memory = self.measure_memory_usage(model_int4)
        
        print(f"  INT4 Latency: {int4_latency:.2f} ms ({(int4_latency/baseline_latency):.2f}x)")
        print(f"  INT4 Memory: {int4_memory:.2f} KB ({(int4_memory/baseline_memory):.2f}x)")
        
        # 5. Check deployment feasibility
        print("\n5. Checking Deployment Feasibility...")
        
        variants = [
            ('baseline', model, baseline_latency, baseline_memory),
            ('int8', model_int8, int8_latency, int8_memory),
            ('int4', model_int4, int4_latency, int4_memory)
        ]
        
        feasibility_results = {}
        for name, m, lat, mem in variants:
            for device in ['microcontroller', 'rpi', 'desktop']:
                feasible = self.check_deployment_feasibility(m, lat, mem, device)
                if device not in feasibility_results:
                    feasibility_results[device] = {}
                feasibility_results[device][name] = feasible
        
        for device, results in feasibility_results.items():
            print(f"\n  {device.upper()}:")
            for variant, feasible in results.items():
                status = "✓" if feasible['overall_feasible'] else "✗"
                print(f"    {status} {variant}: {feasible['max_memory_kb']}KB max, {feasible['max_latency_ms']}ms max")
        
        # 6. Estimate power consumption
        print("\n6. Estimating Power Consumption...")
        
        power_results = {}
        for name, m, lat, mem in variants:
            power_results[name] = {
                'microcontroller_joules': self.estimate_power_consumption(lat, 'microcontroller'),
                'rpi_joules': self.estimate_power_consumption(lat, 'rpi'),
                'desktop_joules': self.estimate_power_consumption(lat, 'desktop')
            }
        
        for name, power in power_results.items():
            print(f"  {name}:")
            print(f"    Microcontroller: {power['microcontroller_joules']*1000:.2f} mJ")
            print(f"    Raspberry Pi: {power['rpi_joules']*1000:.2f} mJ")
            print(f"    Desktop: {power['desktop_joules']*1000:.2f} mJ")
        
        # Compile results
        self.results = {
            'accuracy': accuracy,
            'baseline': {
                'latency_ms': baseline_latency,
                'memory_kb': baseline_memory
            },
            'int8': {
                'latency_ms': int8_latency,
                'memory_kb': int8_memory,
                'latency_speedup': baseline_latency / int8_latency,
                'memory_reduction': baseline_memory / int8_memory
            },
            'int4': {
                'latency_ms': int4_latency,
                'memory_kb': int4_memory,
                'latency_speedup': baseline_latency / int4_latency,
                'memory_reduction': baseline_memory / int4_memory
            },
            'feasibility': feasibility_results,
            'power': power_results
        }
        
        # Determine outcome
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        # Check if any variant is feasible for microcontroller
        mc_feasible = any(
            feasibility_results['microcontroller'][v]['overall_feasible']
            for v in ['baseline', 'int8', 'int4']
        )
        
        # Check if accuracy is acceptable
        accuracy_acceptable = accuracy >= 0.7
        
        print(f"\nDeployment Analysis:")
        print(f"  Model Accuracy: {accuracy:.2%} {'✓' if accuracy_acceptable else '✗'}")
        print(f"  Microcontroller Feasible: {'✓' if mc_feasible else '✗'}")
        
        if mc_feasible and accuracy_acceptable:
            print(f"\n✓ EDGE DEPLOYMENT SUCCESSFUL")
            print(f"  Publication: Industry conference / Workshop")
            print(f"  Application: Smart assistants, IoT devices")
            print(f"  Value: Real-world deployment capability")
        elif mc_feasible and not accuracy_acceptable:
            print(f"\n⚠ EDGE DEPLOYMENT PARTIAL")
            print(f"  Publication: Feasibility study")
            print(f"  Next Steps: Improve accuracy, reduce size")
        else:
            print(f"\n✗ EDGE DEPLOYMENT NOT FEASIBLE (yet)")
            print(f"  Publication: Limitations analysis")
            print(f"  Next Steps: Further optimization, hardware advances")
        
        # Save results
        output_dir = Path(__file__).parent.parent / 'experiments' / 'edge'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'edge_deployment.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Results saved: {output_dir / 'edge_deployment.json'}")
        
        return self.results


if __name__ == '__main__':
    benchmark = EdgeDeploymentBenchmark()
    results = benchmark.run_edge_deployment_experiment()
    print("\n✓ Edge deployment experiments complete!")
