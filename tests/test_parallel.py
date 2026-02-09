
import unittest
import torch
import torch.nn as nn
from ana.models import ANAModel, LinearRecurrentUnit, ANAConfig

class TestParallelScan(unittest.TestCase):
    def setUp(self):
        self.config = ANAConfig(
            vocab_size=10,
            d_model=8,
            state_dim=16,
            num_layers=1,
            batch_size=2,
            key_dim=4,
            use_hololink=True,
            use_controller=True,
            track_count=2
        )
        self.lru = LinearRecurrentUnit(self.config)

    def test_lru_equivalence(self):
        x = torch.randn(self.config.batch_size, 10, self.config.d_model) # 10 steps

        # Method 1: JIT Scan (forward_sequence with use_parallel_scan=False)
        self.config.use_parallel_scan = False
        # self.lru.config is a reference to self.config?
        # Let's assume it is. If not, manual set.
        self.lru.config.use_parallel_scan = False
        y_jit, h_jit = self.lru.forward_sequence(x)

        # Method 2: Log Scan (forward_sequence with use_parallel_scan=True)
        self.config.use_parallel_scan = True
        self.lru.config.use_parallel_scan = True
        y_log, h_log = self.lru.forward_sequence(x)

        # Compare
        diff = torch.abs(y_jit - y_log).max()
        print(f"LRU Max Diff (JIT vs Log): {diff.item()}")
        self.assertLess(diff.item(), 1e-5)

    def test_anamodel_equivalence(self):
        model = ANAModel(self.config)
        model.eval()

        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, 10))

        # Sequential (Python Loop + Single Step)
        self.config.use_parallel_scan = False
        model.config.use_parallel_scan = False

        logits_seq, _ = model(input_ids)

        # Parallel (Log Scan)
        self.config.use_parallel_scan = True
        model.config.use_parallel_scan = True

        logits_par, _ = model(input_ids)

        diff = torch.abs(logits_seq - logits_par).max()
        print(f"Model Max Diff (Seq vs LogPar): {diff.item()}")
        self.assertLess(diff.item(), 1e-5)

if __name__ == '__main__':
    unittest.main()
