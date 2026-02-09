
import unittest
import torch
import torch.nn as nn
from ana.models import LinearRecurrentUnit, ANAConfig, ANAModel

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
            use_controller=True
        )
        self.lru = LinearRecurrentUnit(self.config)

    def test_lru_equivalence(self):
        x = torch.randn(self.config.batch_size, 10, self.config.d_model) # 10 steps

        # Run sequential
        h_prev = None
        outputs_seq = []

        # We need to make sure static params are used identically.
        # forward() uses static params if dynamic_gates is None.

        for t in range(x.size(1)):
            xt = x[:, t, :]
            yt, h_prev, _ = self.lru(xt, h_prev)
            outputs_seq.append(yt)

        y_seq = torch.stack(outputs_seq, dim=1)

        # Run parallel
        y_par, h_par = self.lru.forward_sequence(x)

        # Compare
        diff = torch.abs(y_seq - y_par).max()
        print(f"LRU Max Diff: {diff.item()}")
        self.assertLess(diff.item(), 1e-5)

    def test_anamodel_equivalence(self):
        model = ANAModel(self.config)
        model.eval() # Ensure dropout (if any) is off, though we have none.

        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, 10))

        # Sequential
        self.config.use_parallel_scan = False
        logits_seq, _ = model(input_ids)

        # Parallel
        self.config.use_parallel_scan = True
        logits_par, _ = model(input_ids)

        diff = torch.abs(logits_seq - logits_par).max()
        print(f"Model Max Diff: {diff.item()}")
        self.assertLess(diff.item(), 1e-5)

if __name__ == '__main__':
    unittest.main()
