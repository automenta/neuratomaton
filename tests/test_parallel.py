import unittest
import torch
from ana.models import ANAModel, LinearRecurrentUnit
from ana.config import ANAConfig


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
        x = torch.randn(self.config.batch_size, 10, self.config.d_model)

        self.config.use_parallel_scan = False
        self.lru.config.use_parallel_scan = False
        y_jit, h_jit = self.lru.forward_sequence(x)

        self.config.use_parallel_scan = True
        self.lru.config.use_parallel_scan = True
        y_log, h_log = self.lru.forward_sequence(x)

        diff = torch.abs(y_jit - y_log).max()
        self.assertLess(diff.item(), 1e-5)

    def test_anamodel_equivalence(self):
        model = ANAModel(self.config)
        model.eval()

        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, 10))

        self.config.use_parallel_scan = False
        model.config.use_parallel_scan = False
        logits_seq, _ = model(input_ids)

        self.config.use_parallel_scan = True
        model.config.use_parallel_scan = True
        logits_par, _ = model(input_ids)

        diff = torch.abs(logits_seq - logits_par).max()
        self.assertLess(diff.item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
