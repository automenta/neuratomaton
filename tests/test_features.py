
import unittest
import torch
from ana.models import ANAModel, HoloLink
from ana.config import ANAConfig

class TestNewFeatures(unittest.TestCase):
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
            track_count=2,
            max_thinking_steps=0
        )

    def test_binding_strength(self):
        holo = HoloLink(self.config, input_dim=self.config.state_dim * self.config.track_count)
        self.assertTrue(hasattr(holo, 'binding_strength'))
        self.assertEqual(holo.binding_strength.item(), 1.0)

        # Test forward pass with binding
        x = torch.randn(self.config.batch_size, self.config.d_model)
        h = torch.randn(self.config.batch_size, self.config.state_dim * self.config.track_count)
        m_prev = None

        ret, m_new = holo(x, h, m_prev)
        self.assertEqual(ret.shape, (self.config.batch_size, self.config.d_model))

    def test_thinking_steps(self):
        self.config.max_thinking_steps = 2
        model = ANAModel(self.config)

        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, 5))

        # Should use forward_sequential
        logits, info = model(input_ids)
        self.assertEqual(logits.shape, (self.config.batch_size, 5, self.config.vocab_size))

    def test_thinking_steps_equivalence_zero(self):
        # With 0 thinking steps, sequential should match parallel (if logic is sound)
        # Note: parallel scan applies updates in parallel. Sequential loop applies sequentially.
        # They match mathematically.

        self.config.max_thinking_steps = 0
        model = ANAModel(self.config)

        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, 5))

        # Parallel
        self.config.use_parallel_scan = True
        model.config.use_parallel_scan = True
        y_par, _ = model(input_ids)

        # Sequential
        self.config.use_parallel_scan = False
        model.config.use_parallel_scan = False
        y_seq, _ = model(input_ids)

        # Note: forward_sequential implementation logic for tracks/mixing must match forward_parallel
        # We implemented a complex mixing logic in forward_sequential in previous step.
        # Let's hope it matches.

        diff = torch.abs(y_par - y_seq).max()
        print(f"Thinking 0 Seq vs Par Diff: {diff.item()}")
        self.assertLess(diff.item(), 1e-5)

if __name__ == '__main__':
    unittest.main()
