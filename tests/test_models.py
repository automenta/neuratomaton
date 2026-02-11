import unittest
import torch
import torch.nn as nn
from ana.models import ANAModel, LinearRecurrentUnit, HoloLink, HyperController, BaselineSSM
from ana.config import ANAConfig


class TestModels(unittest.TestCase):
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

    def test_lru_forward(self):
        lru = LinearRecurrentUnit(self.config)
        x = torch.randn(self.config.batch_size, self.config.d_model)
        h_prev = torch.zeros(self.config.batch_size, self.config.state_dim)

        y, h_new = lru(x, h_prev)

        self.assertEqual(y.shape, (self.config.batch_size, self.config.d_model))
        self.assertEqual(h_new.shape, (self.config.batch_size, self.config.state_dim))

    def test_hololink_forward(self):
        holo = HoloLink(self.config, input_dim=self.config.state_dim * self.config.track_count)
        x = torch.randn(self.config.batch_size, self.config.d_model)
        h = torch.randn(self.config.batch_size, self.config.state_dim * self.config.track_count)
        m_prev = None

        retrieved, m_new = holo(x, h, m_prev)

        self.assertEqual(retrieved.shape, (self.config.batch_size, self.config.d_model))
        self.assertEqual(m_new.shape, (self.config.batch_size, self.config.key_dim, self.config.d_model))

    def test_controller_forward(self):
        ctl = HyperController(self.config)
        x = torch.randn(self.config.batch_size, self.config.d_model)

        track_outputs, g_ret, g_halt = ctl(x)

        self.assertEqual(len(track_outputs), self.config.track_count)
        alpha, beta, mix = track_outputs[0]
        self.assertEqual(alpha.shape, (self.config.batch_size, 1))
        self.assertEqual(beta.shape, (self.config.batch_size, 1))
        self.assertEqual(mix.shape, (self.config.batch_size, 1))

    def test_anamodel_forward(self):
        model = ANAModel(self.config)
        seq_len = 5
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, seq_len))

        logits, info = model(input_ids, return_info=True)

        self.assertEqual(logits.shape, (self.config.batch_size, seq_len, self.config.vocab_size))

    def test_ablation_no_controller(self):
        self.config.use_controller = False
        model = ANAModel(self.config)
        seq_len = 5
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, seq_len))

        logits, _ = model(input_ids)
        self.assertEqual(logits.shape, (self.config.batch_size, seq_len, self.config.vocab_size))

    def test_ablation_no_hololink(self):
        self.config.use_hololink = False
        model = ANAModel(self.config)
        seq_len = 5
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, seq_len))

        logits, _ = model(input_ids)
        self.assertEqual(logits.shape, (self.config.batch_size, seq_len, self.config.vocab_size))

    def test_baseline_ssm_forward(self):
        model = BaselineSSM(self.config)
        seq_len = 5
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, seq_len))

        logits, _ = model(input_ids)
        self.assertEqual(logits.shape, (self.config.batch_size, seq_len, self.config.vocab_size))

    def test_param_comparison(self):
        ana = ANAModel(self.config)
        baseline = BaselineSSM(self.config)
        
        ana_params = sum(p.numel() for p in ana.parameters())
        baseline_params = sum(p.numel() for p in baseline.parameters())
        
        # ANA should have more params due to HoloLink and Controller
        self.assertGreater(ana_params, baseline_params)

    def test_thinking_steps_execution(self):
        self.config.max_thinking_steps = 2
        model = ANAModel(self.config)
        seq_len = 5
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, seq_len))

        logits, info = model(input_ids, return_info=True)
        self.assertEqual(logits.shape, (self.config.batch_size, seq_len, self.config.vocab_size))

        # Check info log has steps info
        if info:
            self.assertIn('avg_steps', info[0])

    def test_dynamic_halting(self):
        self.config.max_thinking_steps = 5
        self.config.use_controller = True
        model = ANAModel(self.config)

        # Force controller to output high halt probability
        # The head outputs: [alpha, beta, mix] * tracks + ret + halt
        # halt is the last element
        with torch.no_grad():
            for layer in model.layers:
                layer['controller'].head.bias[-1].fill_(10.0) # High positive bias => high prob

        input_ids = torch.randint(0, self.config.vocab_size, (2, 5))

        logits, info = model(input_ids, return_info=True)

        # Should halt after step 0 (so 1 step total)
        # Because at step 0, it predicts halt=True.
        # Halting mask becomes 1 at end of step 0.
        # Loop check at start of step 1 sees all halted -> Break.
        # So steps_taken should be 1.

        if info:
            avg_steps = info[0]['avg_steps']
            self.assertAlmostEqual(avg_steps, 1.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()
