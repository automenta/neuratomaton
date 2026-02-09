
import unittest
import torch
import torch.nn as nn
from ana.models import ANAModel, LinearRecurrentUnit, HoloLink, HyperController
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
            use_controller=True
        )

    def test_lru_forward(self):
        lru = LinearRecurrentUnit(self.config)
        x = torch.randn(self.config.batch_size, self.config.d_model)
        h_prev = torch.zeros(self.config.batch_size, self.config.state_dim)

        y, h_new, gates = lru(x, h_prev)

        self.assertEqual(y.shape, (self.config.batch_size, self.config.d_model))
        self.assertEqual(h_new.shape, (self.config.batch_size, self.config.state_dim))

    def test_hololink_forward(self):
        # Input to HoloLink is concatenated state of track A and B (state_dim * 2)
        holo = HoloLink(self.config, input_dim=self.config.state_dim * 2)
        x = torch.randn(self.config.batch_size, self.config.d_model)
        h = torch.randn(self.config.batch_size, self.config.state_dim * 2)
        m_prev = None

        retrieved, m_new = holo(x, h, m_prev)

        self.assertEqual(retrieved.shape, (self.config.batch_size, self.config.d_model))
        self.assertEqual(m_new.shape, (self.config.batch_size, self.config.key_dim, self.config.d_model))

    def test_controller_forward(self):
        ctl = HyperController(self.config)
        x = torch.randn(self.config.batch_size, self.config.d_model)

        ga_A, gb_A, ga_B, gb_B, g_ret = ctl(x)

        self.assertEqual(ga_A.shape, (self.config.batch_size, 1))
        self.assertEqual(g_ret.shape, (self.config.batch_size, 1))

    def test_anamodel_forward(self):
        model = ANAModel(self.config)
        seq_len = 5
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.batch_size, seq_len))

        logits, info = model(input_ids, return_info=True)

        self.assertEqual(logits.shape, (self.config.batch_size, seq_len, self.config.vocab_size))
        self.assertTrue(isinstance(info, list))

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

if __name__ == '__main__':
    unittest.main()
