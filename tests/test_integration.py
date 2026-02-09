
import unittest
import torch
import torch.nn as nn
from ana.models import ANAModel
from ana.config import ANAConfig
import torch.optim as optim

class TestIntegration(unittest.TestCase):
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
        self.model = ANAModel(self.config)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def test_training_step(self):
        seq_len = 5
        x = torch.randint(0, self.config.vocab_size, (self.config.batch_size, seq_len))
        y = torch.randint(0, self.config.vocab_size, (self.config.batch_size, seq_len))

        self.optimizer.zero_grad()

        logits, _ = self.model(x)

        loss = self.criterion(logits.view(-1, self.config.vocab_size), y.view(-1))

        loss.backward()
        self.optimizer.step()

        self.assertIsNotNone(loss.item())
        self.assertGreater(loss.item(), 0.0)

if __name__ == '__main__':
    unittest.main()
