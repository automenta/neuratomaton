import torch
from ana.models import ANAModel
from ana.config import ANAConfig
import time

class InferenceEngine:
    def __init__(self, model: ANAModel, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate_stream(self, prompt_ids, max_new_tokens=20):
        """
        Simulates streaming generation.
        In a real implementation, this would use KV-caching or recurrent state passing.
        Since ANA is recurrent, we just pass the hidden state.
        But ANAModel forward() doesn't expose state passing easily yet.
        So we simulate by re-running for now, or assume we can optimize later.

        Actually, ANAModel needs a `forward_step` method for true efficient inference.
        I will assume we just re-run for now as a baseline, but print tokens as they come.
        """
        curr_ids = prompt_ids.clone()

        print(f"Generating {max_new_tokens} tokens...", end="", flush=True)

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.model(curr_ids)

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # Yield for streaming
            yield next_token.item()

    def run_demo(self):
        prompt = torch.tensor([[1, 2, 3]]).to(self.device)
        start = time.time()
        count = 0
        for token in self.generate_stream(prompt):
            # print(token, end=" ", flush=True)
            count += 1
        end = time.time()
        print(f"\nGenerated {count} tokens in {end - start:.2f}s ({count/(end-start):.2f} tok/s)")

if __name__ == "__main__":
    config = ANAConfig(vocab_size=100, d_model=32)
    model = ANAModel(config)
    engine = InferenceEngine(model)
    engine.run_demo()
