import torch
from ana.models import ANAModel
from ana.config import ANAConfig
import time
import sys

class InferenceEngine:
    def __init__(self, model: ANAModel, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def generate_stream(self, prompt_ids, max_new_tokens=20, delay=0.05):
        """
        Simulates streaming generation with a small delay for visualization.
        """
        curr_ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits, _ = self.model(curr_ids)

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            time.sleep(delay)
            yield next_token.item()

    def run_interactive(self):
        print("\n--- ANA Interactive Chat (Phase 2) ---")
        print("Type 'exit' to quit.\n")

        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break

                # Tokenize (Dummy: just map chars to int if possible, or random)
                # In real scenario: tokenizer.encode(user_input)
                # Here we just use a dummy start token sequence
                prompt = torch.tensor([[1, 2, 3]]).to(self.device)

                print("ANA: ", end="", flush=True)
                for token in self.generate_stream(prompt, max_new_tokens=30):
                    # Detokenize (Dummy: print token ID)
                    # In real scenario: tokenizer.decode([token])
                    print(f"{token}", end=" ", flush=True)
                print("\n")

            except KeyboardInterrupt:
                break
        print("\nGoodbye!")

    def run_demo(self):
        print("Running non-interactive demo...")
        prompt = torch.tensor([[1, 2, 3]]).to(self.device)
        start = time.time()
        count = 0
        print("Output: ", end="", flush=True)
        for token in self.generate_stream(prompt, delay=0.02):
            print(token, end=" ", flush=True)
            count += 1
        end = time.time()
        print(f"\nGenerated {count} tokens in {end - start:.2f}s ({count/(end-start):.2f} tok/s)")

if __name__ == "__main__":
    config = ANAConfig(vocab_size=100, d_model=32)
    model = ANAModel(config)
    engine = InferenceEngine(model)
    # Check if interactive mode requested
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        engine.run_interactive()
    else:
        engine.run_demo()
