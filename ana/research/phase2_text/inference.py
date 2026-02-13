import torch
import time
import sys
from ana.models import ANAModel
from ana.config import ANAConfig
from ana.research.core import ExperimentBase, ExperimentRegistry

@ExperimentRegistry.register(phase=2, name="inference")
class InferenceExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "inference"

    @property
    def phase(self) -> int:
        return 2

    def setup(self):
        self.model = ANAModel(self.config).to(self.device)
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
                prompt = torch.tensor([[1, 2, 3]]).to(self.device)

                print("ANA: ", end="", flush=True)
                for token in self.generate_stream(prompt, max_new_tokens=30):
                    # Detokenize (Dummy: print token ID)
                    print(f"{token}", end=" ", flush=True)
                print("\n")

            except KeyboardInterrupt:
                break
        print("\nGoodbye!")

    def run_demo(self):
        self.results.log("Running non-interactive demo...")
        prompt = torch.tensor([[1, 2, 3]]).to(self.device)
        start = time.time()
        count = 0

        output_tokens = []
        for token in self.generate_stream(prompt, delay=0.02):
            output_tokens.append(token)
            count += 1

        end = time.time()
        self.results.log(f"Generated tokens: {output_tokens}")
        self.results.log(f"Generated {count} tokens in {end - start:.2f}s ({count/(end-start):.2f} tok/s)")
        self.results.save_json("inference_results.json", {"tokens": output_tokens, "speed": count/(end-start)})

    def execute(self):
        # We can't easily capture the interactive flag from the base class config yet,
        # but we can rely on an external flag or just default to demo.
        # Since interactive mode is special, we might want to check if sys.argv has it,
        # or if we added it to config. run_research.py adds `interactive` arg but doesn't pass it to config.
        # Let's assume we run demo unless we hack a way to know.

        # Actually, run_research.py doesn't put `interactive` into `ANAConfig`.
        # However, `ExperimentBase` takes `config`.
        # I should probably update `ANAConfig` to support `interactive` or handle it differently.
        # For now, I'll check sys.argv as a fallback or just run demo.

        is_interactive = "--interactive" in sys.argv

        if is_interactive:
            self.run_interactive()
        else:
            self.run_demo()

if __name__ == "__main__":
    config = ANAConfig(vocab_size=100, d_model=32)
    exp = InferenceExperiment(config)
    exp.run()
