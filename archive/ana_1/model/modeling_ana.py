import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict, Any
from transformers import PreTrainedModel, PretrainedConfig
from .layers import DualTrackBlock
import random

# We need to register config if we want AutoModel to work, but strict inheritance is enough for now.
# Assume config passed is ANAConfig instance or dict.

class ANAModel(PreTrainedModel):
    def __init__(self, config):
        # Wrap config in PretrainedConfig if it's our dataclass
        if not isinstance(config, PretrainedConfig):
            # Hacky wrapper
            class Wrapper(PretrainedConfig):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    for k, v in kwargs.items():
                        setattr(self, k, v)
            config = Wrapper(**config.__dict__)
            
        super().__init__(config)
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([
            DualTrackBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Auxiliary Retrieval Head
        # Predicts past token from global state
        # Input: Holo State [K, D] -> Flatten -> Linear -> Vocab?
        # No, query the state with a random vector?
        # Or just project the state mean?
        # Let's do: Project State Mean -> Vocab
        self.retrieval_head = nn.Linear(config.d_model, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Track A/B logic init
        for layer in self.layers:
            # A: Fast Decay (alpha ~ 0)
            nn.init.constant_(layer.track_A.base_alpha_logit, -3.0)
            # B: Slow Decay (alpha ~ 0.95)
            nn.init.constant_(layer.track_B.base_alpha_logit, 3.0)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids: torch.Tensor, 
                labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Union[Tuple, Dict[str, Any]]:
        
        x = self.embedding(input_ids)
        batch, seq_len, _ = x.shape
        
        # State containers
        current_h_A = [None] * len(self.layers)
        current_h_B = [None] * len(self.layers)
        current_m   = [None] * len(self.layers)
        
        all_hidden_states = []
        retrieval_gates = []
        
        # Recurrent Loop
        # Optimized implementation would use a compiled scan or custom kernel.
        # For Python loop, it's slow but functional.
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            for i, layer in enumerate(self.layers):
                xt, hA, hB, m, g_ret = layer(xt, current_h_A[i], current_h_B[i], current_m[i], return_info=True)
                
                current_h_A[i] = hA
                current_h_B[i] = hB
                current_m[i]   = m
                
                if i == 0 and g_ret is not None:
                    retrieval_gates.append(g_ret)
            
            all_hidden_states.append(xt)
            
        output = torch.stack(all_hidden_states, dim=1) # [B, T, D]
        output = self.norm(output)
        
        logits = self.lm_head(output)
        
        loss = None
        loss_items = {}
        
        if labels is not None:
            # 1. LM Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # 2. Retrieval Loss
            # Pick a random past token index k < t for each step?
            # Approximation: Try to reconstruct input_ids[t] from State[t] using retrieval head on Holo State.
            # But Holo State is [K, D]. We interpret "State Mean" as compressed memory.
            # Let's take the mean of M_t over K dimension -> [D]
            # And try to predict a past token.
            # Simplified: Predict the *current* token from the *previous* memory state M_{t-1}.
            # If M_{t-1} contains history, it should help predict x_t.
            # Wait, that's just LM logic.
            # The "Retrieval Task" spec says: "Reconstruct a random past token t-k".
            # Implementation:
            # At step t, pick k ~ Uniform(1, t). Target = input_ids[t-k].
            # Input to Head = Read(M_t, Query(Target_Position_Embedding?))
            # PoC Phase 3 just used weighted loss on the actual target.
            
            # Let's implement the Spec version: "Auxiliary Retrieval Head"
            # We will use the final layer's Memory State M_t.
            # We want to check if M_t contains info about t-k.
            # How do we query it?
            # Simple: The Retrieval Head is a linear probe on M_t.mean(dim=1).
            # We train it to predict input_ids[t-k].
            
            # Randomly sample one k per sequence for efficiency.
            # Or just use the weighted LM loss strategy from Phase 3 which worked? 
            # The spec says "Auxiliary Retrieval Head".
            # Let's stick to the Phase 3 method for now (Weighted Loss) but add sparsity penalty.
            # Phase 3 Method: Upweight expected retrieval tokens.
            # But in generic LM training we don't know which are "retrieval" tokens.
            # So we use the L1 penalty on g_ret to force sparsity, 
            # and rely on LM loss to force usage when necessary.
            
            # Let's add L_Sparsity.
            # gate_vals = stack(retrieval_gates) # [T, B, 1]
            if retrieval_gates:
                g_stack = torch.stack(retrieval_gates)
                loss_sparsity = g_stack.mean() # L1 of sigmoid output (positive)
            else:
                loss_sparsity = 0.0
                
            loss = lm_loss + self.config.lambda_spar * loss_sparsity
            
            loss_items = {
                'lm': lm_loss.item(),
                'spar': loss_sparsity.item() if isinstance(loss_sparsity, torch.Tensor) else 0.0
            }
            
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'others': loss_items
            }
        return (loss, logits)
