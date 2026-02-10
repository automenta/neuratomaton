import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .config_v2 import ANAv2Config
from .models_v3 import (
    GumbelSoftmax, StackFrame, MetaStateStack, 
    FaultTraceBuffer, CortexController, SpecializedTracks,
    parallel_scan_hillis_steele_v2
)


class ANAv2Interpreter:
    @staticmethod
    def execute_opcode(opcode_vector, stack_vector, tracks_states, config):
        batch_size = stack_vector.size(0)
        num_tracks = 3
        
        op_type = torch.argmax(opcode_vector, dim=-1)
        
        A_matrices = []
        B_matrices = []
        
        for i in range(num_tracks):
            A = torch.sigmoid(torch.randn(batch_size, 1, device=stack_vector.device))
            B = torch.sigmoid(torch.randn(batch_size, 1, device=stack_vector.device))
            A_matrices.append(A)
            B_matrices.append(B)
        
        return A_matrices, B_matrices


class ANAv2Model(nn.Module):
    def __init__(self, config: ANAv2Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        if config.use_position_encoding:
            self.register_buffer('pos_encoding', self._create_sinusoidal_encoding(config.max_seq_len, config.d_model))
        
        self.tracks = SpecializedTracks(config)
        
        self.fault_buffer = FaultTraceBuffer(config)
        self.cortex = CortexController(config)
        self.stack = MetaStateStack(config)
        
        self.mixer = nn.Linear(config.total_track_dim, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        self.rule_success_head = nn.Linear(config.d_model, 2)
        
        nn.init.xavier_uniform_(self.mixer.weight)
        nn.init.zeros_(self.mixer.bias)
        nn.init.xavier_uniform_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
    
    def _create_sinusoidal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def _add_position_encoding(self, x):
        if not self.config.use_position_encoding:
            return x
        seq_len = x.size(1)
        return x + self.pos_encoding[:, :seq_len, :]
    
    def _initialize_stack(self, batch_size):
        return [[]]
    
    def _initialize_track_states(self, batch_size):
        return {
            'syntax': None,
            'semantic': None,
            'logic': None
        }
    
    def forward(self, input_ids, return_info=False, return_stack_trace=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        x = self.embedding(input_ids)
        x = self._add_position_encoding(x)
        
        stack_list = self._initialize_stack(batch_size)
        track_states = self._initialize_track_states(batch_size)
        
        fault_summary = self.fault_buffer.get_summary().expand(batch_size, -1)
        
        outputs = []
        all_info = []
        stack_traces = [] if return_stack_trace else None
        
        for t in range(seq_len):
            xt = x[:, t, :]
            
            top_stack_vec = torch.zeros(batch_size, self.config.stack_dim, device=device)
            
            cortex_out = self.cortex(xt, top_stack_vec, fault_summary)
            
            stack_result = self.stack(xt, fault_summary, stack_list)
            stack_list = stack_result['stack']
            opcodes = stack_result['opcodes']
            
            track_out, new_track_states = self.tracks(
                xt,
                h_syntax=track_states['syntax'],
                h_semantic=track_states['semantic'],
                h_logic=track_states['logic'],
                alpha_mods=cortex_out['alpha_mods'],
                beta_mods=cortex_out['beta_mods']
            )
            track_states = new_track_states
            
            layer_out = self.mixer(track_out)
            
            xt = xt + layer_out
            
            outputs.append(xt)
            
            if return_info:
                info = {
                    'opcode': opcodes[0].detach().cpu().numpy(),
                    'stack_depth': 1 if stack_list and isinstance(stack_list[0], StackFrame) else 0,
                    'temp': self.stack.gumbel_temp
                }
                all_info.append(info)
            
            if return_stack_trace:
                stack_traces.append([f.vector.clone() if isinstance(f, StackFrame) else f for f in stack_list])
        
        output_seq = torch.stack(outputs, dim=1)
        output_seq = self.norm(output_seq)
        
        logits = self.output_head(output_seq)
        rule_logits = self.rule_success_head(output_seq)
        
        if return_stack_trace:
            return logits, rule_logits, stack_traces
        if return_info:
            return logits, rule_logits, all_info
        return logits, rule_logits
    
    def forward_parallel(self, input_ids, return_info=False):
        return self.forward(input_ids, return_info=return_info)
    
    def compute_loss(self, logits, rule_logits, targets, loss_weights=(1.0, 0.1, 0.01)):
        batch, seq, vocab = logits.shape
        
        ce_loss = F.cross_entropy(logits.view(-1, vocab), targets.view(-1), ignore_index=0)
        
        rule_loss = F.cross_entropy(rule_logits.view(-1, 2), torch.zeros(batch * seq, dtype=torch.long, device=logits.device))
        
        density_reg = 0.0
        for name, param in self.named_parameters():
            if 'opcode_logits' in name or 'delta' in name:
                density_reg += torch.mean(torch.abs(param))
        
        total_loss = loss_weights[0] * ce_loss + loss_weights[1] * rule_loss + loss_weights[2] * density_reg
        
        return {
            'total': total_loss,
            'ce': ce_loss,
            'rule': rule_loss,
            'density': density_reg
        }
    
    def update_fault_buffer(self, predictions, targets, token_ids):
        with torch.no_grad():
            errors = predictions - targets
            batch_size = errors.size(0)
            
            for b in range(batch_size):
                error_b = errors[b:b+1]
                tok_id = token_ids[b:b+1] if token_ids is not None else None
                self.fault_buffer(error_b, tok_id)
    
    def reset_state(self):
        self.fault_buffer.reset()
