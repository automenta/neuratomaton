import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass


@dataclass
class ARSample:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    key_position: int
    value_position: int
    query_position: int
    num_noise: int


class AssociativeRecallDataset(Dataset):
    def __init__(
        self,
        vocab_size: int = 50,
        seq_len: int = 32,
        num_samples: int = 10000,
        noise_tokens_min: int = 5,
        noise_tokens_max: int = 15,
        num_kv_pairs: int = 1,
        seed: Optional[int] = None,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.noise_tokens_min = noise_tokens_min
        self.noise_tokens_max = noise_tokens_max
        self.num_kv_pairs = num_kv_pairs
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[ARSample]:
        samples = []
        
        for _ in range(self.num_samples):
            num_noise = random.randint(self.noise_tokens_min, self.noise_tokens_max)
            
            input_ids = torch.zeros(self.seq_len, dtype=torch.long)
            target_ids = torch.zeros(self.seq_len, dtype=torch.long)
            
            kv_pairs = []
            for i in range(self.num_kv_pairs):
                key = random.randint(1, self.vocab_size - 1)
                value = random.randint(1, self.vocab_size - 1)
                kv_pairs.append((key, value))
            
            noise_tokens = [random.randint(1, self.vocab_size - 1) for _ in range(num_noise)]
            
            positions = list(range(self.seq_len))
            random.shuffle(positions)
            
            kv_positions = sorted(positions[:2 * self.num_kv_pairs])
            query_position = positions[2 * self.num_kv_pairs]
            
            while query_position <= kv_positions[1]:
                remaining = [p for p in positions[2 * self.num_kv_pairs + 1:] if p > kv_positions[1]]
                if remaining:
                    query_position = random.choice(remaining)
                else:
                    query_position = max(kv_positions) + 1
                    if query_position >= self.seq_len:
                        query_position = self.seq_len - 1
                    break
            
            key_positions = []
            value_positions = []
            
            for i, (key, value) in enumerate(kv_pairs):
                key_pos = kv_positions[i * 2]
                value_pos = kv_positions[i * 2 + 1]
                key_positions.append(key_pos)
                value_positions.append(value_pos)
                input_ids[key_pos] = key
                input_ids[value_pos] = value
            
            query_kv_idx = random.randint(0, self.num_kv_pairs - 1)
            query_key = kv_pairs[query_kv_idx][0]
            input_ids[query_position] = query_key
            
            target_value = kv_pairs[query_kv_idx][1]
            target_ids[query_position] = target_value
            
            noise_positions = [p for p in positions if p not in kv_positions and p != query_position]
            for i, pos in enumerate(noise_positions[:num_noise]):
                if i < len(noise_tokens):
                    input_ids[pos] = noise_tokens[i]
            
            samples.append(ARSample(
                input_ids=input_ids,
                target_ids=target_ids,
                key_position=key_positions[query_kv_idx],
                value_position=value_positions[query_kv_idx],
                query_position=query_position,
                num_noise=num_noise,
            ))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return sample.input_ids, sample.target_ids


class MQARDataset(Dataset):
    def __init__(
        self,
        vocab_size: int = 100,
        seq_len: int = 128,
        num_samples: int = 10000,
        noise_tokens_min: int = 30,
        noise_tokens_max: int = 50,
        num_kv_pairs: int = 16,
        seed: Optional[int] = None,
    ):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.noise_tokens_min = noise_tokens_min
        self.noise_tokens_max = noise_tokens_max
        self.num_kv_pairs = num_kv_pairs
        
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Tuple[torch.Tensor, torch.Tensor, Dict]]:
        samples = []
        
        for _ in range(self.num_samples):
            num_noise = random.randint(self.noise_tokens_min, self.noise_tokens_max)
            
            input_ids = torch.zeros(self.seq_len, dtype=torch.long)
            target_ids = torch.zeros(self.seq_len, dtype=torch.long)
            
            kv_pairs = []
            key_space = list(range(1, self.vocab_size // 2))
            value_space = list(range(self.vocab_size // 2, self.vocab_size))
            random.shuffle(key_space)
            random.shuffle(value_space)
            
            for i in range(self.num_kv_pairs):
                key = key_space[i]
                value = value_space[i]
                kv_pairs.append((key, value))
            
            positions = list(range(self.seq_len))
            random.shuffle(positions)
            
            kv_positions = []
            for i, (key, value) in enumerate(kv_pairs):
                key_pos = positions[i * 2]
                value_pos = positions[i * 2 + 1]
                input_ids[key_pos] = key
                input_ids[value_pos] = value
                kv_positions.append((key_pos, value_pos))
            
            num_queries = min(self.num_kv_pairs, 8)
            query_positions = positions[2 * self.num_kv_pairs:2 * self.num_kv_pairs + num_queries]
            
            for i, query_pos in enumerate(query_positions):
                kv_idx = i % self.num_kv_pairs
                query_key = kv_pairs[kv_idx][0]
                input_ids[query_pos] = query_key
                target_ids[query_pos] = kv_pairs[kv_idx][1]
            
            noise_positions = positions[2 * self.num_kv_pairs + num_queries:]
            noise_tokens = [random.randint(1, self.vocab_size - 1) for _ in noise_positions]
            for pos, token in zip(noise_positions, noise_tokens):
                if len(noise_tokens) < num_noise:
                    break
                input_ids[pos] = token
            
            info = {
                'num_kv_pairs': self.num_kv_pairs,
                'num_queries': num_queries,
                'num_noise': num_noise,
            }
            
            samples.append((input_ids, target_ids, info))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, target_ids, _ = self.samples[idx]
        return input_ids, target_ids


class CurriculumStage:
    STAGE_0 = '0'
    STAGE_1 = '1'
    STAGE_2 = '2'
    
    @staticmethod
    def get_config(stage: str) -> Dict:
        configs = {
            CurriculumStage.STAGE_0: {
                'noise_tokens_min': 5,
                'noise_tokens_max': 15,
                'num_kv_pairs': 1,
                'seq_len': 32,
                'vocab_size': 50,
            },
            CurriculumStage.STAGE_1: {
                'noise_tokens_min': 15,
                'noise_tokens_max': 30,
                'num_kv_pairs': 4,
                'seq_len': 64,
                'vocab_size': 100,
            },
            CurriculumStage.STAGE_2: {
                'noise_tokens_min': 30,
                'noise_tokens_max': 50,
                'num_kv_pairs': 16,
                'seq_len': 128,
                'vocab_size': 200,
            },
        }
        return configs.get(stage, configs[CurriculumStage.STAGE_0])


def create_curriculum_dataloader(
    stage: str,
    batch_size: int = 32,
    num_samples: int = 10000,
    seed: Optional[int] = None,
) -> DataLoader:
    config = CurriculumStage.get_config(stage)
    
    if stage == CurriculumStage.STAGE_2:
        dataset = MQARDataset(
            **config,
            num_samples=num_samples,
            seed=seed,
        )
    else:
        dataset = AssociativeRecallDataset(
            **config,
            num_samples=num_samples,
            seed=seed,
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
