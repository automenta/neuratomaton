
import pytest
import os
import shutil
import sys
import torch
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from ana.cli import main, train_command
from ana.utils.datasets import HuggingFaceDataset

# Mock argparse args
class MockArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def test_huggingface_dataset_mock():
    """Test HuggingFaceDataset wrapper with mocks"""
    # Mock where it's used/imported inside the function
    with patch.dict(sys.modules, {'datasets': MagicMock(), 'transformers': MagicMock()}):

        mock_datasets = sys.modules['datasets']
        mock_transformers = sys.modules['transformers']

        # Setup mocks
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.__len__.return_value = 1
        # emulate set_format(type='torch') by returning tensors
        mock_dataset_obj.__getitem__.return_value = {'input_ids': torch.tensor([1, 2, 3, 4])}
        mock_dataset_obj.set_format = MagicMock()

        mock_raw_dataset = MagicMock()
        mock_raw_dataset.column_names = ['text']
        mock_raw_dataset.map.return_value = mock_dataset_obj

        mock_datasets.load_dataset.return_value = mock_raw_dataset

        mock_tok_instance = MagicMock()
        mock_tok_instance.vocab_size = 100
        mock_tok_instance.pad_token = None
        mock_tok_instance.pad_token_id = None
        mock_tok_instance.eos_token = '<eos>'
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok_instance

        # Init dataset
        ds = HuggingFaceDataset(dataset_name="dummy", split="train", seq_len=2)

        assert ds.vocab_size == 100
        assert len(ds) == 1

        # Get item
        x, y, mask = ds[0]
        # input_ids: [1, 2, 3, 4]
        # seq_len=2
        # x: [1, 2], y: [2, 3]
        assert torch.equal(x, torch.tensor([1, 2]))
        assert torch.equal(y, torch.tensor([2, 3]))

def test_cli_train_command_synthetic():
    """Test the train command with a synthetic dataset"""
    checkpoint_dir = "tests/cli_checkpoints"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    args = MockArgs(
        command="train",
        dataset="text_generation", # Use internal one
        hf_dataset=None,
        overrides="d_model=16,num_layers=1,epochs=1,batch_size=2",
        checkpoint_dir=checkpoint_dir
    )

    try:
        train_command(args)
    except SystemExit:
        pass # Handle potential sys.exit

    # Check if checkpoints created
    if os.path.exists(checkpoint_dir):
        assert any(f.endswith(".pt") for f in os.listdir(checkpoint_dir))
        shutil.rmtree(checkpoint_dir)
    else:
        # If directory doesn't exist, it implies failure
        pytest.fail("Checkpoint directory was not created.")

if __name__ == "__main__":
    test_huggingface_dataset_mock()
    test_cli_train_command_synthetic()
    print("All tests passed!")
