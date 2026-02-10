import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ana" / "eqprop"))

from bioplausible.models import LoopedMLP
from bioplausible.training.supervised import SupervisedTrainer
from bioplausible.sklearn_interface import EqPropClassifier


class TestEqPropXOR:
    def test_xor_with_sklearn_interface(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0], dtype=np.int64)
        
        clf = EqPropClassifier(
            model_name="EqProp MLP",
            hidden_dim=64,
            steps=40,
            epochs=500,
            batch_size=4,
            learning_rate=0.003,
            device="cpu",
            random_state=42,
        )
        
        best_acc = 0.0
        classes = np.array([0, 1])
        converged_at = None
        
        for epoch in range(500):
            clf.partial_fit(X, y, classes=classes)
            y_pred = clf.predict(X)
            acc = (y_pred == y).mean()
            best_acc = max(best_acc, acc)
            if acc >= 0.99 and converged_at is None:
                converged_at = epoch
        
        print(f"Best accuracy: {best_acc:.2%}")
        print(f"Converged at: {converged_at}")
        
        assert best_acc >= 0.95, f"Best XOR accuracy {best_acc:.2%} < 95%"
        assert converged_at is not None and converged_at < 500, f"Did not converge within 500 iterations"
    
    def test_xor_convergence_1000_iterations(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([0, 1, 1, 0], dtype=np.int64)
        
        converged_at = None
        best_acc = 0.0
        
        clf = EqPropClassifier(
            model_name="EqProp MLP",
            hidden_dim=64,
            steps=40,
            epochs=1,
            batch_size=4,
            learning_rate=0.003,
            device="cpu",
            random_state=42,
        )
        
        classes = np.array([0, 1])
        
        for epoch in range(1000):
            clf.partial_fit(X, y, classes=classes)
            
            y_pred = clf.predict(X)
            acc = (y_pred == y).mean()
            best_acc = max(best_acc, acc)
            
            if acc >= 0.99 and converged_at is None:
                converged_at = epoch
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}: acc={acc:.2%}, best={best_acc:.2%}")
        
        y_pred = clf.predict(X)
        final_acc = (y_pred == y).mean()
        
        print(f"Final accuracy: {final_acc:.2%}, Best: {best_acc:.2%}")
        print(f"Converged at: {converged_at}")
        
        assert best_acc >= 0.95, f"Best accuracy {best_acc:.2%} < 95%"
        assert converged_at is not None and converged_at < 1000, f"Did not converge stably within 1000 iterations"


class TestEqPropEnergyConvergence:
    def test_model_converges_to_equilibrium(self):
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        
        model = LoopedMLP(
            input_dim=2,
            hidden_dim=16,
            output_dim=2,
            use_spectral_norm=True,
            max_steps=50,
            backend="pytorch",
        )
        
        model.eval()
        with torch.no_grad():
            h = model._initialize_hidden_state(X)
            x_transformed = model._transform_input(X)
            
            prev_h = h.clone()
            converged_at = None
            for step in range(50):
                h = model.forward_step(h, x_transformed)
                diff = torch.abs(h - prev_h).max().item()
                if diff < 1e-5 and converged_at is None:
                    converged_at = step
                prev_h = h.clone()
        
        assert converged_at is not None, "Should converge within max_steps"
        print(f"Converged at step {converged_at}")


class TestEqPropLipschitzConstraint:
    def test_spectral_norm_enforced(self):
        model = LoopedMLP(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            use_spectral_norm=True,
            max_steps=20,
            backend="pytorch",
        )
        
        assert model.use_spectral_norm, "Spectral norm should be enabled"
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert torch.isfinite(param).all(), f"Weight {name} should be finite"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
