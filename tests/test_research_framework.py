import pytest
import os
import shutil
from ana.research.core import ExperimentBase, ExperimentRegistry, ResultManager, load_config_overrides
from ana.config import ANAConfig

# Dummy experiment for testing
@ExperimentRegistry.register(phase=99, name="test_experiment")
class TestExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "test_experiment"

    @property
    def phase(self) -> int:
        return 99

    def setup(self):
        self.results.log("Setup complete")

    def execute(self):
        self.results.log("Executing")
        self.results.save_json("test_results.json", {"status": "success"})
        self.results.save_report("report.md", "# Test Report")

    def teardown(self):
        self.results.log("Teardown complete")

def test_registry():
    exp_cls = ExperimentRegistry.get(99, "test_experiment")
    assert exp_cls is TestExperiment

def test_config_overrides():
    config = ANAConfig(d_model=64)
    new_config = load_config_overrides(config, "d_model=128,dropout=0.5,use_hololink=False")
    assert new_config.d_model == 128
    assert new_config.dropout == 0.5
    assert new_config.use_hololink is False

def test_experiment_run():
    config = ANAConfig()
    exp = TestExperiment(config)

    # Run experiment
    exp.run()

    # Check if results directory exists
    base_dir = exp.results.base_dir
    assert os.path.exists(base_dir)
    assert os.path.exists(os.path.join(base_dir, "experiment.log"))
    assert os.path.exists(os.path.join(base_dir, "test_results.json"))
    assert os.path.exists(os.path.join(base_dir, "report.md"))

    # Cleanup
    # Note: In a real CI environment, we might want to keep artifacts or use a temp dir.
    # Here we clean up to avoid clutter.
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

def test_result_manager():
    rm = ResultManager("test_manager", 99)
    rm.log("Test log")
    rm.save_json("data.json", {"a": 1})

    assert os.path.exists(os.path.join(rm.base_dir, "experiment.log"))
    assert os.path.exists(os.path.join(rm.base_dir, "data.json"))

    # Cleanup
    if os.path.exists(rm.base_dir):
        shutil.rmtree(rm.base_dir)
