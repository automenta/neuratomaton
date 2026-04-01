from ana.research.core import ExperimentBase, ExperimentRegistry
from ana.experiments.discovery import DiscoveryEngine

@ExperimentRegistry.register(phase=3, name="discovery")
class DiscoveryExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "discovery"

    @property
    def phase(self) -> int:
        return 3

    def execute(self, quick: bool = False, **kwargs):
        self.results.log(f"Starting Phase 3: Discovery (Quick={quick})")

        engine = DiscoveryEngine(output_dir=self.results.output_dir)
        engine.run_full_suite(quick=quick)

        self.results.log("Phase 3 Complete")
