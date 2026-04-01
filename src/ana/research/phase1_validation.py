from ana.research.core import ExperimentBase, ExperimentRegistry
from ana.experiments.automated_researcher import AutomatedResearcher

@ExperimentRegistry.register(phase=1, name="validation")
class ValidationExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "validation"

    @property
    def phase(self) -> int:
        return 1

    def execute(self, quick: bool = False, tune: bool = False, trials: int = 20, **kwargs):
        self.results.log(f"Starting Phase 1: Validation (Quick={quick}, Tune={tune}, Trials={trials})")

        researcher = AutomatedResearcher(output_dir=self.results.output_dir)
        researcher.run_pipeline(quick=quick, tune=tune, trials=trials)

        if researcher.status != "completed":
            raise RuntimeError(f"Validation failed with status: {researcher.status}")

        self.results.log("Phase 1 Complete")
