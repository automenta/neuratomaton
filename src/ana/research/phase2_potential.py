from ana.research.core import ExperimentBase, ExperimentRegistry
from ana.experiments.potential_reveal import PotentialRevealer

@ExperimentRegistry.register(phase=2, name="potential")
class PotentialExperiment(ExperimentBase):
    @property
    def name(self) -> str:
        return "potential"

    @property
    def phase(self) -> int:
        return 2

    def execute(self, quick: bool = False, sub_experiments: list = None, **kwargs):
        self.results.log(f"Starting Phase 2: Potential (Quick={quick}, SubExperiments={sub_experiments})")

        revealer = PotentialRevealer(output_dir=self.results.output_dir)

        if not sub_experiments:
            sub_experiments = ["induction", "generalization", "multiquery", "reasoning", "noise", "curriculum", "sensitivity"]

        if "induction" in sub_experiments:
            revealer.run_induction_head_experiment(quick=quick)
        if "generalization" in sub_experiments:
            revealer.run_length_generalization_experiment(quick=quick)
        if "multiquery" in sub_experiments:
            revealer.run_multi_query_experiment(quick=quick)
        if "reasoning" in sub_experiments:
            revealer.run_reasoning_experiment(quick=quick)
        if "noise" in sub_experiments:
            revealer.run_noise_robustness_experiment(quick=quick)
        if "curriculum" in sub_experiments:
            revealer.run_curriculum_experiment(quick=quick)
        if "sensitivity" in sub_experiments:
            revealer.run_sensitivity_experiment(quick=quick)

        revealer.generate_potential_report()
        self.results.log("Phase 2 Complete")
